#!/usr/bin/env python3
"""차량 상태 집계 노드 — IMU + VESC 텔레메트리를 /car_state 하나로 모은다.

역할: /imu 쿼터니언 → 출발 기준 상대 yaw(wrap-safe), /vesc/state → 속도/거리/전압/폴트.
      20Hz 스냅샷으로 /car_state 발행 (계약 레이아웃, lib.drive_cmd가 단일 진실).
입력: /imu (sensor_msgs/Imu, sensor_data QoS, 35Hz),
      /vesc/state (Float32MultiArray, 도커 ROS1 relay 경유),
      /capture_yaw (std_msgs/Empty — yaw 기준각 재캡처 트리거)
출력: /car_state (Float32MultiArray, 20Hz)
예선 대비 변경점: 신규 파일. 예선은 planner가 /imu를 직접 구독해 절대 heading만 썼음 —
      실차는 출발 기준각 캡처 + VESC 실측 속도/적산거리가 필요해 전용 노드로 분리.
      쿼터니언→yaw 수식은 예선 path_planner_node._on_imu 원문 이식.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty, Float32MultiArray

from .lib import drive_cmd


def _wrap_deg(a):
    """각도 차이를 -180~180으로 접기 (wrap-safe)."""
    return (a + 180.0) % 360.0 - 180.0


class CarStateNode(Node):
    def __init__(self):
        super().__init__("car_state_node")

        # ---------------- 파라미터 (전부 실차 실측/계약 유래) ----------------
        self.declare_parameter("rate_hz", 20.0)          # /car_state 발행 주기 (계약)
        self.declare_parameter("auto_capture", True)     # 첫 IMU 수신 시 yaw 기준각 자동 캡처
        self.declare_parameter("erpm_per_ms", 4614.0)    # eRPM = 4614 × m/s (실측검증)
        self.declare_parameter("tacho_scale", 0.0)       # tacho→m 환산. 0이면 eRPM 적분 폴백
        self.declare_parameter("low_voltage_warn", 8.6)  # NiMH 경고 임계 (실측)

        rate_hz = float(self.get_parameter("rate_hz").value)
        self._auto_capture = bool(self.get_parameter("auto_capture").value)
        self._erpm_per_ms = float(self.get_parameter("erpm_per_ms").value)
        self._tacho_scale = float(self.get_parameter("tacho_scale").value)
        self._low_voltage = float(self.get_parameter("low_voltage_warn").value)

        # ---------------- 상태 ----------------
        # yaw
        self._heading_deg = None      # 최신 절대 heading (첫 IMU 전 None)
        self._yaw_ref_deg = None      # 출발 기준각 (캡처 전 None → yaw_ref_set=0)
        self._capture_pending = self._auto_capture   # 다음 IMU에서 기준각 캡처
        self._yaw_rate_dps = 0.0
        self._last_imu_heading = None
        self._last_imu_t = None
        # vesc
        self._vesc = None             # decode_vesc_state 결과 (미수신 None)
        self._vesc_stamp = None       # 마지막 수신 시각 (sec)
        self._tacho0 = None           # 첫 수신 tacho (기준점)
        self._dist_fallback_m = 0.0   # eRPM 적분 폴백 누적거리

        # ---------------- 통신 ----------------
        self.create_subscription(Imu, "/imu", self._on_imu, qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, "/vesc/state", self._on_vesc, 10)
        self.create_subscription(Empty, "/capture_yaw", self._on_capture_yaw, 1)
        self._pub = self.create_publisher(Float32MultiArray, "/car_state", 1)

        self._dt = 1.0 / rate_hz
        self.create_timer(self._dt, self._tick)

    # --------------------------------------------------------
    # 콜백
    # --------------------------------------------------------

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_imu(self, msg: Imu):
        # 쿼터니언 → yaw: 예선 path_planner_node._on_imu 원문 이식
        q = msg.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._heading_deg = math.degrees(math.atan2(siny, cosy))

        # 기준각 캡처 (자동 1회 + /capture_yaw 재캡처)
        if self._capture_pending:
            self._yaw_ref_deg = self._heading_deg
            self._capture_pending = False
            self.get_logger().info(
                "yaw 기준각 캡처: %.1f deg" % self._yaw_ref_deg)

        # yaw_rate: 연속 IMU 차분 (wrap-safe)
        now = self._now_sec()
        if self._last_imu_heading is not None and self._last_imu_t is not None:
            dt = now - self._last_imu_t
            if 1e-4 < dt < 0.5:   # 비정상 dt(정지/점프)면 이전 값 유지
                self._yaw_rate_dps = _wrap_deg(
                    self._heading_deg - self._last_imu_heading) / dt
        self._last_imu_heading = self._heading_deg
        self._last_imu_t = now

    def _on_vesc(self, msg: Float32MultiArray):
        v = drive_cmd.decode_vesc_state(msg.data)
        self._vesc = v
        self._vesc_stamp = self._now_sec()
        if self._tacho0 is None:
            self._tacho0 = v["tacho"]
        # 저전압 경고 (5초 스로틀). 0 근처는 릴레이 초기 쓰레기값으로 보고 무시.
        if 0.5 < v["voltage"] < self._low_voltage:
            self.get_logger().warning(
                "배터리 저전압 %.2fV (< %.1fV) — 교체 필요" % (v["voltage"], self._low_voltage),
                throttle_duration_sec=5.0)

    def _on_capture_yaw(self, _msg: Empty):
        if self._heading_deg is not None:
            self._yaw_ref_deg = self._heading_deg
            self.get_logger().info("yaw 기준각 재캡처: %.1f deg" % self._yaw_ref_deg)
        else:
            # IMU 미수신 상태 — 다음 IMU에서 캡처
            self._capture_pending = True
            self.get_logger().warning("IMU 미수신 — 다음 수신 시 yaw 기준각 캡처 예약")

    # --------------------------------------------------------
    # 20Hz 발행
    # --------------------------------------------------------

    def _tick(self):
        now = self._now_sec()
        fresh = (self._vesc_stamp is not None
                 and (now - self._vesc_stamp) < drive_cmd.VESC_FRESH_SEC)

        # 속도 (미수신/스테일이어도 마지막 값 유지 — 소비측은 vesc_fresh로 판정)
        if self._vesc is not None:
            speed_ms = self._vesc["erpm"] / self._erpm_per_ms
            voltage = self._vesc["voltage"]
            fault = self._vesc["fault_code"]
        else:
            speed_ms = 0.0
            voltage = 0.0
            fault = 0.0

        # 거리: tacho_scale>0 이면 타코미터, 아니면 |speed| 적분 폴백 (fresh일 때만 적분)
        if self._tacho_scale > 0.0 and self._vesc is not None and self._tacho0 is not None:
            dist_m = (self._vesc["tacho"] - self._tacho0) * self._tacho_scale
        else:
            if fresh:
                self._dist_fallback_m += abs(speed_ms) * self._dt
            dist_m = self._dist_fallback_m

        # yaw (기준각 캡처 전에는 0 + yaw_ref_set=0)
        ref_set = self._yaw_ref_deg is not None and self._heading_deg is not None
        yaw_rel = _wrap_deg(self._heading_deg - self._yaw_ref_deg) if ref_set else 0.0

        out = Float32MultiArray()
        out.data = drive_cmd.encode_car_state({
            "yaw_rel_deg": yaw_rel,
            "yaw_rate_dps": self._yaw_rate_dps,
            "speed_ms": speed_ms,
            "dist_m": dist_m,
            "voltage": voltage,
            "fault_code": fault,
            "vesc_fresh": 1.0 if fresh else 0.0,
            "yaw_ref_set": 1.0 if ref_set else 0.0,
        })
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CarStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
