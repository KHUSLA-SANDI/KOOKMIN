#!/usr/bin/env python3
"""VESC 텔레메트리 릴레이 (도커 ROS1 noetic 안에서 실행) — vesc_msgs → std 타입 변환.

역할: /sensors/core(vesc_msgs/VescStateStamped)를 /vesc/state(Float32MultiArray)로
      변환 발행. ros1_bridge는 std 타입만 통과시키므로 이 변환이 이 파일의 존재 이유.
      옵션으로 데드맨 가드 — ROS2측 다운으로 /xycar_motor가 끊겨도 차가 굴러가는 것 방지.
입력: /sensors/core (vesc_msgs/VescStateStamped), /xycar_motor (Float32MultiArray, 데드맨용)
출력: /vesc/state (Float32MultiArray) — 계약 순서(lib/drive_cmd.py와 일치):
      [0] voltage [1] current_motor [2] erpm [3] duty [4] tacho(displacement) [5] fault_code
      데드맨 발동 시 /xycar_motor [트림상당, 0] 반복 발행.
예선 대비 변경점: 신규 파일. 예선(시뮬)에는 VESC 자체가 없었음.

설치·실행 (도커 ROS1 noetic 컨테이너, py3):
  1) 호스트: docker cp vesc_relay.py <컨테이너>:/root/vesc_relay.py
  2) 접속:   docker exec -it <컨테이너> bash
  3) 환경:   source /opt/ros/noetic/setup.bash
             source /root/catkin_ws/devel/setup.bash   # vesc_msgs 포함 워크스페이스
             export ROS_MASTER_URI=http://localhost:11311   # 마스터는 컨테이너 로컬
  4) 실행:   python3 /root/vesc_relay.py
     데드맨 켜기: python3 /root/vesc_relay.py _deadman_enable:=true
"""

import rospy
from std_msgs.msg import Float32MultiArray
from vesc_msgs.msg import VescStateStamped

# /vesc/state 계약 레이아웃 (ROS2측 lib/drive_cmd.py와 반드시 일치)
VESC_STATE_LEN = 6


class VescRelay(object):
    def __init__(self):
        # ---------------- 파라미터 ----------------
        self.deadman_enable = rospy.get_param("~deadman_enable", False)
        self.deadman_timeout = rospy.get_param("~deadman_timeout", 0.4)  # /xycar_motor 침묵 판정
        # 실제 영점 오프셋은 하위 xycar_motor 노드가 일괄 적용한다.
        self.steer_trim = rospy.get_param("~steer_trim", 0.0)
        self.tick_hz = rospy.get_param("~tick_hz", 20.0)                 # 데드맨 검사/발행 주기

        # ---------------- 상태 (데드맨) ----------------
        self.last_cmd_t = None      # 마지막 외부 /xycar_motor 수신 시각
        self.last_speed = 0.0       # 마지막 외부 명령 speed
        self.deadman_active = False

        # ---------------- 통신 ----------------
        self.pub_state = rospy.Publisher("/vesc/state", Float32MultiArray, queue_size=1)
        rospy.Subscriber("/sensors/core", VescStateStamped, self.on_core, queue_size=1)
        if self.deadman_enable:
            self.pub_motor = rospy.Publisher("/xycar_motor", Float32MultiArray, queue_size=1)
            rospy.Subscriber("/xycar_motor", Float32MultiArray, self.on_motor, queue_size=1)
            rospy.Timer(rospy.Duration(1.0 / self.tick_hz), self.on_tick)
            rospy.loginfo("vesc_relay: 데드맨 가드 활성 (timeout %.2fs)", self.deadman_timeout)
        rospy.loginfo("vesc_relay: /sensors/core -> /vesc/state 릴레이 시작")

    # --------------------------------------------------------
    # /sensors/core → /vesc/state
    # --------------------------------------------------------

    def on_core(self, msg):
        s = msg.state
        out = Float32MultiArray()
        out.data = [
            float(s.voltage_input),      # [0] voltage
            float(s.current_motor),      # [1] current_motor
            float(s.speed),              # [2] erpm
            float(s.duty_cycle),         # [3] duty
            float(s.displacement),       # [4] tacho (displacement)
            float(s.fault_code),         # [5] fault_code
        ]
        self.pub_state.publish(out)

    # --------------------------------------------------------
    # 데드맨 가드 — /xycar_motor 0.4s 침묵 + 최근 speed≠0 이면 정지 반복 발행
    # --------------------------------------------------------

    def on_motor(self, msg):
        if len(msg.data) < 2:
            return
        angle = float(msg.data[0])
        speed = float(msg.data[1])
        # 데드맨 발동 중 자기 자신이 발행한 정지 명령(또는 동등한 정지)의 에코는
        # 신선도 갱신에서 제외 — 안 그러면 자기 에코로 침묵 판정이 풀려 반복 발행이 끊긴다.
        if self.deadman_active and speed == 0.0 and angle == float(self.steer_trim):
            return
        self.last_cmd_t = rospy.get_time()
        self.last_speed = speed
        self.deadman_active = False

    def on_tick(self, _event):
        if self.last_cmd_t is None:
            return   # 외부 명령을 한 번도 못 봄 — 개입하지 않음
        silent = (rospy.get_time() - self.last_cmd_t) > self.deadman_timeout
        if silent and (self.deadman_active or self.last_speed != 0.0):
            if not self.deadman_active:
                rospy.logwarn("vesc_relay 데드맨 발동: /xycar_motor %.1fs 침묵 — 정지 발행",
                              self.deadman_timeout)
            self.deadman_active = True
            stop = Float32MultiArray()
            stop.data = [float(self.steer_trim), 0.0]   # 트림상당 조향 + 정지
            self.pub_motor.publish(stop)


def main():
    rospy.init_node("vesc_relay")
    VescRelay()
    rospy.spin()


if __name__ == "__main__":
    main()
