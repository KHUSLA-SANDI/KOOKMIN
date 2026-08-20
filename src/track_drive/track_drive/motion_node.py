#!/usr/bin/env python3
"""모터 제어 노드(실차) — /center_path 섀도우 추종 + /drive_cmd 출력 게이트 → /xycar_motor.

역할: LANE 다점 추종(가우시안 focus 가중 PP) 조향·속도를 경로가 있으면 owner와 무관하게
  매 틱 섀도우 계산(내부 상태 갱신 유지)하고, 최종 출력만 /drive_cmd owner로 게이트.
  트림/기계스톱/데드존/램프 등 실차 변환은 CarInterface가 전담.
입력: /center_path(PoseArray), /drive_cmd(Float32MultiArray), /teleop_cmd(Float32MultiArray),
  /car_state(Float32MultiArray)
출력: /xycar_motor(Float32MultiArray [angle, speed], 절대토픽, 20Hz 연속),
  /debug/arbitration(String, 1Hz)
예선 대비 변경점: XycarMotor → Float32MultiArray. 12개 플래그 구독·/target 구독 제거 →
  /drive_cmd 스냅샷 + arbitration 중재로 대체(버그 2·3·4·5·9 대응). lane-lost ±100 복구
  삭제 → 직전각 유지 + 1초 선형 감속(버그 1 대응). 전환 시 스무딩 기준각은
  CarInterface.last_angle_out(유일 진실원)에서 재시드(버그 6 대응). 상수는 전부
  declare_parameter — 시뮬 유래 값은 # [SIM] 표시.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float32MultiArray, String

from .lib import drive_cmd as dc
from .lib.arbitration import arbitrate_speed, detect_transition
from .lib.car_interface import CarInterface, DEFAULT_CFG

# ======================== 계약 상수 (튜닝 대상 아님) ========================

ANGLE_MIN = -100.0        # 논리 조향 스케일 (CarInterface 입력 계약)
ANGLE_MAX = 100.0
SPEED_CMD_MAX = 100.0     # 속도 명령단위 상한 (×0.08 = m/s)
DEBUG_HZ = 1.0            # /debug/arbitration 발행 주기


# ======================== ROS 노드 ========================

class MotionNode(Node):
    def __init__(self):
        super().__init__("motion_node")

        # ---------------- 파라미터 (declare + load) ----------------
        p = self._prm

        self.CONTROL_HZ = float(p("control_hz", 20.0))  # 계약: 20Hz 연속 발행

        # --- 다점 추종 게인: 조향 lookahead 주변 점에 가우시안 가중 ---
        self.STRAIGHT_PURSUIT_GAIN = float(p("straight_pursuit_gain", 0.08))  # [SIM]
        self.STRAIGHT_HEADING_GAIN = float(p("straight_heading_gain", 0.08))  # [SIM]
        self.STRAIGHT_BOOST_PURSUIT_GAIN = float(p("straight_boost_pursuit_gain", 0.40))  # [SIM]
        self.STRAIGHT_BOOST_HEADING_GAIN = float(p("straight_boost_heading_gain", 0.40))  # [SIM]
        self.CURVE_PURSUIT_GAIN = float(p("curve_pursuit_gain", 1.62))  # [SIM]
        self.CURVE_HEADING_GAIN = float(p("curve_heading_gain", 2.22))  # [SIM]
        self.S_STEER_FOCUS_X = float(p("s_steer_focus_x", 3.7))  # [SIM] S_ZONE 전용 lookahead
        self.STEER_FOCUS_SIGMA = float(p("steer_focus_sigma", 1.8))  # [SIM] 가중 폭 (m)
        # 평상시 속도비례 lookahead: clip(BASE + K*직전속도, MIN, MAX) — 빠를수록 멀리
        self.STEER_FOCUS_BASE = float(p("steer_focus_base", 3.8))  # [SIM]
        self.STEER_FOCUS_K = float(p("steer_focus_k", 0.055))  # [SIM]
        self.STEER_FOCUS_MIN = float(p("steer_focus_min", 4.2))  # [SIM]
        self.STEER_FOCUS_MAX = float(p("steer_focus_max", 5.0))  # [SIM]
        self.STEER_GAIN_DISTANCE_M = float(p("steer_gain_distance_m", 2.4))  # [SIM]
        self.STEER_PREVIEW_DISTANCE_M = float(p("steer_preview_distance_m", 5.5))  # [SIM]
        self.STEER_HEADING_START_DEG = float(p("steer_heading_start_deg", 4.0))  # [SIM]
        self.STEER_HEADING_FULL_DEG = float(p("steer_heading_full_deg", 20.0))  # [SIM]
        self.STEER_CURVE_SWITCH_RATIO = float(p("steer_curve_switch_ratio", 0.50))  # [SIM]

        # --- 직진 부스트 ---
        self.STRAIGHT_BOOST_ON_DEG = float(p("straight_boost_on_deg", 5.0))  # [SIM]
        self.STRAIGHT_BOOST_FULL_DEG = float(p("straight_boost_full_deg", 10.0))  # [SIM]
        self.STRAIGHT_BOOST_OFF_DEG = float(p("straight_boost_off_deg", 0.0))  # [SIM]
        self.STRAIGHT_BOOST_MAX_LEVEL = float(p("straight_boost_max_level", 1.8))  # [SIM]
        self.STRAIGHT_BOOST_BLEND_ALPHA = float(p("straight_boost_blend_alpha", 0.45))  # [SIM]
        # 부스트 허용 최대 곡률비 = 커브 전환비 (예선과 동일하게 묶음)
        self.STRAIGHT_BOOST_MAX_CURVE_RATIO = self.STEER_CURVE_SWITCH_RATIO

        # --- corner-exit unwind ---
        self.CORNER_EXIT_UNWIND_START_RATIO = float(p("corner_exit_unwind_start_ratio", 0.45))  # [SIM]
        self.CORNER_EXIT_UNWIND_END_RATIO = float(p("corner_exit_unwind_end_ratio", 0.22))  # [SIM]
        self.CORNER_EXIT_UNWIND_FRAMES = int(p("corner_exit_unwind_frames", 5))  # [SIM]
        self.CORNER_EXIT_UNWIND_SCALE = float(p("corner_exit_unwind_scale", 0.80))  # [SIM]
        # 전환 직후 unwind 트리거 금지 게이트 (신규 — 전환 리셋 규칙의 일부)
        self.UNWIND_BLOCK_TICKS = int(p("unwind_block_ticks", 5))

        # --- 조향 스무딩 (IIR + 스텝 + 데드밴드) ---
        self.ANGLE_SMOOTH_ALPHA = float(p("angle_smooth_alpha", 0.55))  # [SIM]
        self.ANGLE_MAX_STEP = float(p("angle_max_step", 12.0))  # [SIM]
        self.ANGLE_DEADBAND = float(p("angle_deadband", 0.6))  # [SIM]

        # --- steer_profile 반영 ---
        self.CONE_STEER_GAIN = float(p("cone_steer_gain", 1.12))  # [SIM] CONE 조향 보정

        # --- 속도 (각도 cap + preview cap + 코너 inside floor) ---
        self.SPEED_DEFAULT = float(p("speed_default", 24.0))  # [SIM]
        self.SPEED_CURVE_BASE = float(p("speed_curve_base", 20.0))  # [SIM] 직진미확정 cap 겸용
        self.SPEED_MIN = float(p("speed_min", 3.0))  # [SIM] _speed_from_angle 내부 최저 (최종 하한은 speed_min_drive)
        self.SPEED_SLOW_START_DEG = float(p("speed_slow_start_deg", 12.0))  # [SIM]
        self.SPEED_SLOW_FULL_DEG = float(p("speed_slow_full_deg", 40.0))  # [SIM]
        self.PREVIEW_SLOW_SPEED = float(p("preview_slow_speed", 4.5))  # [SIM]
        self.STRAIGHT_CONFIRM_TICKS = int(p("straight_confirm_ticks", 20))  # [SIM] 직진 확정까지 틱 수
        # 예선 _map_kmh_to_speed_cmd 전개값: kmh × 4.5/5.0 (# [SIM] 환산계수 자체가 시뮬 실측)
        self.CORNER_INSIDE_SPEED = float(p("corner_inside_speed", 7.2))  # [SIM] = 8.0km/h × 4.5/5.0
        self.CORNER_BOOST_SPEED = float(p("corner_boost_speed", 9.0))  # [SIM] = 10.0km/h × 4.5/5.0
        self.CORNER_ENTER_RATIO = float(p("corner_enter_ratio", 0.55))  # [SIM]
        self.CORNER_HOLD_FRAMES = int(p("corner_hold_frames", 4))  # [SIM]
        self.CORNER_EXIT_RATIO = float(p("corner_exit_ratio", 0.35))  # [SIM]
        self.CORNER_EXIT_ANGLE_SCALE = float(p("corner_exit_angle_scale", 0.60))  # [SIM]
        self.CORNER_INSIDE_ANGLE_SCALE = float(p("corner_inside_angle_scale", 0.82))  # [SIM]
        self.PREVIEW_CURV_START = float(p("preview_curv_start", 0.018))  # [SIM]
        self.PREVIEW_CURV_FULL = float(p("preview_curv_full", 0.080))  # [SIM]
        self.PREVIEW_LATERAL_START_M = float(p("preview_lateral_start_m", 0.45))  # [SIM]
        self.PREVIEW_LATERAL_FULL_M = float(p("preview_lateral_full_m", 1.35))  # [SIM]
        self.PREVIEW_DISTANCE_M = float(p("preview_distance_m", 5.0))  # [SIM]

        # --- 경로 세그먼트 ---
        self.TARGET_Y_LIMIT = float(p("target_y_limit", 4.0))  # [SIM]
        self.LOOKAHEAD_X_MIN = float(p("lookahead_x_min", 1.0))  # [SIM]
        self.LOOKAHEAD_X_MAX = float(p("lookahead_x_max", 10.0))  # [SIM]
        self.REANCHOR_FAR_M = float(p("reanchor_far_m", 0.65))  # [SIM]

        # --- lane-lost (실차 정책: 직전각 유지 + 선형 감속, ±100 복구 금지) ---
        self.PATH_STALE_SEC = float(p("path_stale_sec", 0.5))  # [SIM] 예선 PATH_MEMORY_TIMEOUT_S
        self.LANE_LOST_DECEL_SEC = float(p("lane_lost_decel_sec", 1.0))  # 1초에 걸쳐 0으로

        # --- BLOCK 실행값 (종료 판정은 planner 몫 — motion은 실행만) ---
        self.BLOCK_STEER_ANGLE = float(p("block_steer_angle", -100.0))  # [SIM] 좌회전 하드코딩 조향
        self.BLOCK_SPEED = float(p("block_speed", 10.0))  # [SIM] 좌회전 속도

        # --- 중재 하한 (실차: 데드존 4 미만은 안 구름 — 버그 7 대응) ---
        self.SPEED_MIN_DRIVE = float(p("speed_min_drive", 5.0))

        # --- car.yaml 키 (실차 실측 — CarInterface cfg, 시뮬 값 아님) ---
        car_cfg = {key: p(key, DEFAULT_CFG[key]) for key in sorted(DEFAULT_CFG)}
        self._car = CarInterface(car_cfg)
        # 후진 허용을 '자율' / '수동(teleop)' 으로 나눠 둔다 — 매 틱 _car.allow_reverse 를
        # 이 둘 중 하나로 갈아끼운다. car.yaml 의 allow_reverse 는 자율 쪽 값으로 쓴다.
        self._auto_allow_reverse = bool(self._car.allow_reverse)
        self.declare_parameter("teleop_allow_reverse", True)
        self._teleop_allow_reverse = bool(
            self.get_parameter("teleop_allow_reverse").value)

        # ---------------- 내부 상태 ----------------
        # 경로
        self._path_xs = None
        self._path_ys = None
        self._path_stamp = None

        # 최신 스냅샷 (미수신 초기값: drive_cmd는 워치독이 MISSION_STOP 취급)
        self._cmd = dc.decode_drive_cmd([])
        self._cmd_stamp = None
        self._teleop = None
        self._teleop_stamp = None
        self._cs = dc.decode_car_state([])

        # 섀도우 LANE 상태 (예선 보존)
        self._prev_angle = 0.0
        self._corner_frames = 0
        self._straight_ticks = 0        # 직진 지속 카운터 (확정돼야 풀가속 허용)
        self._prev_steer_ratio = 0.0
        self._exit_unwind_frames = 0
        self._straight_boost_level = 0.0
        self._last_speed = 0.0          # 직전 틱 주행속도 (속도비례 lookahead용)
        # 예선 /corner_boost 플래그 제거 — 실차에선 항상 False (어린이보호 후 부스트는 planner가 floor로 대체)
        self._corner_boost = False

        # 전환/게이트 상태
        self._prev_owner = int(dc.Owner.MISSION_STOP)   # 워치독 초기 상태와 일치
        self._prev_block_id = 0
        self._unwind_block_ticks = 0
        self._teleop_was_active = False

        # lane-lost 감속 상태
        self._lane_lost_ticks = 0
        self._lane_lost_speed0 = 0.0

        # 출력 기록 (논리 명령단)
        self._last_angle_cmd = 0.0
        self._last_speed_cmd = 0.0

        # 디버그 스냅샷 (1Hz 발행용)
        self._dbg = {}

        # ---------------- 배선 ----------------
        self.create_subscription(PoseArray, "/center_path", self._on_path, 10)
        self.create_subscription(Float32MultiArray, "/drive_cmd", self._on_drive_cmd, 10)
        self.create_subscription(Float32MultiArray, "/teleop_cmd", self._on_teleop, 10)
        self.create_subscription(Float32MultiArray, "/car_state", self._on_car_state, 10)
        self._pub_motor = self.create_publisher(Float32MultiArray, "/xycar_motor", 10)
        self._pub_dbg = self.create_publisher(String, "/debug/arbitration", 10)
        self.create_timer(1.0 / self.CONTROL_HZ, self._tick)
        self.create_timer(1.0 / DEBUG_HZ, self._tick_debug)

        self.get_logger().info("motion_node up (realcar)")

    def _prm(self, name, default):
        """declare_parameter + 즉시 로드 (tuning.yaml 추출 대상)."""
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    # ==================== 콜백 (스냅샷 저장만) ====================

    def _on_path(self, msg: PoseArray):
        if not msg.poses:
            return
        xs = np.array([q.position.x for q in msg.poses], dtype=np.float64)
        ys = np.array([q.position.y for q in msg.poses], dtype=np.float64)
        valid = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[valid]
        ys = ys[valid]
        if xs.size < 2:
            return
        order = np.argsort(xs)
        self._path_xs = xs[order]
        self._path_ys = ys[order]
        self._path_stamp = self.get_clock().now()

    def _on_drive_cmd(self, msg: Float32MultiArray):
        self._cmd = dc.decode_drive_cmd(msg.data)
        self._cmd_stamp = self.get_clock().now()

    def _on_teleop(self, msg: Float32MultiArray):
        self._teleop = dc.decode_teleop_cmd(msg.data)
        self._teleop_stamp = self.get_clock().now()

    def _on_car_state(self, msg: Float32MultiArray):
        self._cs = dc.decode_car_state(msg.data)

    # ==================== 20Hz 틱 = 섀도우 계산 + 출력 게이트 ====================

    def _age(self, stamp, now):
        if stamp is None:
            return float("inf")
        return (now - stamp).nanoseconds * 1e-9

    def _tick(self):
        now = self.get_clock().now()
        cmd = self._cmd

        # --- /drive_cmd 워치독: 0.3s 미수신 → MISSION_STOP 취급 (버그 3 대응) ---
        cmd_stale = self._age(self._cmd_stamp, now) > dc.DRIVE_CMD_STALE_SEC
        owner = dc.Owner.MISSION_STOP if cmd_stale else cmd["owner"]
        profile = cmd["steer_profile"]

        # --- teleop 신선도 + 수동→자율 복귀 리셋 (제어권 재장악 = 전환과 동일 취급) ---
        tele = self._teleop
        tele_fresh = (tele is not None
                      and self._age(self._teleop_stamp, now) <= dc.TELEOP_STALE_SEC)
        if self._teleop_was_active and not tele_fresh:
            self._transition_reset()
        self._teleop_was_active = tele_fresh

        # --- owner/block_id 전환 리셋 (버그 1·6 대응) ---
        if detect_transition(self._prev_owner, owner,
                             self._prev_block_id, cmd["block_id"]):
            self._transition_reset()
        self._prev_owner = int(owner)
        self._prev_block_id = cmd["block_id"]

        # --- 섀도우 LANE 계산: 경로가 신선하면 owner와 무관하게 매 틱 수행 ---
        shadow_ok = False
        lane_angle = self._prev_angle
        v_base = 0.0
        in_corner = False
        path_fresh = (self._path_xs is not None and self._path_xs.size >= 2
                      and self._age(self._path_stamp, now) <= self.PATH_STALE_SEC)
        if path_fresh:
            seg_xs, seg_ys, seg_s = self._reanchored_segment(self._path_xs, self._path_ys)
            if seg_xs.size >= 2:
                shadow_ok = True
                lane_angle, v_base, in_corner = self._shadow_lane(
                    seg_xs, seg_ys, seg_s, profile)

        # --- 출력 게이트: owner가 최종 출력 결정 ---
        stop_owner = owner in (dc.Owner.MISSION_STOP, dc.Owner.SAFETY_STOP)
        # 내부 cap: 직진미확정이면 SPEED_CURVE_BASE (예선 직진 확정 게이트의 cap화)
        caps = [
            cmd["speed_cap"],
            self.SPEED_CURVE_BASE if self._straight_ticks < self.STRAIGHT_CONFIRM_TICKS else 0.0,
        ]
        # floor: 코너 inside floor(코너일 때만) + 데드존 초과 최소 주행 명령
        #        + planner speed_floor (<=0은 arbitrate_speed가 무시)
        inside_floor = self.CORNER_BOOST_SPEED if self._corner_boost else self.CORNER_INSIDE_SPEED
        floors = [
            inside_floor if in_corner else 0.0,
            self.SPEED_MIN_DRIVE,
            cmd["speed_floor"],
        ]

        if owner == dc.Owner.BLOCK:
            # 실행만 — 종료 판정은 planner 몫 (owner가 LANE으로 돌아옴)
            angle_cmd = self.BLOCK_STEER_ANGLE
            # BLOCK 속도도 planner cap에 종속 (하드코딩 속도가 cap을 뚫지 않게)
            speed_cmd = arbitrate_speed(self.BLOCK_SPEED, caps=[cmd["speed_cap"]],
                                        floors=[], fixed=0.0, stop=False)
            self._reset_lane_lost()
        elif stop_owner:
            # 감속 램프는 CarInterface slew 담당. angle은 마지막 발행각 유지.
            angle_cmd = self._last_angle_cmd
            speed_cmd = arbitrate_speed(v_base, caps=caps, floors=floors,
                                        fixed=cmd["speed_fixed"], stop=True)  # = 0.0
            self._reset_lane_lost()
        else:  # LANE
            if shadow_ok:
                self._reset_lane_lost()
                angle_cmd = lane_angle
                speed_cmd = arbitrate_speed(v_base, caps=caps, floors=floors,
                                            fixed=cmd["speed_fixed"], stop=False)
            else:
                angle_cmd, speed_cmd = self._lane_lost_cmd()

        # --- teleop mux: 신선하면 자율 출력 대신 teleop 값 (estop이면 0) ---
        if tele_fresh:
            if tele["estop"]:
                angle_cmd, speed_cmd = 0.0, 0.0
            else:
                angle_cmd, speed_cmd = tele["angle"], tele["speed"]

        # --- 후진 허용은 '누가 몰고 있나'에 달렸다 ---
        # car.yaml 의 allow_reverse:false 는 "자율주행 경로는 후진 금지"라는 뜻이지
        # 수동 조종까지 막으려던 게 아니다. 그런데 CarInterface 는 플래그 하나뿐이라
        # mux 경유 teleop 의 음수 속도까지 0 으로 잘려 후진이 안 됐다(실측 확인).
        # 그래서 teleop 이 잡고 있는 동안만 teleop_allow_reverse 를 따른다.
        self._car.allow_reverse = (self._teleop_allow_reverse if tele_fresh
                                   else self._auto_allow_reverse)

        # --- 실차 변환 + 발행 (20Hz 연속) ---
        angle_out, speed_out = self._car.to_motor(angle_cmd, speed_cmd)
        out = Float32MultiArray()
        out.data = [float(angle_out), float(speed_out)]
        self._pub_motor.publish(out)

        self._last_angle_cmd = float(angle_cmd)
        self._last_speed_cmd = float(speed_cmd)
        # 속도비례 lookahead용 직전 속도 — LANE 자율 주행 중엔 명령값, 그 외엔 실측값
        if owner == dc.Owner.LANE and shadow_ok and not tele_fresh:
            self._last_speed = float(speed_cmd)
        else:
            self._last_speed = self._car.ms_to_cmd(self._cs["speed_ms"])

        # --- 디버그 스냅샷 (1Hz 발행용) ---
        self._dbg = {
            "owner": owner.name,
            "profile": profile.name,
            "cmd_stale": cmd_stale,
            "cap_cmd": cmd["speed_cap"],
            "cap_straight": caps[1],
            "floor_corner": floors[0],
            "floor_min": floors[1],
            "floor_cmd": cmd["speed_floor"],
            "fixed": cmd["speed_fixed"],
            "v_base": v_base,
            "angle_cmd": angle_cmd,
            "speed_cmd": speed_cmd,
            "angle_out": angle_out,
            "speed_out": speed_out,
            "teleop": tele_fresh,
            "shadow": shadow_ok,
            "lost_ticks": self._lane_lost_ticks,
            "lane_offset": cmd["lane_offset"],
        }

    def _shadow_lane(self, seg_xs, seg_ys, seg_s, profile):
        """LANE 조향·속도 섀도우 계산 — 예선 _tick 본체 보존 (플래그 로직만 제거)."""
        preview_ratio = self._preview_slow_ratio(seg_xs, seg_ys)
        near_steer_ratio, line_angle_error = self._steer_gain_ratio(
            seg_xs, seg_ys, seg_s, self.STEER_GAIN_DISTANCE_M)
        preview_steer_ratio, _ = self._steer_gain_ratio(
            seg_xs, seg_ys, seg_s, self.STEER_PREVIEW_DISTANCE_M)
        steer_ratio = max(near_steer_ratio, preview_steer_ratio)

        # in_corner 히스테리시스 (예선 보존)
        if preview_ratio >= self.CORNER_ENTER_RATIO:
            self._corner_frames = min(self.CORNER_HOLD_FRAMES, self._corner_frames + 1)
        elif preview_ratio <= self.CORNER_EXIT_RATIO:
            self._corner_frames = 0
        in_corner = self._corner_frames >= self.CORNER_HOLD_FRAMES

        # 직진 부스트 (예선 보존)
        if steer_ratio <= self.STRAIGHT_BOOST_MAX_CURVE_RATIO:
            boost_target = np.clip(
                (line_angle_error - self.STRAIGHT_BOOST_OFF_DEG)
                / max(self.STRAIGHT_BOOST_ON_DEG - self.STRAIGHT_BOOST_OFF_DEG, 1e-6),
                0.0, 1.0)
            boost_target = boost_target * boost_target
            if line_angle_error > self.STRAIGHT_BOOST_ON_DEG:
                extra_boost = np.clip(
                    (line_angle_error - self.STRAIGHT_BOOST_ON_DEG)
                    / max(self.STRAIGHT_BOOST_FULL_DEG - self.STRAIGHT_BOOST_ON_DEG, 1e-6),
                    0.0, 1.0)
                boost_target = 1.0 + (
                    (self.STRAIGHT_BOOST_MAX_LEVEL - 1.0) * extra_boost)
        else:
            boost_target = 0.0
        self._straight_boost_level += (
            self.STRAIGHT_BOOST_BLEND_ALPHA
            * (float(boost_target) - self._straight_boost_level)
        )

        # corner-exit unwind 트리거 — 전환 직후 UNWIND_BLOCK_TICKS 동안 트리거 금지 (신규 게이트)
        if self._unwind_block_ticks > 0:
            self._unwind_block_ticks -= 1
        elif (
            self._prev_steer_ratio >= self.CORNER_EXIT_UNWIND_START_RATIO
            and near_steer_ratio <= self.CORNER_EXIT_UNWIND_END_RATIO
        ):
            self._exit_unwind_frames = self.CORNER_EXIT_UNWIND_FRAMES
        elif self._exit_unwind_frames > 0:
            self._exit_unwind_frames -= 1

        angle = self._compute_steering(
            seg_xs, seg_ys, seg_s, steer_ratio, self._straight_boost_level, profile)
        # steer_profile CONE: 조향 보정 (owner 이전 없이 LANE 계산에 반영)
        if profile == dc.SteerProfile.CONE:
            angle = float(np.clip(angle * self.CONE_STEER_GAIN, ANGLE_MIN, ANGLE_MAX))
        angle = self._apply_corner_exit_unwind(angle)
        angle = self._smooth_angle(angle)
        self._prev_steer_ratio = near_steer_ratio

        v_base = self._speed_from_angle(angle, seg_xs, seg_ys, preview_ratio, in_corner)

        # 직진 확정 카운터 (예선 보존 — cap 적용 자체는 중재부의 내부 cap으로)
        if (in_corner or preview_ratio >= self.CORNER_EXIT_RATIO
                or abs(angle) > self.SPEED_SLOW_START_DEG):
            self._straight_ticks = 0
        else:
            self._straight_ticks += 1

        return angle, v_base, in_corner

    # ==================== 전환/소실 처리 ====================

    def _transition_reset(self):
        """owner/block_id 전환 시 리셋 — 스무딩 기준각은 마지막 실제 발행각(유일 진실원)."""
        lao = self._car.last_angle_out
        if lao is None:
            self._prev_angle = 0.0
        else:
            # 기계 발행각 → 논리각 역변환 (좌우별 스케일 역산)
            self._prev_angle = float(np.clip(
                self._car.steering_logical(lao),
                ANGLE_MIN, ANGLE_MAX))
        self._corner_frames = 0
        self._straight_ticks = 0
        self._straight_boost_level = 0.0
        self._prev_steer_ratio = 0.0
        self._exit_unwind_frames = 0
        self._unwind_block_ticks = self.UNWIND_BLOCK_TICKS   # unwind 트리거 금지 게이트
        self._last_speed = self._car.ms_to_cmd(self._cs["speed_ms"])  # 실측 속도로 재시드

    def _reset_lane_lost(self):
        self._lane_lost_ticks = 0
        self._lane_lost_speed0 = 0.0

    def _lane_lost_cmd(self):
        """경로 소실(LANE 중): 직전각 유지 + 선형 감속(LANE_LOST_DECEL_SEC에 걸쳐 0).

        예선의 ±100 풀조향 복구·_prev_angle 직접 오염(버그 1) 재도입 금지.
        /target 신선도도 사용하지 않는다(버그 5).
        """
        if self._lane_lost_ticks == 0:
            self._lane_lost_speed0 = max(self._last_speed_cmd, 0.0)
        self._lane_lost_ticks += 1
        total_ticks = max(self.LANE_LOST_DECEL_SEC * self.CONTROL_HZ, 1.0)
        ratio = max(0.0, 1.0 - self._lane_lost_ticks / total_ticks)
        return self._last_angle_cmd, self._lane_lost_speed0 * ratio

    # ==================== 예선 보존 함수들 ====================

    @staticmethod
    def _path_distance(xs, ys):
        if xs.size == 0:
            return np.array([], dtype=np.float64)
        seg = np.hypot(np.diff(xs), np.diff(ys))
        return np.concatenate(([0.0], np.cumsum(seg)))

    def _reanchored_segment(self, xs, ys):
        xs = np.asarray(xs, dtype=np.float64)
        valid = np.isfinite(xs) & np.isfinite(ys) & (xs >= 0.0)
        if int(np.count_nonzero(valid)) < 2:
            valid = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[valid]
        ys = ys[valid]
        if xs.size < 2:
            return xs, ys, np.zeros_like(xs)

        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        keep = np.concatenate(([True], np.diff(xs) > 1e-4))
        xs = xs[keep]
        ys = ys[keep]
        if xs.size < 2:
            return xs, ys, np.zeros_like(xs)

        anchor_idx = int(np.argmin(xs * xs + ys * ys))
        anchor_dist = math.hypot(float(xs[anchor_idx]), float(ys[anchor_idx]))
        dist = self._path_distance(xs, ys)
        rel_s = dist - float(dist[anchor_idx])

        # 경로에서 멀리 떨어졌으면 가까운 점에서 다시 붙고, 아니면 앞쪽 preview만 쓴다.
        min_s = 0.0 if anchor_dist > self.REANCHOR_FAR_M else self.LOOKAHEAD_X_MIN
        mask = (rel_s >= min_s) & (rel_s <= self.LOOKAHEAD_X_MAX)
        if int(np.count_nonzero(mask)) < 2:
            mask = rel_s >= 0.0
        return xs[mask], ys[mask], rel_s[mask]

    def _compute_steering(self, xs, ys, rel_s, steer_ratio,
                          straight_boost_level=0.0,
                          profile=dc.SteerProfile.NORMAL):
        if xs.size < 2:
            return 0.0

        ys = np.clip(ys, -self.TARGET_Y_LIMIT, self.TARGET_Y_LIMIT)
        pursuit_angles = np.degrees(np.arctan2(-ys, np.maximum(xs, 1e-3)))

        if xs.size >= 2:
            slopes = np.gradient(ys, xs)
        else:
            slopes = np.zeros_like(xs)
        heading_angles = np.degrees(np.arctan(-slopes))

        if steer_ratio >= self.STEER_CURVE_SWITCH_RATIO:
            pursuit_gain = self.CURVE_PURSUIT_GAIN
            heading_gain = self.CURVE_HEADING_GAIN
        else:
            pursuit_gain = self.STRAIGHT_PURSUIT_GAIN
            heading_gain = self.STRAIGHT_HEADING_GAIN
            boost_mix = float(np.clip(straight_boost_level, 0.0,
                                      self.STRAIGHT_BOOST_MAX_LEVEL))
            pursuit_gain += (
                self.STRAIGHT_BOOST_PURSUIT_GAIN - self.STRAIGHT_PURSUIT_GAIN
            ) * boost_mix
            heading_gain += (
                self.STRAIGHT_BOOST_HEADING_GAIN - self.STRAIGHT_HEADING_GAIN
            ) * boost_mix

        sample_angles = (
            pursuit_gain * pursuit_angles +
            heading_gain * heading_angles
        )
        focus = rel_s if rel_s is not None and rel_s.size == xs.size else xs
        # lookahead 가중 peak: S_ZONE 프로파일 전용 > 평상시 속도비례
        # (예선의 튜닝 override·추월 lookahead 분기는 구독 삭제와 함께 제거)
        if profile == dc.SteerProfile.S_ZONE:
            focus_x = self.S_STEER_FOCUS_X   # S자 전용
        else:
            # 속도비례 lookahead — 빠를수록 멀리 봐서 위빙 방지
            focus_x = float(np.clip(
                self.STEER_FOCUS_BASE + self.STEER_FOCUS_K * self._last_speed,
                self.STEER_FOCUS_MIN, self.STEER_FOCUS_MAX))
        weights = np.exp(-((focus - focus_x) / self.STEER_FOCUS_SIGMA) ** 2)
        angle = float(np.average(sample_angles, weights=weights))
        return float(np.clip(angle, ANGLE_MIN, ANGLE_MAX))

    def _apply_corner_exit_unwind(self, angle):
        if self._exit_unwind_frames <= 0:
            return angle
        if self._prev_angle == 0.0 or angle == 0.0:
            return angle
        if math.copysign(1.0, angle) != math.copysign(1.0, self._prev_angle):
            return angle
        max_mag = min(abs(angle), abs(self._prev_angle))
        angle = math.copysign(max_mag * self.CORNER_EXIT_UNWIND_SCALE, angle)
        return float(np.clip(angle, ANGLE_MIN, ANGLE_MAX))

    def _smooth_angle(self, angle):
        delta = float(angle - self._prev_angle)
        if abs(delta) < self.ANGLE_DEADBAND:
            angle = self._prev_angle
        else:
            angle = self._prev_angle + self.ANGLE_SMOOTH_ALPHA * delta
        step = float(np.clip(angle - self._prev_angle,
                             -self.ANGLE_MAX_STEP, self.ANGLE_MAX_STEP))
        angle = float(np.clip(self._prev_angle + step, ANGLE_MIN, ANGLE_MAX))
        self._prev_angle = angle
        return angle

    def _steer_gain_ratio(self, xs, ys, rel_s, distance_m):
        if xs is None or ys is None or xs.size < 3:
            return 0.0, 0.0

        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        rel_s = (
            np.asarray(rel_s, dtype=np.float64)
            if rel_s is not None and rel_s.size == xs.size else None
        )
        valid = (
            np.isfinite(xs) & np.isfinite(ys)
            & (xs >= 0.0)
        )
        if rel_s is not None:
            valid &= np.isfinite(rel_s) & (rel_s >= 0.0) & (rel_s <= distance_m)
        else:
            valid &= xs <= distance_m
        xs = xs[valid]
        ys = ys[valid]
        if rel_s is not None:
            rel_s = rel_s[valid]
        if xs.size < 3:
            return 0.0, 0.0

        order = np.argsort(rel_s if rel_s is not None else xs)
        xs = xs[order]
        ys = ys[order]
        keep = np.concatenate(([True], np.diff(xs) > 1e-4))
        xs = xs[keep]
        ys = ys[keep]
        if xs.size < 3:
            return 0.0, 0.0

        try:
            dy = np.gradient(ys, xs)
            ddy = np.gradient(dy, xs)
            curv = np.abs(ddy) / np.maximum((1.0 + dy * dy) ** 1.5, 1e-6)
            curv = curv[np.isfinite(curv)]
            kappa = float(np.percentile(curv, 85)) if curv.size else 0.0
            headings = np.degrees(np.arctan(dy))
            bend_delta = float(np.percentile(np.abs(headings - headings[0]), 85))
            line_angle_error = float(abs(np.average(headings)))
        except (FloatingPointError, ValueError):
            kappa = 0.0
            bend_delta = 0.0
            line_angle_error = 0.0

        curv_ratio = np.clip(
            (kappa - self.PREVIEW_CURV_START)
            / max(self.PREVIEW_CURV_FULL - self.PREVIEW_CURV_START, 1e-6),
            0.0, 1.0)
        bend_ratio = np.clip(
            (bend_delta - self.STEER_HEADING_START_DEG)
            / max(self.STEER_HEADING_FULL_DEG - self.STEER_HEADING_START_DEG, 1e-6),
            0.0, 1.0)
        return float(max(curv_ratio, bend_ratio)), line_angle_error

    def _preview_slow_ratio(self, xs, ys):
        if xs is None or ys is None or xs.size < 4:
            return 0.0

        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        valid = (
            np.isfinite(xs) & np.isfinite(ys)
            & (xs >= 0.0) & (xs <= self.PREVIEW_DISTANCE_M)
        )
        xs = xs[valid]
        ys = ys[valid]
        if xs.size < 4:
            return 0.0

        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        keep = np.concatenate(([True], np.diff(xs) > 1e-4))
        xs = xs[keep]
        ys = ys[keep]
        if xs.size < 4:
            return 0.0

        try:
            dy = np.gradient(ys, xs)
            ddy = np.gradient(dy, xs)
            curv = np.abs(ddy) / np.maximum((1.0 + dy * dy) ** 1.5, 1e-6)
            curv = curv[np.isfinite(curv)]
            kappa = float(np.percentile(curv, 85)) if curv.size else 0.0
        except (FloatingPointError, ValueError):
            kappa = 0.0

        lateral_delta = float(np.max(np.abs(ys - ys[0])))
        curv_ratio = np.clip(
            (kappa - self.PREVIEW_CURV_START)
            / max(self.PREVIEW_CURV_FULL - self.PREVIEW_CURV_START, 1e-6),
            0.0, 1.0)
        lateral_ratio = np.clip(
            (lateral_delta - self.PREVIEW_LATERAL_START_M)
            / max(self.PREVIEW_LATERAL_FULL_M - self.PREVIEW_LATERAL_START_M, 1e-6),
            0.0, 1.0)
        return float(max(curv_ratio, lateral_ratio))

    def _speed_from_angle(self, angle, xs, ys, preview_ratio=None, in_corner=False):
        if preview_ratio is None:
            preview_ratio = self._preview_slow_ratio(xs, ys)

        exit_ratio = np.clip(
            float(preview_ratio) / max(self.CORNER_EXIT_RATIO, 1e-6),
            0.0, 1.0,
        )
        angle_scale = (self.CORNER_EXIT_ANGLE_SCALE
                       + (1.0 - self.CORNER_EXIT_ANGLE_SCALE) * exit_ratio)
        if in_corner and preview_ratio >= self.CORNER_EXIT_RATIO:
            angle_scale *= self.CORNER_INSIDE_ANGLE_SCALE
        a = abs(angle) * angle_scale
        if a <= self.SPEED_SLOW_START_DEG:
            angle_speed = self.SPEED_DEFAULT
        else:
            t = min(1.0, (a - self.SPEED_SLOW_START_DEG)
                    / (self.SPEED_SLOW_FULL_DEG - self.SPEED_SLOW_START_DEG))
            angle_speed = max(self.SPEED_MIN,
                              self.SPEED_DEFAULT - (self.SPEED_DEFAULT - self.SPEED_MIN) * t)

        # 코너 inside floor 로직 (예선 보존 — 반드시 유지)
        inside_floor = self.CORNER_BOOST_SPEED if self._corner_boost else self.CORNER_INSIDE_SPEED
        corner_low_speed = inside_floor if in_corner else self.PREVIEW_SLOW_SPEED
        curve_preview_speed = (self.SPEED_CURVE_BASE
                               - (self.SPEED_CURVE_BASE - corner_low_speed) * preview_ratio)
        if preview_ratio < self.CORNER_EXIT_RATIO:
            exit_t = preview_ratio / max(self.CORNER_EXIT_RATIO, 1e-6)
            preview_speed = (self.SPEED_DEFAULT
                             - (self.SPEED_DEFAULT - curve_preview_speed) * exit_t)
        else:
            preview_speed = curve_preview_speed
        speed = min(angle_speed, preview_speed)
        if in_corner:
            speed = max(inside_floor, speed)
        return float(np.clip(max(self.SPEED_MIN, speed), 0.0, SPEED_CMD_MAX))

    # ==================== 디버그 ====================

    @staticmethod
    def _fmt_lim(v):
        """cap/floor/fixed 표시 — <=0(없음)은 '-'."""
        try:
            f = float(v)
        except (TypeError, ValueError):
            return "-"
        return "%.1f" % f if f > 0.0 else "-"

    def _tick_debug(self):
        d = self._dbg
        if not d:
            return
        msg = String()
        msg.data = (
            "owner=%s%s prof=%s | cap: cmd=%s straight=%s | "
            "floor: corner=%s min=%s cmd=%s | fixed=%s | v_base=%.1f -> "
            "cmd(a=%.1f, v=%.1f) out(a=%.1f, v=%.1f) | "
            "teleop=%s shadow=%s lost=%d off=%.2f"
            % (
                d["owner"], "(stale)" if d["cmd_stale"] else "", d["profile"],
                self._fmt_lim(d["cap_cmd"]), self._fmt_lim(d["cap_straight"]),
                self._fmt_lim(d["floor_corner"]), self._fmt_lim(d["floor_min"]),
                self._fmt_lim(d["floor_cmd"]),
                self._fmt_lim(d["fixed"]), d["v_base"],
                d["angle_cmd"], d["speed_cmd"], d["angle_out"], d["speed_out"],
                "on" if d["teleop"] else "off", "ok" if d["shadow"] else "no",
                d["lost_ticks"], d["lane_offset"],
            )
        )
        self._pub_dbg.publish(msg)


# ======================== main ========================

def main(args=None):
    rclpy.init(args=args)
    node = MotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
