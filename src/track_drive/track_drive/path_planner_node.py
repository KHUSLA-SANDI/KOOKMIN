#!/usr/bin/env python3
"""통합 경로계획(실차) — phase 머신 보존 + 배선·기준 수술판.

역할: WAIT→LANE(→CONE/PEDESTRIAN/SHORTCUT) phase 머신으로 /center_path·/target을
  만들고, 속도·조향 정책은 /drive_cmd 20Hz 스냅샷 하나로 motion에 전달한다.
입력: /detect/lane(PoseArray, z=cls), /detect/objects(PoseArray, u/v/cls),
  /detect/traffic_light(String), /detect/lane_quality(Float32),
  /scan(LaserScan, BEST_EFFORT), /car_state(Float32MultiArray),
  /corridor_path(PoseArray, 외부팀 콘 통로), /manual_go(Bool).
출력: /center_path /target /lane_left /lane_right (예선 형식 그대로),
  /drive_cmd (Float32MultiArray — lib/drive_cmd.py 레이아웃).
예선 대비 변경점:
  - Bool 라치류(/emergency_stop /child_zone /cone_mode /shortcut_slow /corner_boost
    /overtake_speed /left_turn1·2 /s_zone) 전부 제거 → /drive_cmd 스냅샷으로 대체.
  - WAIT→LANE 직행(무조건 CONE 진입 제거). CONE은 LANE 중 라이다 클러스터 감지 기반
    진입, miss '연속 스캔 수'로 종료. /corridor_path 신선 시 기준선 우선 사용.
  - heading 게이트 전부 IMU 절대각 → /car_state yaw_rel(출발 기준각, wrap-safe) 기준.
  - SHORTCUT TURNING은 시간 틱 대신 BLOCK(block_id 원샷 + 목표 |Δyaw| 도달 판정).
    _shortcut_done 평생 1회 래치. POLICE/GREEN-정지-후-공략/종료 시 child 강제 제거.
  - plan()엔 빈 장애물(내부 0.8m 시프트 비활성) → path_gen 오프셋 최적화가 회피 전담.
  - OVERTAKE는 스텁(인터페이스만). /sc_go 제거 → /manual_go 실소비.
"""

import math
import os

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose, PoseArray, PointStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Empty, Float32, Float32MultiArray, String

# LANE 모드: 친구 plan() 재사용 (lane_planner.py는 수정하지 않는다)
from .lane_planner import plan as lane_plan
from .lane_planner import (
    YELLOW_CLS_IDS, YELLOW_DASH_CLS_IDS, WHITE_CLS_IDS,
    PLAN_HZ, TARGET_X, TARGET_Y_LIMIT,
)
from .lib.drive_cmd import Owner, SteerProfile, encode_drive_cmd, decode_car_state
from .lib.cluster import cluster_scan, ClusterTracker
from .lib.path_gen import PathGen

# /detect/objects 의 v 를 기준 해상도로 환산할 때 실제 프레임 높이의 출처.
# (cv2 를 끌고 오는 lib.bev 대신 yaml 한 줄만 읽는다 — planner 는 영상 처리 안 함)
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CAMERA_YAML = os.path.join(_PKG_ROOT, "config", "camera.yaml")

# ======================== YOLO 클래스 ID (best.pt 학습 시점 기준) ========================
# 0=BLACK_CAR 1=CHILD_END 2=CHILD_LANE 3=CHILD_START 4=CROSSROAD_OUT 5=GOAL
# 6=GREEN 7=GREEN_CAR 8=HUMAN 9=LANE 10=LEFT 11=MID 12=POLICE 13=RED 14=STOP 15=YELLOW
# (신호등 GREEN/RED는 /detect/traffic_light 문자열로 받는다 — 여기선 ID 불필요)

LEFT_SIGN_CLS_ID = 10
STOP_CLS_ID = 14
CROSSROAD_CLS_ID = 4
CHILD_START_CLS_ID = 3
CHILD_END_CLS_ID = 1
BLACK_CAR_CLS_ID = 0
GREEN_CAR_CLS_ID = 7


# ======================== 헬퍼 ========================

def _poses_from_xy(stamp, xs, ys):
    msg = PoseArray()
    msg.header.stamp = stamp
    msg.header.frame_id = "lidar_frame"
    for x, y in zip(xs, ys):
        p = Pose()
        p.position.x = float(x)
        p.position.y = float(y)
        msg.poses.append(p)
    return msg


def _circ_diff_deg(a, b):
    """wrap-safe 원형 차이 a-b (deg) → -180~180."""
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


# ======================== OVERTAKE 스텁 ========================

class OvertakeStub:
    """추월 스텁 — 예선 상태머신은 이식하지 않음(본선 재설계 자리).

    인터페이스만 유지:
      update(front_min_m, left_on, right_on, car_v, car_state) → None | dict
        None  = 개입 없음 (현재 항상)
        dict  = {"offset": m(좌+), "speed_cap": 명령단위}
                활성화되면 planner가 path_gen 기준선 오프셋과
                drive_cmd.speed_cap에 그대로 반영할 자리.
    TODO(본선): 전방 저속 차량 감지 — 라이다 트래커 vel_xy(동적 판별) +
      /detect/objects 차량 클래스(BLACK_CAR/GREEN_CAR) bbox v 트리거,
      2차선 오프셋 → 측면 창 확인 → 복귀 시퀀스 설계.
    """

    active = False

    def update(self, front_min_m=float("inf"), left_on=False, right_on=False,
               car_v=None, car_state=None):
        return None


# ======================== ROS 노드 ========================

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__("path_planner_node")

        # ---------------- 파라미터 (tuning.yaml 추출 대상) ----------------
        # 출발/공통
        self.declare_parameter("start_confirm_ticks", 6)        # [SIM] 기존 6틱 출발 확인 유지
        self.declare_parameter("start_grace_sec", 20.0)         # [SIM] 기존 400틱 — 출발 후 정지선/지름길 무시
        self.declare_parameter("objects_fresh_sec", 0.5)        # /detect/objects 신선도 (v임계 판정 전제)
        # ── /detect/objects 의 v(픽셀)를 기준 해상도로 환산 ──────────────────
        # 아래 *_v_threshold 들은 전부 예선 시뮬 480p 에서 튜닝된 픽셀값이다.
        # 실차 카메라는 1920x1080 으로 올라갔으므로(캘리브 기준) 그냥 두면
        # v 가 2.25배로 들어와 STOP/CROSSROAD 가 훨씬 멀리서 오발동한다.
        # → 들어오는 v 를 obj_v_ref_height 기준으로 스케일해서 임계값의 의미를
        #   해상도와 무관하게 고정한다. 실제 높이는 camera.yaml 이 단일 출처.
        self.declare_parameter("obj_v_ref_height", 480.0)       # 임계값들이 튜닝된 기준 높이
        self.declare_parameter("camera_yaml", DEFAULT_CAMERA_YAML)
        self.declare_parameter("lane_stale_sec", 0.6)           # /detect/lane 신선도 — 초과 시 경로 발행 중단
        self.declare_parameter("scan_stale_sec", 1.0)           # /scan 신선도 — 초과 시 장애물 무시·CONE miss 계수
        self.declare_parameter("manual_go_fresh_sec", 2.0)      # /manual_go 유효 시간 (실소비 전 잔류 방지)
        self.declare_parameter("lane_quality_min", 0.3)         # 이 미만이면 감속 cap
        self.declare_parameter("low_quality_speed_cap", 7.0)    # [SIM] 차선 품질 저하 시 speed_cap
        # 클러스터/트래커 (lib/cluster.py)
        self.declare_parameter("cluster_gap_m", 0.35)           # [SIM] 예선 CLUSTER_GAP 계승
        self.declare_parameter("cluster_padding", 0.05)         # [SIM] 예선 CLUSTER_PADDING 계승
        self.declare_parameter("cluster_gap_per_m", 0.05)
        self.declare_parameter("cluster_min_pts", 2)
        self.declare_parameter("cluster_r_min", 0.05)
        self.declare_parameter("cluster_r_max", 8.0)
        self.declare_parameter("tracker_k", 2)                  # k-of-n 확정 관측
        self.declare_parameter("tracker_n", 3)
        self.declare_parameter("tracker_match_dist", 0.5)
        # 회피 (lib/path_gen.py)
        self.declare_parameter("avoid_clear_margin", 0.45)      # 차폭/2 + 여유 (m)
        self.declare_parameter("avoid_ramp_step", 0.06)         # 호출당 오프셋 변화 상한 (m)
        self.declare_parameter("avoid_left_offset_max", 0.9)    # 좌(+y) 상한 — 좌 조향 스톱 여유 부족
        self.declare_parameter("avoid_obs_r_max", 0.6)          # 이보다 큰 클러스터(벽 등)는 회피 대상 제외
        self.declare_parameter("avoid_obs_x_max", 8.0)          # 회피 대상 전방 거리 상한 (m)
        # CONE — 진입/종료 (실차 신규: 감지 기반)
        self.declare_parameter("cone_enter_r_min", 0.1)         # 콘으로 볼 클러스터 반경 하한 (m)
        self.declare_parameter("cone_enter_r_max", 0.4)         # 콘으로 볼 클러스터 반경 상한 (m)
        self.declare_parameter("cone_enter_min_clusters", 4)    # ROI 안 콘 클러스터 최소 개수
        self.declare_parameter("cone_enter_scans", 3)           # 연속 스캔 수 (진입 디바운스)
        self.declare_parameter("cone_exit_miss_scans", 20)      # miss 연속 스캔 수 (기존 40틱 → 20스캔)
        self.declare_parameter("cone_grace_sec", 1.0)           # [SIM] 기존 20틱 — 진입 직후 miss 미계수
        self.declare_parameter("cone_reenter_block_sec", 5.0)   # CONE 종료 직후 재진입 차단 (핑퐁 방지)
        self.declare_parameter("cone_speed_cap", 16.0)          # [SIM] 예선 CONE_SPEED_FIXED — fixed 아님, cap
        self.declare_parameter("corridor_fresh_sec", 0.5)       # /corridor_path 신선도 (기준선 우선 사용)
        # CONE — 왼쪽 콘 피팅 (기존 _tick_cone 이식)
        self.declare_parameter("cone_x_min", 0.3)               # [SIM] 전방 ROI 최소 (m)
        self.declare_parameter("cone_x_max", 10.0)              # [SIM] 전방 ROI 최대 (m)
        self.declare_parameter("cone_y_min", -6.0)              # [SIM] 좌우 ROI (m)
        self.declare_parameter("cone_y_max", 6.0)               # [SIM]
        self.declare_parameter("track_width", 4.4)              # [SIM] 예선 트랙 폭 — 본선 실측 예정
        self.declare_parameter("cone_fit_min_points", 2)        # [SIM]
        self.declare_parameter("cone_fit_min_x_span", 0.8)      # [SIM]
        self.declare_parameter("cone_fit_deg2_span", 2.0)       # [SIM] x스팬 이 이상 + 3점이면 2차 피팅
        self.declare_parameter("cone_fit_curve_max", 1.5)       # [SIM]
        self.declare_parameter("cone_fit_slope_max", 3.0)       # [SIM]
        self.declare_parameter("cone_left_accept_dist", 1.5)    # [SIM]
        self.declare_parameter("cone_fit_smooth_alpha", 0.20)   # [SIM]
        self.declare_parameter("cone_coast_decay", 0.2)         # [SIM] miss당 곡률항 감쇠
        self.declare_parameter("cone_single_bias", 0.268)       # [SIM] 콘 1개 이하 시 시계방향 15°(tan15°) 보정
        self.declare_parameter("cone_sample_x_start", 0.5)      # [SIM]
        self.declare_parameter("cone_sample_x_end", 6.0)        # [SIM]
        self.declare_parameter("cone_sample_n", 25)             # [SIM]
        # 정지선(STOP) / SHORTCUT — heading은 전부 yaw_rel(출발 기준각) 기준
        self.declare_parameter("stop_v_threshold", 312.0)       # [SIM] 480p 기준 STOP bbox 하단 v
        self.declare_parameter("cross_v_threshold", 278.0)      # [SIM] CROSSROAD_OUT bbox 하단 v
        self.declare_parameter("stop_heading_rel", 0.0)         # 정지선 직선 구간 yaw_rel (deg) — 코스 실측 예정
        self.declare_parameter("stop_heading_tol", 10.0)        # [SIM] ±허용 (deg)
        self.declare_parameter("stop_block_heading_rel", -90.0)  # [SIM] 기존 절대 270° 등가 — 코스 실측 예정
        self.declare_parameter("stop_block_tol", 5.0)           # [SIM] ±허용 (deg)
        self.declare_parameter("stop_block_sec", 3.0)           # [SIM] 기존 60틱 — 통과 후 정지선 트리거 차단
        self.declare_parameter("sc_turn_target_deg", 90.0)      # 좌회전 목표 |Δyaw| (deg) — 코스 실측 예정
        self.declare_parameter("sc_turn_timeout_sec", 6.0)      # BLOCK 안전망 (yaw 미갱신 대비 강제 진행)
        self.declare_parameter("sc_follow_timeout_sec", 15.0)   # [SIM] 기존 300틱 — 하드 타임아웃 복원
        self.declare_parameter("sc_wait_timeout_sec", 8.0)      # WAITING 타임아웃 — 초과 시 지름길 포기(래치 해제)
        self.declare_parameter("left_debounce_hits", 2)         # [SIM] LEFT 이벤트 최소 검출 횟수
        self.declare_parameter("shortcut_speed_cap", 15.0)      # [SIM] 예선 SHORTCUT_SPEED — cap으로
        # PEDESTRIAN (본선은 회피로 대체 예정 — 기본 비활성)
        self.declare_parameter("enable_pedestrian", False)
        self.declare_parameter("ped_x_min", 1.5)                # [SIM]
        self.declare_parameter("ped_x_max", 8.0)                # [SIM]
        self.declare_parameter("ped_r_min", 0.1)                # [SIM] 사람 크기 클러스터 반경
        self.declare_parameter("ped_r_max", 0.4)                # [SIM]
        self.declare_parameter("ped_road_half_width", 1.2)      # [SIM] center 기준 도로 반폭
        self.declare_parameter("ped_stop_sec", 2.2)             # [SIM] 기존 44틱 정지
        self.declare_parameter("ped_cooldown_sec", 30.0)        # [SIM] 기존 600틱
        self.declare_parameter("ped_freeze_sec", 1.5)           # [SIM] 기존 30틱 center freeze
        self.declare_parameter("ped_car_cluster_count", 3)      # [SIM] 밀집=차 판정 개수
        self.declare_parameter("ped_car_spread", 1.5)           # [SIM] 밀집 판정 거리
        # CHILD ZONE (기본 비활성)
        self.declare_parameter("enable_child_zone", False)
        self.declare_parameter("child_start_v_thr", 200.0)      # [SIM]
        self.declare_parameter("child_end_v_thr", 400.0)        # [SIM]
        self.declare_parameter("child_exit_delay_sec", 3.0)     # [SIM] 기존 60틱
        self.declare_parameter("child_enter_cooldown_sec", 3.0)  # [SIM] 기존 60틱
        self.declare_parameter("child_exit_cooldown_sec", 10.0)  # [SIM] 기존 200틱
        self.declare_parameter("child_hard_timeout_sec", 17.0)  # [SIM] 기존 340틱
        self.declare_parameter("child_speed_cap", 22.0)         # [SIM] 예선 CHILD_ZONE_SPEED
        self.declare_parameter("corner_boost_window_sec", 20.0)  # [SIM] 기존 400틱 — child 종료 후 부스트 창
        self.declare_parameter("corner_boost_floor", 0.0)       # speed_floor. 0=비활성 — 본선 튜닝 예정
        # S존 (기본 비활성 — 켜면 yaw_rel 기준. 절대각 판정 재도입 금지)
        self.declare_parameter("enable_s_zone", False)
        self.declare_parameter("s_zone_yaw_lo", 10.0)           # [SIM] yaw_rel 하한 (deg)
        self.declare_parameter("s_zone_yaw_hi", 170.0)          # [SIM] yaw_rel 상한 (deg)
        self.declare_parameter("s_zone_block_sec", 8.0)         # [SIM] 기존 160틱 — CONE→LANE 직후 오인 차단
        # 라이다 측면/정면 창 (기존 _check_side 이식 — 창 ±1°→±3°, OVERTAKE 스텁 입력)
        self.declare_parameter("side_left_angle_deg", 90.0)     # [SIM] 왼쪽 9시 방향
        self.declare_parameter("side_right_angle_deg", -88.0)   # [SIM]
        self.declare_parameter("side_half_deg", 3.0)            # 기존 ±1° → ±3° (500빔 incr 0.72° 대응)
        self.declare_parameter("side_left_r_min", 0.5)          # [SIM]
        self.declare_parameter("side_left_r_max", 3.2)          # [SIM]
        self.declare_parameter("side_right_r_min", 0.01)        # [SIM]
        self.declare_parameter("side_right_r_max", 3.0)         # [SIM]
        self.declare_parameter("side_debounce_scans", 3)        # 연속 스캔 수 기준 (기존 틱 기준 대체)
        self.declare_parameter("front_half_deg", 6.0)           # [SIM] 정면 콘 반각
        self.declare_parameter("front_r_min", 0.3)              # [SIM]
        self.declare_parameter("front_r_max", 8.0)              # [SIM]

        P = self._prm

        # ---------------- 상태 ----------------
        self.phase = "IDLE"
        self._wait_confirm = 0            # WAIT 출발 확인 카운트다운 (기존 6틱)
        self._grace_until = 0.0           # 출발 후 정지선/지름길 무시 만료 시각
        self._log_counter = 0

        # /drive_cmd 이번 틱 스냅샷 (매 틱 _tick에서 재계산 — 라치 금지)
        self._cmd = self._fresh_cmd()
        self._block_id = 0                # 마지막 발행 블록 id (새 블록마다 +1 = 원샷)

        # 센서 캐시
        self._yellow_xs = np.array([], dtype=np.float64)
        self._yellow_ys = np.array([], dtype=np.float64)
        self._yellow_dash_xs = np.array([], dtype=np.float64)
        self._yellow_dash_ys = np.array([], dtype=np.float64)
        self._white_xs = np.array([], dtype=np.float64)
        self._white_ys = np.array([], dtype=np.float64)
        self._has_lane = False
        self._lane_rx_time = -1e9         # /detect/lane 마지막 수신 시각 (신선도 가드)
        self._lane_quality = 0.0          # 첫 수신 전 0 = 안전측(감속 cap)
        self._light = "NONE"              # /detect/traffic_light (디바운스는 yolo가 함)
        self._car = decode_car_state([])  # /car_state 안전 기본값
        self._corridor = None             # [(x, y)] 외부팀 콘 통로
        self._corridor_time = -1e9
        self._manual_go_time = -1e9       # /manual_go 수신 시각 (실소비 시 리셋)

        # /detect/objects 의 v 스케일 (기동 시 1회 확정 — 매 메시지 계산 금지)
        self._obj_v_scale = self._resolve_obj_v_scale()

        # /detect/objects 파생 (메시지마다 덮어씀 — 클래스 없으면 None)
        self._objects_time = -1e9
        self._obj_cls = set()
        self._stop_max_v = None
        self._cross_max_v = None
        self._child_start_max_v = None
        self._child_end_max_v = None
        self._car_max_v = None            # OVERTAKE 스텁 입력

        # /scan 파생
        self._scan_seq = 0
        self._scan_rx_time = -1e9         # /scan 마지막 수신 시각 (신선도 가드)
        self._tracks = []
        self._cone_enter_streak = 0       # ROI 안 콘 클러스터 충분 — 연속 스캔 수
        self._side_left = False
        self._side_right = False
        self._left_on_s = self._left_off_s = 0
        self._right_on_s = self._right_off_s = 0
        self._front_min = float("inf")

        # CONE
        self._cone_prev_fit = None
        self._cone_left_count = 0
        self._cone_miss = 0               # 연속 miss 스캔 수
        self._cone_seen_seq = -1          # miss 계수용 마지막 처리 스캔 번호
        self._cone_grace_until = 0.0
        self._cone_reenter_until = 0.0    # 종료 직후 재진입 차단 만료 시각

        # 정지선/SHORTCUT
        self._red_stopping = False
        self._stop_block_until = 0.0      # 270° 등가 통과 후 정지선 트리거 차단 만료 시각
        self._sc_sub = None               # None | WAITING | TURNING_1 | FOLLOW | TURNING_2
        self._sc_started = 0.0
        self._sc_left_hits = 0
        self._sc_yaw_start = 0.0          # TURNING 진입 시 yaw_rel 캡처
        self._shortcut_done = False       # 원샷 래치 — 평생 1회 (WAITING 타임아웃 시 해제)
        self._sc_retry_until = 0.0        # WAITING 타임아웃 포기 후 재진입 쿨다운 만료 시각

        # PEDESTRIAN
        self._ped_until = 0.0
        self._ped_cooldown_until = 0.0
        self._path_freeze_until = 0.0
        self._frozen_center = None

        # CHILD ZONE
        self._child_zone = False
        self._child_enter_cd_until = 0.0
        self._child_exit_cd_until = 0.0
        self._child_exit_at = None        # END 감지 후 해제 예정 시각
        self._child_deadline = 0.0
        self._cb_until = 0.0              # 코너 부스트 창 만료 시각
        self._cb_prev_child = False

        # LANE
        self._lane_center_coef = None
        self._last_lane_result = None
        self._lane_offset = 0.0           # path_gen이 적용 중인 횡오프셋 (디버그 필드)
        self._s_zone_block_until = 0.0
        self._s_zone_active = False

        # 순수 파이썬 모듈
        self._tracker = ClusterTracker(
            k=int(P("tracker_k")), n=int(P("tracker_n")),
            match_dist=float(P("tracker_match_dist")))
        self._path_gen = PathGen({
            "clear_margin": float(P("avoid_clear_margin")),
            "ramp_step": float(P("avoid_ramp_step")),
            "left_offset_max": float(P("avoid_left_offset_max")),
        })
        self._overtake = OvertakeStub()

        # ---------------- 구독 ----------------
        self.create_subscription(PoseArray, "/detect/lane", self._on_lane, 10)
        self.create_subscription(PoseArray, "/detect/objects", self._on_objects, 10)
        self.create_subscription(String, "/detect/traffic_light", self._on_light, 10)
        self.create_subscription(Float32, "/detect/lane_quality", self._on_quality, 10)
        self.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, "/car_state", self._on_car_state, 10)
        self.create_subscription(PoseArray, "/corridor_path", self._on_corridor, 10)
        self.create_subscription(Bool, "/manual_go", self._on_manual_go, 10)

        # ---------------- 발행 ----------------
        self._pub_center = self.create_publisher(PoseArray, "/center_path", 10)
        self._pub_target = self.create_publisher(PointStamped, "/target", 10)
        self._pub_left = self.create_publisher(PoseArray, "/lane_left", 10)
        self._pub_right = self.create_publisher(PoseArray, "/lane_right", 10)
        self._pub_cmd = self.create_publisher(Float32MultiArray, "/drive_cmd", 10)
        # 출발 순간 yaw 기준각 캡처 트리거 (car_state_node가 구독)
        self._pub_capture_yaw = self.create_publisher(Empty, "/capture_yaw", 1)

        self.create_timer(1.0 / PLAN_HZ, self._tick)
        self.get_logger().info("PLANNER(실차) 시작")

    # ---------------- 공통 헬퍼 ----------------

    def _prm(self, name):
        return self.get_parameter(name).value

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _fresh_cmd():
        """이번 틱 /drive_cmd 스냅샷 초기값 — 매 틱 재계산 (라치 금지)."""
        return {"owner": Owner.LANE, "steer_profile": SteerProfile.NORMAL,
                "caps": [], "floor": 0.0, "fixed": 0.0, "block_param": 0.0}

    def _take_manual_go(self, now):
        """/manual_go 실소비 — 신선하면 True 반환 후 소진 (잔류 오발 방지)."""
        if (now - self._manual_go_time) <= float(self._prm("manual_go_fresh_sec")):
            self._manual_go_time = -1e9
            return True
        return False

    def _objects_fresh(self, now):
        return (now - self._objects_time) <= float(self._prm("objects_fresh_sec"))

    def _lane_fresh(self, now):
        """/detect/lane 신선도 — 스테일이면 경로 발행 중단 (인지 소실 폴백)."""
        return (now - self._lane_rx_time) <= float(self._prm("lane_stale_sec"))

    def _scan_fresh(self, now):
        """/scan 신선도 — 스테일이면 동결 트랙 기반 회피·CONE 유지 금지."""
        return (now - self._scan_rx_time) <= float(self._prm("scan_stale_sec"))

    # ---------------- 콜백 ----------------

    def _on_lane(self, msg: PoseArray):
        """기존 _on_lane 이식 — z(cls_id)로 노랑8/점선16/흰6,2 분리."""
        self._lane_rx_time = self._now()   # 수신 시각 기록 (신선도 가드)
        self._has_lane = True
        if not msg.poses:
            self._yellow_xs = np.array([], dtype=np.float64)
            self._yellow_ys = np.array([], dtype=np.float64)
            self._yellow_dash_xs = np.array([], dtype=np.float64)
            self._yellow_dash_ys = np.array([], dtype=np.float64)
            self._white_xs = np.array([], dtype=np.float64)
            self._white_ys = np.array([], dtype=np.float64)
            return
        xs = np.fromiter((p.position.x for p in msg.poses),
                         dtype=np.float64, count=len(msg.poses))
        ys = np.fromiter((p.position.y for p in msg.poses),
                         dtype=np.float64, count=len(msg.poses))
        cls = np.fromiter((int(p.position.z) for p in msg.poses),
                          dtype=np.int32, count=len(msg.poses))
        ym = np.isin(cls, list(YELLOW_CLS_IDS))
        dm = np.isin(cls, list(YELLOW_DASH_CLS_IDS))
        wm = np.isin(cls, list(WHITE_CLS_IDS))
        self._yellow_xs, self._yellow_ys = xs[ym], ys[ym]
        self._yellow_dash_xs, self._yellow_dash_ys = xs[dm], ys[dm]
        self._white_xs, self._white_ys = xs[wm], ys[wm]

    def _resolve_obj_v_scale(self):
        """/detect/objects 의 v 를 obj_v_ref_height 기준으로 환산할 배율.

        camera.yaml 의 image_height 가 실제 프레임 높이의 단일 출처다
        (sensors.launch.py 의 해상도도 여기에 맞춰져 있어야 한다).
        읽기 실패 / 미기재면 1.0 (환산 안 함) + 경고 — 조용히 틀리지 않게.
        """
        ref = float(self.get_parameter("obj_v_ref_height").value)
        path = str(self.get_parameter("camera_yaml").value)
        h = None
        try:
            import yaml as _yaml
            with open(path, "r", encoding="utf-8") as f:
                data = _yaml.safe_load(f)
            if isinstance(data, dict):
                h = float(data["image_height"])
        except Exception:                     # noqa: BLE001 — 어떤 실패든 폴백
            h = None

        if not ref or ref <= 0 or h is None or h <= 0:
            self.get_logger().warn(
                f"[objects] camera.yaml 에서 image_height 를 못 읽었다 (path={path}) — "
                f"v 환산 생략(x1.0). *_v_threshold 는 {ref:.0f}p 기준값이라 "
                "실제 해상도가 다르면 STOP/CROSSROAD 가 오발동한다.")
            return 1.0

        scale = ref / h
        self.get_logger().info(
            f"[objects] v 환산: 실제 {h:.0f}p → 기준 {ref:.0f}p (x{scale:.4f}) "
            f"— *_v_threshold 는 {ref:.0f}p 픽셀값 그대로 쓴다")
        return scale

    def _on_objects(self, msg: PoseArray):
        """기존 _on_road 이식 — (u,v,cls)에서 클래스별 bbox 하단 v 최댓값(거리 proxy).
        LEFT 등 이벤트 존재 검사도 여기(클래스 집합)서 한다."""
        stop_v = cross_v = cs_v = ce_v = car_v = None
        cls_set = set()
        for p in msg.poses:
            cls = int(p.position.z)
            # bbox 하단 v (클수록 가까움). 임계값이 튜닝된 기준 해상도로 환산.
            v_bot = float(p.position.y) * self._obj_v_scale
            cls_set.add(cls)
            if cls == STOP_CLS_ID:
                if stop_v is None or v_bot > stop_v:
                    stop_v = v_bot
            elif cls == CROSSROAD_CLS_ID:
                if cross_v is None or v_bot > cross_v:
                    cross_v = v_bot
            elif cls == CHILD_START_CLS_ID:
                if cs_v is None or v_bot > cs_v:
                    cs_v = v_bot
            elif cls == CHILD_END_CLS_ID:
                if ce_v is None or v_bot > ce_v:
                    ce_v = v_bot
            elif cls in (BLACK_CAR_CLS_ID, GREEN_CAR_CLS_ID):
                if car_v is None or v_bot > car_v:
                    car_v = v_bot
        self._stop_max_v = stop_v
        self._cross_max_v = cross_v
        self._child_start_max_v = cs_v
        self._child_end_max_v = ce_v
        self._car_max_v = car_v
        self._obj_cls = cls_set
        self._objects_time = self._now()

    def _on_light(self, msg: String):
        self._light = msg.data   # "GREEN" | "RED" | "NONE" (디바운스는 yolo가 함)

    def _on_quality(self, msg: Float32):
        self._lane_quality = float(msg.data)

    def _on_car_state(self, msg: Float32MultiArray):
        self._car = decode_car_state(msg.data)

    def _on_corridor(self, msg: PoseArray):
        if msg.poses:
            self._corridor = [(p.position.x, p.position.y) for p in msg.poses]
            self._corridor_time = self._now()
        # 빈 메시지 = 통로 없음 → 신선도 갱신 안 함 (자연 만료)

    def _on_manual_go(self, msg: Bool):
        if msg.data:
            self._manual_go_time = self._now()

    @staticmethod
    def _scan_window(msg, angle, half, r_min, r_max):
        """기존 _check_side 창 로직 이식 — 창 안 최소거리 반환 (미검출=inf)."""
        lo, hi = angle - half, angle + half
        idx_lo = max(0, int((lo - msg.angle_min) / msg.angle_increment))
        idx_hi = min(len(msg.ranges), int((hi - msg.angle_min) / msg.angle_increment))
        best = float("inf")
        for i in range(idx_lo, idx_hi):
            r = msg.ranges[i]
            if math.isfinite(r) and r_min <= r <= r_max and r < best:
                best = r
        return best

    def _on_scan(self, msg: LaserScan):
        P = self._prm
        self._scan_seq += 1
        self._scan_rx_time = self._now()   # 수신 시각 기록 (신선도 가드)

        # 클러스터링 + 트래커 (확정 장애물 = path_gen/CONE/PED 공통 재료)
        clusters = cluster_scan(
            msg.ranges, msg.angle_min, msg.angle_increment,
            r_min=float(P("cluster_r_min")), r_max=float(P("cluster_r_max")),
            gap_m=float(P("cluster_gap_m")), min_pts=int(P("cluster_min_pts")),
            gap_per_m=float(P("cluster_gap_per_m")), padding=float(P("cluster_padding")))
        self._tracks = self._tracker.update(clusters)

        # CONE 진입 감지 — ROI 안 콘 반경 클러스터 개수, 연속 스캔 디바운스
        n_cone = 0
        for t in self._tracks:
            if not t["confirmed"]:
                continue
            if not (float(P("cone_x_min")) <= t["x"] <= float(P("cone_x_max"))):
                continue
            if not (float(P("cone_y_min")) <= t["y"] <= float(P("cone_y_max"))):
                continue
            if float(P("cone_enter_r_min")) <= t["r"] <= float(P("cone_enter_r_max")):
                n_cone += 1
        if n_cone >= int(P("cone_enter_min_clusters")):
            self._cone_enter_streak += 1
        else:
            self._cone_enter_streak = 0

        # 측면/정면 창 — 디바운스는 '연속 스캔 수' 기준 (OVERTAKE 스텁 입력)
        half = math.radians(float(P("side_half_deg")))
        deb = int(P("side_debounce_scans"))
        left_hit = math.isfinite(self._scan_window(
            msg, math.radians(float(P("side_left_angle_deg"))), half,
            float(P("side_left_r_min")), float(P("side_left_r_max"))))
        right_hit = math.isfinite(self._scan_window(
            msg, math.radians(float(P("side_right_angle_deg"))), half,
            float(P("side_right_r_min")), float(P("side_right_r_max"))))
        if left_hit:
            self._left_on_s += 1
            self._left_off_s = 0
        else:
            self._left_off_s += 1
            self._left_on_s = 0
        if self._left_on_s >= deb:
            self._side_left = True
        elif self._left_off_s >= deb:
            self._side_left = False
        if right_hit:
            self._right_on_s += 1
            self._right_off_s = 0
        else:
            self._right_off_s += 1
            self._right_on_s = 0
        if self._right_on_s >= deb:
            self._side_right = True
        elif self._right_off_s >= deb:
            self._side_right = False
        self._front_min = self._scan_window(
            msg, 0.0, math.radians(float(P("front_half_deg"))),
            float(P("front_r_min")), float(P("front_r_max")))

    # ---------------- 메인 틱 (20Hz) ----------------

    def _tick(self):
        now = self._now()
        stamp = self.get_clock().now().to_msg()
        P = self._prm

        # 스냅샷 초기화 — 이번 틱에 해당 phase가 채운 것만 유효 (라치 금지)
        self._cmd = self._fresh_cmd()

        # 공통 선행 게이트: 270° 등가 통과 → 정지선 트리거 차단 타이머
        # (red_stop과 shortcut '양쪽'에 공통 적용 — 기존 비대칭 버그 수정)
        self._update_stop_block(now)

        if self.phase == "IDLE":
            if self._has_lane or self._scan_seq > 0:
                self.get_logger().info("WAIT — 센서 확인, 출발 신호 대기")
                self.phase = "WAIT"
            self._cmd["owner"] = Owner.MISSION_STOP
        elif self.phase == "WAIT":
            self._tick_wait(stamp, now)
        elif self.phase == "CONE":
            self._tick_cone(stamp, now)
        elif self.phase == "LANE":
            self._tick_lane_phase(stamp, now)
        elif self.phase == "PEDESTRIAN":
            self._tick_pedestrian(stamp, now)
        elif self.phase == "SHORTCUT":
            self._tick_shortcut(stamp, now)
        else:
            # OVERTAKE 스텁 포함 — 진입 경로 없음. 안전망으로 LANE 복귀.
            self.get_logger().warn(f"미지원 phase {self.phase} → LANE 복귀")
            self.phase = "LANE"

        # 공통 후행 (phase와 무관하게 매 틱 진행 — early-return 동결 금지)
        self._update_s_zone(now)
        self._update_corner_boost(now)
        # 차선 품질도 /detect/lane 신선도 적용 — 스테일이면 0.0 취급 (안전측 감속 cap)
        lane_q = self._lane_quality if self._lane_fresh(now) else 0.0
        if lane_q < float(P("lane_quality_min")):
            self._cmd["caps"].append(float(P("low_quality_speed_cap")))

        self._publish_drive_cmd()
        self._log_phase()

    def _log_phase(self):
        self._log_counter += 1
        if self._log_counter < PLAN_HZ:
            return
        self._log_counter = 0
        label = self.phase
        if self.phase == "SHORTCUT":
            label = {"WAITING": "SC WAIT", "TURNING_1": "SC LT1",
                     "FOLLOW": "SC RUN", "TURNING_2": "SC LT2"}.get(self._sc_sub, "SC")
        if self._red_stopping:
            label += " +STOP"
        if self._child_zone:
            label += " +CHILD"
        if self._s_zone_active:
            label += " +S"
        self.get_logger().info(label)

    # ---------------- WAIT ----------------

    def _tick_wait(self, stamp, now):
        """정지 + 경로 워밍업. GREEN 또는 /manual_go → 6틱 확인 후 출발(기존 유지)."""
        self._cmd["owner"] = Owner.MISSION_STOP
        self._tick_lane_publish(stamp)

        if self._wait_confirm > 0:
            self._wait_confirm -= 1
            if self._wait_confirm <= 0:
                self._start_drive(now)
            return

        if self._light == "GREEN" or self._take_manual_go(now):
            self._wait_confirm = int(self._prm("start_confirm_ticks"))
            self.get_logger().info(
                f"출발 신호 — {self._wait_confirm}틱 확인 후 출발")

    def _start_drive(self, now):
        """WAIT→LANE 직행 (예선의 무조건 CONE 진입 제거). start_grace는 시간 기준.
        /capture_yaw 발행으로 '출발 순간 = yaw 기준각'을 실제로 성립시킨다."""
        self.phase = "LANE"
        self._grace_until = now + float(self._prm("start_grace_sec"))
        # 출발 순간 기준각 캡처 — car_state_node가 /capture_yaw 수신 즉시 현재 yaw를 기준각으로 재캡처
        self._pub_capture_yaw.publish(Empty())
        car = self._car
        self.get_logger().info(
            "출발 → LANE. /capture_yaw 발행 — 이 순간 yaw가 출발 기준각으로 캡처됨 "
            f"(발행 전 상태: yaw_ref_set={int(car['yaw_ref_set'])}, "
            f"yaw_rel={car['yaw_rel_deg']:.1f}deg)")

    # ---------------- LANE ----------------

    def _tick_lane_phase(self, stamp, now):
        P = self._prm
        obj_fresh = self._objects_fresh(now)

        # CHILD ZONE (기본 비활성)
        if bool(P("enable_child_zone")):
            self._tick_child_zone(now, obj_fresh)
        if self._child_zone:
            self._cmd["caps"].append(float(P("child_speed_cap")))

        # PEDESTRIAN (기본 비활성 — S존 게이트 제거판)
        if bool(P("enable_pedestrian")) and self._check_pedestrian(now):
            self._cmd["owner"] = Owner.MISSION_STOP
            self._tick_lane_publish(stamp)
            return

        # 정지선 로직 — 출발 grace(시간) 이후에만
        if now >= self._grace_until:
            gate_ok = self._stop_gate_ok(now)
            if not self._red_stopping:
                # 지름길 우선 (원샷 래치 안 섰을 때만) — 그 외엔 일반 red stop
                self._check_shortcut(now, gate_ok, obj_fresh)
                if self.phase == "SHORTCUT":
                    self._cmd["owner"] = Owner.MISSION_STOP
                    self._tick_lane_publish(stamp)
                    return
            self._check_red_stop(now, gate_ok, obj_fresh)
        if self._red_stopping:
            self._cmd["owner"] = Owner.MISSION_STOP

        # CONE 진입 — LANE 중 감지 기반 (ROI 안 콘 클러스터 연속 스캔)
        # 정지선 정지 중엔 진입 안 함(_red_stopping 라치 이월 방지) + 종료 직후 재진입 차단
        if (self._cone_enter_streak >= int(P("cone_enter_scans"))
                and not self._red_stopping
                and now >= self._cone_reenter_until):
            self._enter_cone(now)
            self._tick_cone(stamp, now)
            return

        # OVERTAKE 스텁 — 활성화되면 오프셋/캡을 낼 자리 (현재 항상 None)
        ot = self._overtake.update(
            front_min_m=self._front_min, left_on=self._side_left,
            right_on=self._side_right, car_v=self._car_max_v, car_state=self._car)
        if ot:
            cap = float(ot.get("speed_cap", 0.0))
            if cap > 0.0:
                self._cmd["caps"].append(cap)
            # TODO(본선): ot["offset"]을 path_gen 기준선 오프셋으로 반영

        self._tick_lane_publish(stamp)

    # ---------------- 정지선 공통 게이트 + red stop + shortcut 진입 ----------------

    def _update_stop_block(self, now):
        """yaw_rel이 stop_block_heading_rel 부근이면 stop_block_sec 동안 정지선 차단.
        (기존 절대 270° 로직의 yaw_rel 등가 — red_stop/shortcut 공통)"""
        P = self._prm
        if not self._car["yaw_ref_set"]:
            return
        d = abs(_circ_diff_deg(self._car["yaw_rel_deg"], float(P("stop_block_heading_rel"))))
        if d <= float(P("stop_block_tol")):
            self._stop_block_until = now + float(P("stop_block_sec"))

    def _stop_gate_ok(self, now):
        """정지선 트리거 공통 게이트: yaw 기준각 설정 + heading 창 + 차단 타이머."""
        P = self._prm
        if not self._car["yaw_ref_set"]:
            return False              # 기준각 없인 heading 게이트 판정 불가 → 안전측 차단
        if now < self._stop_block_until:
            return False
        d = abs(_circ_diff_deg(self._car["yaw_rel_deg"], float(P("stop_heading_rel"))))
        return d <= float(P("stop_heading_tol"))

    def _check_red_stop(self, now, gate_ok, obj_fresh):
        """정지선 가까움 + GREEN 아님 → MISSION_STOP. GREEN으로만 해제.
        (기존 POLICE/LEFT 분기 제거 — 지름길 전환은 _check_shortcut 전담)"""
        P = self._prm
        if self._red_stopping:
            if self._light == "GREEN":
                self._red_stopping = False
                self.get_logger().info("GREEN → 출발 (red stop 해제)")
            return
        if not (gate_ok and obj_fresh):
            return
        if self._stop_max_v is None or self._stop_max_v < float(P("stop_v_threshold")):
            return
        if self._light == "GREEN":
            return   # 초록불이면 그냥 통과
        self._red_stopping = True
        self.get_logger().info("정지선 + GREEN 아님 → MISSION_STOP (GREEN으로 해제)")

    def _check_shortcut(self, now, gate_ok, obj_fresh):
        """정지선 가까움 → SHORTCUT.WAITING 진입 — 평생 1회 (원샷 래치)."""
        P = self._prm
        if self._shortcut_done:
            return
        if now < self._sc_retry_until:
            return   # WAITING 타임아웃 포기 직후 재진입 쿨다운 (같은 정지선 즉시 핑퐁 방지)
        if not (gate_ok and obj_fresh):
            return
        if self._stop_max_v is None or self._stop_max_v < float(P("stop_v_threshold")):
            return
        self._shortcut_done = True     # 진입 시점에 래치 — 중도 이탈해도 재진입 없음
        self.phase = "SHORTCUT"
        self._sc_sub = "WAITING"
        self._sc_started = now
        self._sc_left_hits = 0
        self.get_logger().info("SC WAIT — 정지. LEFT 디바운스 또는 /manual_go로 좌회전 확정")

    # ---------------- SHORTCUT ----------------

    def _tick_shortcut(self, stamp, now):
        P = self._prm
        self._cmd["caps"].append(float(P("shortcut_speed_cap")))

        if self._sc_sub == "WAITING":
            self._cmd["owner"] = Owner.MISSION_STOP
            self._tick_lane_publish(stamp)
            # LEFT 이벤트 디바운스 (기존 hit/감소 로직 유지) 또는 /manual_go 실소비
            if self._objects_fresh(now) and LEFT_SIGN_CLS_ID in self._obj_cls:
                self._sc_left_hits += 1
            else:
                self._sc_left_hits = max(0, self._sc_left_hits - 1)
            if (self._sc_left_hits >= int(P("left_debounce_hits"))
                    or self._take_manual_go(now)):
                self._sc_enter_turn("TURNING_1", now)
            elif (now - self._sc_started) >= float(P("sc_wait_timeout_sec")):
                # WAITING 타임아웃 — LEFT/manual_go 없음 → 지름길 포기, LANE 복귀(정지 해제).
                # 래치는 False로 되돌려 다음 기회에 재시도 가능하게 한다.
                self._shortcut_done = False
                self._sc_retry_until = now + float(P("sc_wait_timeout_sec"))
                self.phase = "LANE"
                self._sc_sub = None
                self.get_logger().warn(
                    "SC WAIT 타임아웃 — 지름길 포기 → LANE 복귀 (래치 해제, 재시도 가능)")

        elif self._sc_sub in ("TURNING_1", "TURNING_2"):
            # BLOCK: motion이 좌회전 실행, planner는 yaw 진행으로 종료 판정
            target = float(P("sc_turn_target_deg"))
            self._cmd["owner"] = Owner.BLOCK
            self._cmd["block_param"] = target
            self._tick_lane_publish(stamp)   # 경로는 계속 발행 (복귀 연속성/시각화)
            prog = abs(_circ_diff_deg(self._car["yaw_rel_deg"], self._sc_yaw_start))
            timeout = (now - self._sc_started) >= float(P("sc_turn_timeout_sec"))
            if prog >= target or timeout:
                if timeout:
                    self.get_logger().warn(
                        f"{self._sc_sub} 타임아웃 — |Δyaw|={prog:.1f}deg에서 강제 진행")
                if self._sc_sub == "TURNING_1":
                    self._sc_sub = "FOLLOW"
                    self._sc_started = now
                    self.get_logger().info(
                        f"SC FOLLOW — |Δyaw|={prog:.1f}deg 도달, owner LANE 복귀")
                else:
                    self.get_logger().info(f"SC LT2 완료 — |Δyaw|={prog:.1f}deg")
                    self._exit_shortcut_to_lane()

        elif self._sc_sub == "FOLLOW":
            # 지름길 차선 추종. CROSSROAD v임계 또는 하드 타임아웃(초) → TURNING_2
            self._tick_lane_publish(stamp)
            cross = (self._objects_fresh(now)
                     and self._cross_max_v is not None
                     and self._cross_max_v >= float(P("cross_v_threshold")))
            timeout = (now - self._sc_started) >= float(P("sc_follow_timeout_sec"))
            if cross or timeout:
                if timeout:
                    self.get_logger().warn("SC FOLLOW 하드 타임아웃 → TURNING_2 강제")
                self._sc_enter_turn("TURNING_2", now)

        else:
            self.get_logger().warn("SC 알 수 없는 서브상태 → LANE 복귀")
            self._exit_shortcut_to_lane()

    def _sc_enter_turn(self, sub, now):
        """TURNING 진입 — 새 block_id(+1=원샷) + yaw_rel 캡처."""
        self._sc_sub = sub
        self._sc_started = now
        self._block_id += 1
        self._sc_yaw_start = float(self._car["yaw_rel_deg"])
        self.get_logger().info(
            f"{sub} — BLOCK id={self._block_id}, 목표 |Δyaw|="
            f"{float(self._prm('sc_turn_target_deg')):.0f}deg, "
            f"기준 yaw_rel={self._sc_yaw_start:.1f}deg")

    def _exit_shortcut_to_lane(self):
        """SHORTCUT 종료 → LANE 복귀. (기존의 child_zone 강제 진입 제거)"""
        self.phase = "LANE"
        self._sc_sub = None
        self.get_logger().info("SHORTCUT 완료 → LANE (원샷 — 재진입 없음)")

    # ---------------- CONE ----------------

    def _enter_cone(self, now):
        self.get_logger().info(
            f"CONE 진입 — 콘 클러스터 {self._cone_enter_streak}연속 스캔 감지")
        self.phase = "CONE"
        self._cone_prev_fit = None
        self._cone_miss = 0
        self._cone_seen_seq = self._scan_seq
        self._cone_grace_until = now + float(self._prm("cone_grace_sec"))
        self._cone_enter_streak = 0
        self._lane_offset = 0.0        # 디버그 필드 정리 + LANE 복귀 시 0부터 램프
        self._path_gen.reset()

    def _cone_fit_attempt(self):
        """기존 _tick_cone 왼쪽 콘 피팅 이식 — 확정 트랙에서 좌측 콘 줄 2차/1차 피팅."""
        P = self._prm
        self._cone_left_count = 0
        pts = [(t["x"], t["y"]) for t in self._tracks if t["confirmed"]]
        if not pts:
            return None
        all_x = np.array([p[0] for p in pts], dtype=np.float64)
        all_y = np.array([p[1] for p in pts], dtype=np.float64)
        roi = ((all_x >= float(P("cone_x_min"))) & (all_x <= float(P("cone_x_max")))
               & (all_y >= float(P("cone_y_min"))) & (all_y <= float(P("cone_y_max"))))
        all_x, all_y = all_x[roi], all_y[roi]
        if all_x.size < int(P("cone_fit_min_points")):
            return None

        if self._cone_prev_fit is not None:
            expected_y = np.polyval(self._cone_prev_fit, all_x)
            left_mask = np.abs(all_y - expected_y) < float(P("cone_left_accept_dist"))
        else:
            left_mask = all_y > 0   # 첫 피팅은 좌측(y>0) 가정

        lx, ly = all_x[left_mask], all_y[left_mask]
        self._cone_left_count = int(lx.size)
        if lx.size < int(P("cone_fit_min_points")):
            return None
        x_span = float(np.max(lx) - np.min(lx)) if lx.size >= 2 else 0.0
        if x_span < float(P("cone_fit_min_x_span")):
            return None
        try:
            deg = 2 if lx.size >= 3 and x_span >= float(P("cone_fit_deg2_span")) else 1
            w = 1.0 / (1.0 + lx * lx)   # 가까운 콘 가중
            coef = np.polyfit(lx, ly, deg, w=w)
            if deg == 1:
                coef = np.array([0.0, coef[0], coef[1]])
            a, b, _ = coef
            if abs(a) <= float(P("cone_fit_curve_max")) and abs(b) <= float(P("cone_fit_slope_max")):
                return coef
        except (np.linalg.LinAlgError, ValueError):
            pass
        return None

    def _tick_cone(self, stamp, now):
        """라바콘 주행 — profile=CONE + speed_cap. /corridor_path 신선 시 기준선 우선."""
        P = self._prm
        self._cmd["steer_profile"] = SteerProfile.CONE
        self._cmd["caps"].append(float(P("cone_speed_cap")))

        corridor_fresh = (self._corridor is not None and len(self._corridor) >= 2
                          and (now - self._corridor_time) <= float(P("corridor_fresh_sec")))
        new_scan = self._scan_seq != self._cone_seen_seq
        if new_scan:
            self._cone_seen_seq = self._scan_seq
        in_grace = now < self._cone_grace_until

        # /scan 스테일 가드 — 동결 트랙으로 피팅 유지 금지 + 매 틱 miss 계수
        # (스캔 동결 시 miss가 안 늘어 CONE 탈출 불가가 되는 것 방지)
        scan_fresh = self._scan_fresh(now)
        raw_fit = self._cone_fit_attempt() if scan_fresh else None
        left_fit = None
        if raw_fit is None:
            if not scan_fresh:
                self._cone_miss += 1     # 스테일 — new_scan/grace 무관 매 틱 계수
            elif new_scan and not in_grace:
                self._cone_miss += 1     # miss는 '연속 스캔 수'로 계수 (틱 아님)
            if (self._cone_prev_fit is not None
                    and self._cone_miss < int(P("cone_exit_miss_scans"))):
                coast = self._cone_prev_fit.copy()
                coast[0] *= max(0.0, 1.0 - self._cone_miss * float(P("cone_coast_decay")))
                left_fit = coast
        else:
            alpha = float(P("cone_fit_smooth_alpha"))
            if self._cone_prev_fit is None:
                self._cone_prev_fit = np.asarray(raw_fit, dtype=np.float64)
            else:
                self._cone_prev_fit = ((1.0 - alpha) * self._cone_prev_fit
                                       + alpha * np.asarray(raw_fit, dtype=np.float64))
            left_fit = self._cone_prev_fit
            self._cone_miss = 0

        # 종료 — miss 연속 스캔 (기존 40틱 → 20스캔). corridor 신선하면 유지.
        if self._cone_miss >= int(P("cone_exit_miss_scans")) and not corridor_fresh:
            self.get_logger().info(f"CONE 종료(miss {self._cone_miss}스캔) → LANE")
            self.phase = "LANE"
            self._cone_prev_fit = None
            self._cone_reenter_until = now + float(P("cone_reenter_block_sec"))
            self._s_zone_block_until = now + float(P("s_zone_block_sec"))
            self._tick_lane_publish(stamp)   # 즉시 차선 경로 발행 (경로 끊김 방지)
            return

        # 기준선 1순위: 외부팀 콘 통로 (/corridor_path 신선 시)
        if corridor_fresh:
            self._publish_corridor(stamp)
            return

        n = int(P("cone_sample_n"))
        sample_xs = np.linspace(float(P("cone_sample_x_start")),
                                float(P("cone_sample_x_end")), n)
        if left_fit is None:
            # grace/미피팅 — 살짝 직진 (콘에 다가가기, 기존 유지)
            self._pub_center.publish(_poses_from_xy(stamp, sample_xs, np.zeros(n)))
            self._publish_target(stamp, TARGET_X, 0.0)
            return

        half = float(P("track_width")) / 2.0
        center_coef = left_fit.copy()
        center_coef[2] -= half
        if self._cone_left_count <= 1:
            center_coef[1] -= float(P("cone_single_bias"))   # 콘 1개 이하 — 시계방향 보정

        left_ys = np.polyval(left_fit, sample_xs)
        center_ys = np.clip(np.polyval(center_coef, sample_xs),
                            -TARGET_Y_LIMIT, TARGET_Y_LIMIT)
        right_ys = center_ys - half

        self._pub_left.publish(_poses_from_xy(stamp, sample_xs, left_ys))
        self._pub_right.publish(_poses_from_xy(stamp, sample_xs, right_ys))
        self._pub_center.publish(_poses_from_xy(stamp, sample_xs, center_ys))
        target_y = float(np.clip(np.polyval(center_coef, TARGET_X),
                                 -TARGET_Y_LIMIT, TARGET_Y_LIMIT))
        self._publish_target(stamp, TARGET_X, target_y)

    def _publish_corridor(self, stamp):
        """외부팀 콘 통로를 기준선으로 발행 (/center_path 규격 동일)."""
        xs = np.array([p[0] for p in self._corridor], dtype=np.float64)
        ys = np.array([p[1] for p in self._corridor], dtype=np.float64)
        self._pub_center.publish(_poses_from_xy(stamp, xs, ys))
        half = float(self._prm("track_width")) / 2.0
        self._pub_left.publish(_poses_from_xy(stamp, xs, ys + half))
        self._pub_right.publish(_poses_from_xy(stamp, xs, ys - half))
        order = np.argsort(xs)
        ty = float(np.clip(np.interp(TARGET_X, xs[order], ys[order]),
                           -TARGET_Y_LIMIT, TARGET_Y_LIMIT))
        self._publish_target(stamp, TARGET_X, ty)

    # ---------------- PEDESTRIAN (기본 비활성) ----------------

    def _check_pedestrian(self, now):
        """도로 안 사람 크기 클러스터 → 정지 전환. 밀집 다수=차량 → 무시.
        (기존 이식 — S존 게이트 제거판). True 반환 시 PEDESTRIAN 전환됨."""
        P = self._prm
        if now < self._ped_cooldown_until:
            return False
        if self._lane_center_coef is None:
            return False

        road_obs = []
        for t in self._tracks:
            if not t["confirmed"]:
                continue
            ox, oy, r = t["x"], t["y"], t["r"]
            if ox > float(P("ped_x_max")) or ox < float(P("ped_x_min")):
                continue
            if r < float(P("ped_r_min")) or r >= float(P("ped_r_max")):
                continue
            center_y = float(np.polyval(self._lane_center_coef, ox))
            if abs(oy - center_y) <= float(P("ped_road_half_width")):
                road_obs.append((ox, oy, r))
        if not road_obs:
            return False

        # 후보 존재 → center freeze (차선 가림 흔들림 방지)
        self._path_freeze_until = now + float(P("ped_freeze_sec"))

        dense = 0
        for i, (x1, y1, _) in enumerate(road_obs):
            for x2, y2, _ in road_obs[i + 1:]:
                if abs(x1 - x2) + abs(y1 - y2) < float(P("ped_car_spread")):
                    dense += 1
        if len(road_obs) >= int(P("ped_car_cluster_count")) and dense >= 2:
            return False   # 밀집 클러스터 다수 = 차량 → 무시

        self.get_logger().info("PED — 보행자 감지, 정지")
        self._ped_until = now + float(P("ped_stop_sec"))
        self.phase = "PEDESTRIAN"
        return True

    def _tick_pedestrian(self, stamp, now):
        """고정 시간 정지 후 무조건 재출발 (기존 유지 — 시간 기준으로)."""
        self._cmd["owner"] = Owner.MISSION_STOP
        self._tick_lane_publish(stamp)   # 피팅/발행 유지 (재출발 끊김 방지)
        if now >= self._ped_until:
            self.get_logger().info("PED 종료 → LANE")
            self._ped_cooldown_until = now + float(self._prm("ped_cooldown_sec"))
            self.phase = "LANE"

    # ---------------- CHILD ZONE (기본 비활성) ----------------

    def _tick_child_zone(self, now, obj_fresh):
        """어린이 보호구역 진입/해제 — 기존 이식 (틱 → 초 단위)."""
        P = self._prm
        if not self._child_zone:
            if now < self._child_exit_cd_until:
                return
            if (obj_fresh and self._child_start_max_v is not None
                    and self._child_start_max_v >= float(P("child_start_v_thr"))):
                self._child_zone = True
                self._child_enter_cd_until = now + float(P("child_enter_cooldown_sec"))
                self._child_deadline = now + float(P("child_hard_timeout_sec"))
                self._child_exit_at = None
                self.get_logger().info("CHILD ON")
            return

        if now >= self._child_deadline:
            self._child_off(now, "하드 타임아웃")
            return
        if self._child_exit_at is not None:
            if now >= self._child_exit_at:
                self._child_off(now, "END 지연 해제")
            return
        if now < self._child_enter_cd_until:
            return
        if (obj_fresh and self._child_end_max_v is not None
                and self._child_end_max_v >= float(P("child_end_v_thr"))):
            self._child_exit_at = now + float(P("child_exit_delay_sec"))
            self.get_logger().info("CHILD END 감지 — 지연 후 해제")

    def _child_off(self, now, reason):
        self._child_zone = False
        self._child_exit_cd_until = now + float(self._prm("child_exit_cooldown_sec"))
        self.get_logger().info(f"CHILD OFF ({reason})")

    # ---------------- 공통 cmd 갱신 ----------------

    def _update_s_zone(self, now):
        """S존 판정 — enable_s_zone일 때만, yaw_rel 기준 (절대각 재도입 금지)."""
        P = self._prm
        if not bool(P("enable_s_zone")) or not self._car["yaw_ref_set"]:
            self._s_zone_active = False
            return
        yaw = float(self._car["yaw_rel_deg"])
        raw = float(P("s_zone_yaw_lo")) <= yaw <= float(P("s_zone_yaw_hi"))
        active = raw and self.phase == "LANE" and now >= self._s_zone_block_until
        self._s_zone_active = active
        if active:
            self._cmd["steer_profile"] = SteerProfile.S_ZONE

    def _update_corner_boost(self, now):
        """child 종료 후 일정 시간 speed_floor (예선 corner_boost 대체 — 기본 비활성)."""
        P = self._prm
        if self._cb_prev_child and not self._child_zone:
            self._cb_until = now + float(P("corner_boost_window_sec"))
        self._cb_prev_child = self._child_zone
        floor = float(P("corner_boost_floor"))
        if floor > 0.0 and now < self._cb_until:
            self._cmd["floor"] = max(self._cmd["floor"], floor)

    def _publish_drive_cmd(self):
        """이번 틱 스냅샷 발행 — cap은 후보 최솟값, fixed는 cap에 종속."""
        c = self._cmd
        caps = [v for v in c["caps"] if v > 0.0]
        cap = min(caps) if caps else 0.0
        fixed = float(c["fixed"])
        if cap > 0.0 and fixed > cap:
            fixed = cap   # speed_fixed는 cap에 종속 (예선 '무조건 대입' 버그 재발 방지)
        msg = Float32MultiArray()
        msg.data = encode_drive_cmd({
            "owner": float(c["owner"]),
            "steer_profile": float(c["steer_profile"]),
            "speed_cap": cap,
            "speed_floor": float(c["floor"]),
            "speed_fixed": fixed,
            "block_id": float(self._block_id if c["owner"] == Owner.BLOCK else 0),
            "block_param": float(c["block_param"]),
            "lane_offset": float(self._lane_offset),
        })
        self._pub_cmd.publish(msg)

    # ---------------- LANE 경로 생성/발행 ----------------

    def _tick_lane_publish(self, stamp):
        """기존 _tick_lane 이식 — plan()엔 빈 장애물(내부 시프트 비활성),
        회피는 path_gen 오프셋 최적화가 전담 (이중회피 방지 패턴 전 구간 적용)."""
        # 인지 소실 가드: /detect/lane 스테일이면 /center_path 발행 중단 —
        # motion의 path 워치독(path_stale_sec)이 감속 정지시킨다 (설계된 폴백).
        # /drive_cmd는 계속 나간다 (_publish_drive_cmd는 _tick에서 무조건 호출).
        if not self._lane_fresh(self._now()):
            if self._has_lane:   # 최초 수신 전 기동 소음 방지 — 소실만 경고
                self.get_logger().warning(
                    "/detect/lane 스테일 — /center_path 발행 중단 (drive_cmd만 유지)",
                    throttle_duration_sec=1.0)
            return
        if not self._has_lane:
            self._publish_last_lane(stamp)
            return
        result = lane_plan(self._yellow_xs, self._yellow_ys,
                           self._white_xs, self._white_ys,
                           [],   # 빈 장애물 — plan() 내부 0.8m 시프트 비활성
                           child_zone=self._child_zone,
                           yellow_dash_xs=self._yellow_dash_xs,
                           yellow_dash_ys=self._yellow_dash_ys)
        if result is None:
            self._publish_last_lane(stamp)
            return
        self._last_lane_result = self._copy_lane_result(result)
        self._publish_lane_result(stamp, result)

    @staticmethod
    def _copy_lane_result(result):
        return {
            "target": tuple(result["target"]),
            "sample_xs": np.asarray(result["sample_xs"], dtype=np.float64).copy(),
            "sample_ys": np.asarray(result["sample_ys"], dtype=np.float64).copy(),
            "left_ys": np.asarray(result["left_ys"], dtype=np.float64).copy(),
            "right_ys": np.asarray(result["right_ys"], dtype=np.float64).copy(),
            "all_fits": [
                np.asarray(coef, dtype=np.float64).copy()
                for coef in result.get("all_fits", [])
            ],
        }

    def _publish_last_lane(self, stamp):
        """차선 미검출 시 직전 결과 재발행 (경로 공백 방지)."""
        if self._last_lane_result is None:
            return False
        self._publish_lane_result(stamp, self._last_lane_result)
        return True

    def _publish_lane_result(self, stamp, result):
        """기존 이식 + 수술: 기준선에 path_gen(확정 장애물) 오프셋 적용 후 발행."""
        P = self._prm
        all_fits = result.get("all_fits", [])
        if all_fits:
            self._lane_center_coef = all_fits[0]   # yellow_fit = center (PED 기준선)

        target_x, target_y = result["target"]
        sample_xs = np.asarray(result["sample_xs"], dtype=np.float64)
        sample_ys = np.asarray(result["sample_ys"], dtype=np.float64)

        # 확정(confirmed) 클러스터 중 회피 대상만 → path_gen (램프/히스테리시스 내장)
        # /scan 스테일이면 동결된 트랙으로 회피하지 않도록 빈 리스트
        if self._scan_fresh(self._now()):
            obstacles = [
                {"x": t["x"], "y": t["y"], "r": t["r"]}
                for t in self._tracks
                if t["confirmed"]
                and 0.0 < t["x"] <= float(P("avoid_obs_x_max"))
                and t["r"] <= float(P("avoid_obs_r_max"))   # 벽 등 대형 클러스터 제외
            ]
        else:
            obstacles = []
        gen = self._path_gen.generate(sample_xs, sample_ys, obstacles)
        self._lane_offset = float(gen["offset"])
        pub_ys = np.clip(gen["ys"], -TARGET_Y_LIMIT, TARGET_Y_LIMIT)
        target_y = float(np.clip(target_y + self._lane_offset,
                                 -TARGET_Y_LIMIT, TARGET_Y_LIMIT))

        # center freeze — 보행자 감지 중 직전 경로 재발행 (기존 이식, 시간 기준)
        if self._now() < self._path_freeze_until and self._frozen_center is not None:
            pub_xs, pub_ys = self._frozen_center
        else:
            pub_xs = sample_xs
            self._frozen_center = (pub_xs, pub_ys)

        offset = self._lane_offset
        left_ys = np.clip(np.asarray(result["left_ys"], dtype=np.float64) + offset,
                          -TARGET_Y_LIMIT, TARGET_Y_LIMIT)
        right_ys = np.clip(np.asarray(result["right_ys"], dtype=np.float64) + offset,
                           -TARGET_Y_LIMIT, TARGET_Y_LIMIT)

        self._publish_target(stamp, target_x, target_y)
        self._pub_center.publish(_poses_from_xy(stamp, pub_xs, pub_ys))
        self._pub_left.publish(_poses_from_xy(stamp, sample_xs, left_ys))
        self._pub_right.publish(_poses_from_xy(stamp, sample_xs, right_ys))

    # ---------------- 공통 ----------------

    def _publish_target(self, stamp, x, y):
        t = PointStamped()
        t.header.stamp = stamp
        t.header.frame_id = "lidar_frame"
        t.point.x = float(x)
        t.point.y = float(y)
        self._pub_target.publish(t)


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
