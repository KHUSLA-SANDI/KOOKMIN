"""Float32MultiArray 토픽 레이아웃 계약 — /drive_cmd, /car_state, /vesc/state, /teleop_cmd.

역할: 토픽별 인덱스 상수 + enum(Owner, SteerProfile) 정의, encode/decode (단일 진실 공급원)
입력: encode_* — 필드 dict / decode_* — Float32MultiArray.data로 온 list(float)
출력: encode_* — 길이 고정 list[float] / decode_* — 필드 dict (길이 부족·비정상 값은 안전 기본값)
예선 대비 변경점: 신규 파일. 예선의 암묵적 다중 토픽(/limit_speed 등 Bool 라치, /target
  keepalive)을 명시적 스냅샷 레이아웃으로 대체 (재도입 금지 버그 3·4·5 대응).
  rclpy import 금지 — 노드는 msg.data를 그대로 이 모듈에 넘긴다.
"""

import math
from enum import IntEnum


# ============================================================
# 신선도 타임아웃 (토픽 계약의 일부)
# ============================================================

DRIVE_CMD_STALE_SEC = 0.3   # /drive_cmd 미수신 → motion 감속 정지
TELEOP_STALE_SEC = 0.3      # /teleop_cmd 미수신 → 텔레옵 비활성
VESC_FRESH_SEC = 0.5        # /vesc/state 신선도 (car_state의 vesc_fresh 판정)


# ============================================================
# enum — /drive_cmd [0], [1]
# ============================================================

class Owner(IntEnum):
    """/drive_cmd [0] — 조향/속도 명령의 소유 주체."""
    LANE = 0
    BLOCK = 1
    MISSION_STOP = 2
    SAFETY_STOP = 3


class SteerProfile(IntEnum):
    """/drive_cmd [1] — motion이 적용할 조향 특성 프로파일."""
    NORMAL = 0
    CONE = 1
    S_ZONE = 2


def owner_from_value(v):
    """수신 float → Owner. 알 수 없는 값은 fail-safe로 SAFETY_STOP."""
    try:
        return Owner(int(round(float(v))))
    except (ValueError, TypeError):
        return Owner.SAFETY_STOP


def steer_profile_from_value(v):
    """수신 float → SteerProfile. 알 수 없는 값은 NORMAL."""
    try:
        return SteerProfile(int(round(float(v))))
    except (ValueError, TypeError):
        return SteerProfile.NORMAL


# ============================================================
# /drive_cmd (planner → motion, 20Hz 스냅샷, 라치 금지)
# ============================================================

CMD_OWNER = 0          # Owner
CMD_STEER_PROFILE = 1  # SteerProfile
CMD_SPEED_CAP = 2      # 명령단위. <=0 이면 없음
CMD_SPEED_FLOOR = 3    # <=0 이면 없음
CMD_SPEED_FIXED = 4    # <=0 이면 없음. cap에 종속
CMD_BLOCK_ID = 5       # 0=없음. 새 블록마다 +1 (같은 id 재수신 무시 = 원샷)
CMD_BLOCK_PARAM = 6    # 좌회전 목표 |Δyaw| deg 등
CMD_LANE_OFFSET = 7    # m, 디버그용 (planner가 적용 중인 횡오프셋)
DRIVE_CMD_LEN = 8

# 필드 누락 시 기본값 — owner는 fail-safe로 SAFETY_STOP (잘린 메시지 = 정지)
_DRIVE_CMD_DEFAULTS = {
    "owner": float(Owner.SAFETY_STOP),
    "steer_profile": float(SteerProfile.NORMAL),
    "speed_cap": 0.0,
    "speed_floor": 0.0,
    "speed_fixed": 0.0,
    "block_id": 0.0,
    "block_param": 0.0,
    "lane_offset": 0.0,
}

_DRIVE_CMD_KEYS = (
    "owner", "steer_profile", "speed_cap", "speed_floor",
    "speed_fixed", "block_id", "block_param", "lane_offset",
)


def _field(data, idx, default):
    """data[idx]를 float로. 없거나 비정상(NaN/inf)이면 default."""
    try:
        v = float(data[idx])
    except (IndexError, TypeError, ValueError):
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return v


def _encode(keys, defaults, fields):
    return [float(fields.get(k, defaults[k])) for k in keys]


def encode_drive_cmd(fields):
    """dict → /drive_cmd data 리스트. 누락 필드는 기본값 (owner 누락 = SAFETY_STOP)."""
    return _encode(_DRIVE_CMD_KEYS, _DRIVE_CMD_DEFAULTS, fields)


def decode_drive_cmd(data):
    """/drive_cmd data → dict. 길이 검증 후 부족분은 안전 기본값.

    반환 dict의 valid는 '전체 길이가 계약대로 왔는가'. False여도 나머지 필드는
    기본값으로 채워져 있어 그대로 써도 안전하다 (owner=SAFETY_STOP → 정지).
    """
    out = {
        key: _field(data, idx, _DRIVE_CMD_DEFAULTS[key])
        for idx, key in enumerate(_DRIVE_CMD_KEYS)
    }
    out["owner"] = owner_from_value(out["owner"])
    out["steer_profile"] = steer_profile_from_value(out["steer_profile"])
    out["block_id"] = int(round(out["block_id"]))
    try:
        out["valid"] = len(data) >= DRIVE_CMD_LEN
    except TypeError:
        out["valid"] = False
    return out


# ============================================================
# /car_state (car_state → 모두, 20Hz)
# ============================================================

CS_YAW_REL_DEG = 0   # 출발 기준각 대비, wrap-safe -180~180, CCW+
CS_YAW_RATE_DPS = 1
CS_SPEED_MS = 2      # eRPM / 4614
CS_DIST_M = 3        # 타코미터 × tacho_scale (scale=0이면 eRPM 적분 폴백)
CS_VOLTAGE = 4
CS_FAULT_CODE = 5
CS_VESC_FRESH = 6    # 0/1 (0.5s)
CS_YAW_REF_SET = 7   # 0/1
CAR_STATE_LEN = 8

# vesc_fresh=0, yaw_ref_set=0 이 안전측 (미수신 = 신선하지 않음)
_CAR_STATE_DEFAULTS = {
    "yaw_rel_deg": 0.0,
    "yaw_rate_dps": 0.0,
    "speed_ms": 0.0,
    "dist_m": 0.0,
    "voltage": 0.0,
    "fault_code": 0.0,
    "vesc_fresh": 0.0,
    "yaw_ref_set": 0.0,
}

_CAR_STATE_KEYS = (
    "yaw_rel_deg", "yaw_rate_dps", "speed_ms", "dist_m",
    "voltage", "fault_code", "vesc_fresh", "yaw_ref_set",
)


def encode_car_state(fields):
    return _encode(_CAR_STATE_KEYS, _CAR_STATE_DEFAULTS, fields)


def decode_car_state(data):
    out = {
        key: _field(data, idx, _CAR_STATE_DEFAULTS[key])
        for idx, key in enumerate(_CAR_STATE_KEYS)
    }
    out["vesc_fresh"] = out["vesc_fresh"] >= 0.5
    out["yaw_ref_set"] = out["yaw_ref_set"] >= 0.5
    try:
        out["valid"] = len(data) >= CAR_STATE_LEN
    except TypeError:
        out["valid"] = False
    return out


# ============================================================
# /vesc/state (도커 ROS1 relay → 브릿지)
# ============================================================

VESC_VOLTAGE = 0
VESC_CURRENT_MOTOR = 1
VESC_ERPM = 2
VESC_DUTY = 3
VESC_TACHO = 4       # displacement
VESC_FAULT_CODE = 5
VESC_STATE_LEN = 6

_VESC_STATE_DEFAULTS = {
    "voltage": 0.0,
    "current_motor": 0.0,
    "erpm": 0.0,
    "duty": 0.0,
    "tacho": 0.0,
    "fault_code": 0.0,
}

_VESC_STATE_KEYS = (
    "voltage", "current_motor", "erpm", "duty", "tacho", "fault_code",
)


def encode_vesc_state(fields):
    return _encode(_VESC_STATE_KEYS, _VESC_STATE_DEFAULTS, fields)


def decode_vesc_state(data):
    out = {
        key: _field(data, idx, _VESC_STATE_DEFAULTS[key])
        for idx, key in enumerate(_VESC_STATE_KEYS)
    }
    try:
        out["valid"] = len(data) >= VESC_STATE_LEN
    except TypeError:
        out["valid"] = False
    return out


# ============================================================
# /teleop_cmd (수동 조작, 0.3s 미수신 시 비활성)
# ============================================================

TELEOP_ANGLE = 0
TELEOP_SPEED = 1
TELEOP_ESTOP = 2     # 0/1
TELEOP_LEN = 3

# estop 누락 기본 1.0 — 잘린 텔레옵 메시지는 정지로 처리 (fail-safe)
_TELEOP_DEFAULTS = {
    "angle": 0.0,
    "speed": 0.0,
    "estop": 1.0,
}

_TELEOP_KEYS = ("angle", "speed", "estop")


def encode_teleop_cmd(fields):
    """encode는 estop 기본 0 — 보내는 쪽 누락은 '정상 주행 의도'로 본다."""
    defaults = dict(_TELEOP_DEFAULTS)
    defaults["estop"] = 0.0
    return _encode(_TELEOP_KEYS, defaults, fields)


def decode_teleop_cmd(data):
    out = {
        key: _field(data, idx, _TELEOP_DEFAULTS[key])
        for idx, key in enumerate(_TELEOP_KEYS)
    }
    out["estop"] = out["estop"] >= 0.5
    try:
        out["valid"] = len(data) >= TELEOP_LEN
    except TypeError:
        out["valid"] = False
    return out
