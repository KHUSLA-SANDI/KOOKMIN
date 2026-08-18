"""track_drive.lib.drive_cmd 레이아웃/인코딩 테스트 (rclpy 불필요)."""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from track_drive.lib import drive_cmd as dc  # noqa: E402


# ============================================================
# 계약 고정 — 인덱스/enum 값이 문서와 정확히 일치해야 한다
# ============================================================

def test_drive_cmd_layout_contract():
    assert dc.CMD_OWNER == 0
    assert dc.CMD_STEER_PROFILE == 1
    assert dc.CMD_SPEED_CAP == 2
    assert dc.CMD_SPEED_FLOOR == 3
    assert dc.CMD_SPEED_FIXED == 4
    assert dc.CMD_BLOCK_ID == 5
    assert dc.CMD_BLOCK_PARAM == 6
    assert dc.CMD_LANE_OFFSET == 7
    assert dc.DRIVE_CMD_LEN == 8


def test_enum_contract():
    assert dc.Owner.LANE == 0
    assert dc.Owner.BLOCK == 1
    assert dc.Owner.MISSION_STOP == 2
    assert dc.Owner.SAFETY_STOP == 3
    assert dc.SteerProfile.NORMAL == 0
    assert dc.SteerProfile.CONE == 1
    assert dc.SteerProfile.S_ZONE == 2


def test_car_state_layout_contract():
    assert dc.CS_YAW_REL_DEG == 0
    assert dc.CS_YAW_RATE_DPS == 1
    assert dc.CS_SPEED_MS == 2
    assert dc.CS_DIST_M == 3
    assert dc.CS_VOLTAGE == 4
    assert dc.CS_FAULT_CODE == 5
    assert dc.CS_VESC_FRESH == 6
    assert dc.CS_YAW_REF_SET == 7
    assert dc.CAR_STATE_LEN == 8


def test_vesc_teleop_layout_contract():
    assert dc.VESC_VOLTAGE == 0
    assert dc.VESC_CURRENT_MOTOR == 1
    assert dc.VESC_ERPM == 2
    assert dc.VESC_DUTY == 3
    assert dc.VESC_TACHO == 4
    assert dc.VESC_FAULT_CODE == 5
    assert dc.VESC_STATE_LEN == 6
    assert dc.TELEOP_ANGLE == 0
    assert dc.TELEOP_SPEED == 1
    assert dc.TELEOP_ESTOP == 2
    assert dc.TELEOP_LEN == 3


# ============================================================
# /drive_cmd encode/decode
# ============================================================

def test_drive_cmd_roundtrip():
    fields = {
        "owner": dc.Owner.BLOCK,
        "steer_profile": dc.SteerProfile.CONE,
        "speed_cap": 5.0,
        "speed_floor": 7.2,
        "speed_fixed": 6.0,
        "block_id": 3,
        "block_param": 90.0,
        "lane_offset": -0.35,
    }
    data = dc.encode_drive_cmd(fields)
    assert len(data) == dc.DRIVE_CMD_LEN
    assert all(isinstance(v, float) for v in data)
    assert data[dc.CMD_OWNER] == 1.0
    assert data[dc.CMD_STEER_PROFILE] == 1.0

    out = dc.decode_drive_cmd(data)
    assert out["valid"] is True
    assert out["owner"] is dc.Owner.BLOCK
    assert out["steer_profile"] is dc.SteerProfile.CONE
    assert out["speed_cap"] == 5.0
    assert out["speed_floor"] == 7.2
    assert out["speed_fixed"] == 6.0
    assert out["block_id"] == 3
    assert isinstance(out["block_id"], int)
    assert out["block_param"] == 90.0
    assert abs(out["lane_offset"] - (-0.35)) < 1e-9


def test_drive_cmd_encode_missing_owner_is_safety_stop():
    # 필드 누락 인코딩은 fail-safe (정지)
    data = dc.encode_drive_cmd({})
    assert data[dc.CMD_OWNER] == float(dc.Owner.SAFETY_STOP)
    assert data[dc.CMD_SPEED_CAP] == 0.0
    assert data[dc.CMD_BLOCK_ID] == 0.0


def test_drive_cmd_decode_short_list_uses_safe_defaults():
    out = dc.decode_drive_cmd([0.0, 1.0])  # owner=LANE, profile=CONE 까지만 도착
    assert out["valid"] is False
    assert out["owner"] is dc.Owner.LANE
    assert out["steer_profile"] is dc.SteerProfile.CONE
    assert out["speed_cap"] == 0.0
    assert out["speed_floor"] == 0.0
    assert out["speed_fixed"] == 0.0
    assert out["block_id"] == 0
    assert out["lane_offset"] == 0.0


def test_drive_cmd_decode_empty_is_safety_stop():
    out = dc.decode_drive_cmd([])
    assert out["valid"] is False
    assert out["owner"] is dc.Owner.SAFETY_STOP


def test_drive_cmd_decode_unknown_enum_values():
    data = dc.encode_drive_cmd({"owner": 9.0, "steer_profile": 7.0})
    out = dc.decode_drive_cmd(data)
    assert out["owner"] is dc.Owner.SAFETY_STOP  # 모르는 owner → 정지
    assert out["steer_profile"] is dc.SteerProfile.NORMAL


def test_drive_cmd_decode_nan_falls_back():
    data = [float("nan")] * dc.DRIVE_CMD_LEN
    out = dc.decode_drive_cmd(data)
    assert out["owner"] is dc.Owner.SAFETY_STOP
    assert out["speed_cap"] == 0.0
    assert out["block_id"] == 0


def test_drive_cmd_decode_extra_elements_tolerated():
    # 앞으로 필드가 늘어나도 구버전 디코더가 죽지 않는다
    data = dc.encode_drive_cmd({"owner": dc.Owner.LANE}) + [42.0, 43.0]
    out = dc.decode_drive_cmd(data)
    assert out["valid"] is True
    assert out["owner"] is dc.Owner.LANE


# ============================================================
# /car_state, /vesc/state, /teleop_cmd
# ============================================================

def test_car_state_roundtrip():
    fields = {
        "yaw_rel_deg": -179.5,
        "yaw_rate_dps": 12.0,
        "speed_ms": 1.6,
        "dist_m": 25.3,
        "voltage": 10.4,
        "fault_code": 0.0,
        "vesc_fresh": 1.0,
        "yaw_ref_set": 1.0,
    }
    data = dc.encode_car_state(fields)
    assert len(data) == dc.CAR_STATE_LEN
    out = dc.decode_car_state(data)
    assert out["valid"] is True
    assert out["yaw_rel_deg"] == -179.5
    assert out["speed_ms"] == 1.6
    assert out["vesc_fresh"] is True
    assert out["yaw_ref_set"] is True


def test_car_state_short_list_is_stale():
    out = dc.decode_car_state([0.0, 0.0, 1.2])
    assert out["valid"] is False
    assert out["speed_ms"] == 1.2
    assert out["vesc_fresh"] is False   # 미도착 = 신선하지 않음 (안전측)
    assert out["yaw_ref_set"] is False


def test_vesc_state_roundtrip():
    fields = {
        "voltage": 10.8,
        "current_motor": 3.2,
        "erpm": 4614.0,
        "duty": 0.15,
        "tacho": 1234.0,
        "fault_code": 0.0,
    }
    data = dc.encode_vesc_state(fields)
    assert len(data) == dc.VESC_STATE_LEN
    out = dc.decode_vesc_state(data)
    assert out["valid"] is True
    assert out["erpm"] == 4614.0
    assert out["tacho"] == 1234.0


def test_teleop_roundtrip_and_truncated_estop():
    data = dc.encode_teleop_cmd({"angle": -20.0, "speed": 5.0})
    assert len(data) == dc.TELEOP_LEN
    assert data[dc.TELEOP_ESTOP] == 0.0  # 보내는 쪽 누락 = 정상 주행 의도

    out = dc.decode_teleop_cmd(data)
    assert out["valid"] is True
    assert out["angle"] == -20.0
    assert out["speed"] == 5.0
    assert out["estop"] is False

    # 수신 측에서 estop 필드가 잘려 도착하면 fail-safe로 정지
    out = dc.decode_teleop_cmd([-20.0, 5.0])
    assert out["valid"] is False
    assert out["estop"] is True


def test_helpers_reject_garbage():
    assert dc.owner_from_value(float("nan")) is dc.Owner.SAFETY_STOP
    assert dc.owner_from_value(None) is dc.Owner.SAFETY_STOP
    assert dc.owner_from_value(2.0) is dc.Owner.MISSION_STOP
    assert dc.steer_profile_from_value(None) is dc.SteerProfile.NORMAL
    assert dc.steer_profile_from_value(2.4) is dc.SteerProfile.S_ZONE


def test_stale_timeouts_contract():
    assert math.isclose(dc.DRIVE_CMD_STALE_SEC, 0.3)
    assert math.isclose(dc.TELEOP_STALE_SEC, 0.3)
    assert math.isclose(dc.VESC_FRESH_SEC, 0.5)
