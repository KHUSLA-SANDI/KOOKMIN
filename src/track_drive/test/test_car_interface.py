"""car_interface.CarInterface 유닛테스트 — 순수 파이썬 (rclpy/ROS 불필요).

colcon test와 소스 트리 직접 pytest 실행 양쪽에서 돌도록 경로 폴백 포함.
검증 항목: 트림 적용 / 좌우 비대칭 클램프 / 데드존 스냅 양방향 / slew 상한 / reset.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from track_drive.lib.car_interface import DEFAULT_CFG, CarInterface  # noqa: E402


# slew 간섭 없이 정적 매핑만 볼 때 쓰는 cfg
NO_SLEW = {"slew_angle_per_tick": 1e9, "slew_speed_per_tick": 1e9}


def make(**overrides):
    cfg = dict(NO_SLEW)
    cfg.update(overrides)
    return CarInterface(cfg)


# ============================================================
# 조향 — 트림 / 비대칭 클램프
# ============================================================

class TestSteer:
    def test_trim_applied_at_zero(self):
        # angle_cmd=0 → rel=0 → 출력은 트림 그대로
        ci = make()
        angle, _ = ci.to_motor(0.0, 0.0)
        assert angle == pytest.approx(0.0)

    def test_scale_then_trim(self):
        # 양수 입력은 오른쪽 전용 스케일을 사용한다.
        ci = make()
        angle, _ = ci.to_motor(10.0, 0.0)
        assert angle == pytest.approx(10.0 * DEFAULT_CFG["steer_scale_right"])

    def test_asymmetric_clamp_left(self):
        # 좌측 풀조향: rel이 -32로 클램프
        ci = make()
        angle, _ = ci.to_motor(-1000.0, 0.0)
        assert angle == pytest.approx(DEFAULT_CFG["steer_limit_left"])

    def test_asymmetric_clamp_right(self):
        # 우측 풀조향: rel이 +62로 클램프
        ci = make()
        angle, _ = ci.to_motor(1000.0, 0.0)
        assert angle == pytest.approx(DEFAULT_CFG["steer_limit_right"])

    def test_clamp_is_relative_to_trim(self):
        # 트림을 바꾸면 절대 출력 한계도 같이 이동해야 한다
        ci = make(steer_trim=0.0)
        left, _ = ci.to_motor(-1000.0, 0.0)
        ci.reset()
        right, _ = ci.to_motor(1000.0, 0.0)
        assert left == pytest.approx(DEFAULT_CFG["steer_limit_left"])
        assert right == pytest.approx(DEFAULT_CFG["steer_limit_right"])

    def test_logical_endpoints_use_full_vesc_range(self):
        ci = make()
        left, _ = ci.to_motor(-100.0, 0.0)
        ci.reset()
        right, _ = ci.to_motor(100.0, 0.0)
        assert left == pytest.approx(DEFAULT_CFG["steer_limit_left"])
        assert right == pytest.approx(DEFAULT_CFG["steer_limit_right"])
        assert ci.steering_logical(right) == pytest.approx(100.0)


# ============================================================
# 속도 — 클립 / 데드존 스냅
# ============================================================

class TestSpeedStatic:
    def test_clip_no_reverse(self):
        ci = make()
        assert ci.to_motor(0.0, -5.0)[1] == pytest.approx(0.0)
        ci.reset()
        assert ci.to_motor(0.0, 150.0)[1] == pytest.approx(100.0)

    def test_deadzone_snap_up(self):
        # deadzone/2(=2) 이상 ~ deadzone 미만 → deadzone(4)으로 스냅
        ci = make()
        assert ci.to_motor(0.0, 3.0)[1] == pytest.approx(4.0)

    def test_deadzone_snap_up_boundary(self):
        # 정확히 deadzone/2 → 스냅 (>= 조건)
        ci = make()
        assert ci.to_motor(0.0, 2.0)[1] == pytest.approx(4.0)

    def test_deadzone_snap_down(self):
        # deadzone/2 미만 → 0으로 스냅 (정지 의도)
        ci = make()
        assert ci.to_motor(0.0, 1.9)[1] == pytest.approx(0.0)

    def test_deadzone_passthrough(self):
        # deadzone 이상은 그대로
        ci = make()
        assert ci.to_motor(0.0, 4.0)[1] == pytest.approx(4.0)
        assert ci.to_motor(0.0, 10.0)[1] == pytest.approx(10.0)

    def test_zero_stays_zero(self):
        ci = make()
        assert ci.to_motor(0.0, 0.0)[1] == pytest.approx(0.0)

    def test_reverse_clip_and_snap_symmetric(self):
        # allow_reverse=True: [-100,100] 클립 + 크기 기준 대칭 스냅
        ci = make(allow_reverse=True)
        assert ci.to_motor(0.0, -150.0)[1] == pytest.approx(-100.0)
        ci.reset()
        assert ci.to_motor(0.0, -3.0)[1] == pytest.approx(-4.0)
        ci.reset()
        assert ci.to_motor(0.0, -1.9)[1] == pytest.approx(0.0)


# ============================================================
# slew 상한 (기본 cfg: angle 8/tick, speed 1.5/tick)
# ============================================================

class TestSlew:
    def test_angle_first_tick_free_then_limited(self):
        ci = CarInterface()
        # 첫 틱: 기준 출력이 없어 slew 미적용 → 우측 풀조향 즉시
        angle, _ = ci.to_motor(1000.0, 0.0)
        right = DEFAULT_CFG["steer_limit_right"]
        assert angle == pytest.approx(right)
        # 둘째 틱: 좌측 풀조향 요청도 틱당 8만 이동
        angle, _ = ci.to_motor(-1000.0, 0.0)
        assert angle == pytest.approx(right - 8.0)
        angle, _ = ci.to_motor(-1000.0, 0.0)
        assert angle == pytest.approx(right - 16.0)

    def test_angle_slew_both_directions(self):
        ci = CarInterface()
        ci.to_motor(0.0, 0.0)                     # 기준: 0
        angle, _ = ci.to_motor(1000.0, 0.0)       # +방향 상한
        assert angle == pytest.approx(8.0)
        angle, _ = ci.to_motor(-1000.0, 0.0)      # -방향 상한
        assert angle == pytest.approx(0.0)

    def test_angle_within_slew_not_altered(self):
        ci = CarInterface()
        ci.to_motor(0.0, 0.0)                     # 기준: 0
        angle, _ = ci.to_motor(10.0, 0.0)         # 변화 약 5.86 < 8
        assert angle == pytest.approx(
            10.0 * DEFAULT_CFG["steer_scale_right"])

    def test_speed_ramps_from_standstill(self):
        # 센서리스: 정지→고속 직행 불가. 0부터 틱당 1.5씩 램프
        ci = CarInterface()
        assert ci.to_motor(0.0, 100.0)[1] == pytest.approx(1.5)
        assert ci.to_motor(0.0, 100.0)[1] == pytest.approx(3.0)
        assert ci.to_motor(0.0, 100.0)[1] == pytest.approx(4.5)

    def test_speed_slew_down(self):
        ci = CarInterface()
        for _ in range(4):
            ci.to_motor(0.0, 100.0)               # 6.0까지 램프
        assert ci.last_speed_out == pytest.approx(6.0)
        assert ci.to_motor(0.0, 0.0)[1] == pytest.approx(4.5)
        assert ci.to_motor(0.0, 0.0)[1] == pytest.approx(3.0)

    def test_snap_applied_before_slew(self):
        # 스냅(3→4)이 먼저, 그 결과에 램프 → 첫 틱은 1.5
        ci = CarInterface()
        assert ci.to_motor(0.0, 3.0)[1] == pytest.approx(1.5)


# ============================================================
# reset / 마지막 발행값 프로퍼티
# ============================================================

class TestResetAndState:
    def test_last_out_tracks_published(self):
        ci = CarInterface()
        angle, speed = ci.to_motor(50.0, 10.0)
        assert ci.last_angle_out == pytest.approx(angle)
        assert ci.last_speed_out == pytest.approx(speed)

    def test_initial_state(self):
        ci = CarInterface()
        assert ci.last_angle_out is None
        assert ci.last_speed_out == pytest.approx(0.0)

    def test_reset_clears_slew_state(self):
        ci = CarInterface()
        for _ in range(10):
            ci.to_motor(1000.0, 100.0)
        ci.reset()
        assert ci.last_angle_out is None
        assert ci.last_speed_out == pytest.approx(0.0)
        # reset 후: 조향은 첫 틱 자유, 속도는 0부터 다시 램프
        angle, speed = ci.to_motor(-1000.0, 100.0)
        assert angle == pytest.approx(DEFAULT_CFG["steer_limit_left"])
        assert speed == pytest.approx(1.5)


# ============================================================
# 단위 변환 / 입력 방어
# ============================================================

class TestMisc:
    def test_cmd_ms_helpers(self):
        ci = CarInterface()
        assert ci.cmd_to_ms(10.0) == pytest.approx(0.8)
        assert ci.ms_to_cmd(0.8) == pytest.approx(10.0)
        assert ci.ms_to_cmd(ci.cmd_to_ms(7.0)) == pytest.approx(7.0)

    def test_invalid_cmd_to_ms_raises(self):
        with pytest.raises(ValueError):
            CarInterface({"cmd_to_ms": 0.0})

    def test_nonfinite_input_does_not_poison_state(self):
        # NaN 입력이 slew 상태를 오염시키면 이후 출력이 전부 NaN이 된다 (버그 1 계열)
        ci = CarInterface()
        angle, speed = ci.to_motor(float("nan"), float("inf"))
        assert angle == pytest.approx(0.0)        # 직진 치환
        assert speed == pytest.approx(0.0)        # 정지 치환
        angle, speed = ci.to_motor(10.0, 10.0)
        assert angle == pytest.approx(
            10.0 * DEFAULT_CFG["steer_scale_right"])
        assert speed == pytest.approx(1.5)

    def test_cfg_defaults_match_spec(self):
        assert DEFAULT_CFG["steer_trim"] == 0.0
        assert DEFAULT_CFG["steer_scale_left"] == pytest.approx(0.625933146)
        assert DEFAULT_CFG["steer_scale_right"] == pytest.approx(0.585923661)
        assert DEFAULT_CFG["steer_limit_left"] == pytest.approx(-62.593314622)
        assert DEFAULT_CFG["steer_limit_right"] == pytest.approx(58.592366078)
        assert DEFAULT_CFG["speed_deadzone"] == 4.0
        assert DEFAULT_CFG["cmd_to_ms"] == 0.08
        assert DEFAULT_CFG["allow_reverse"] is False
