"""track_drive.lib.arbitration 속도 중재/전이 테스트 (rclpy 불필요)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from track_drive.lib.arbitration import (  # noqa: E402
    arbitrate_speed,
    detect_transition,
    should_run_block,
)


# ============================================================
# arbitrate_speed — stop 최우선
# ============================================================

def test_stop_beats_everything():
    assert arbitrate_speed(10.0, caps=[5.0], floors=[20.0],
                           fixed=8.0, stop=True) == 0.0
    assert arbitrate_speed(0.0, stop=True) == 0.0


# ============================================================
# arbitrate_speed — cap > floor (예선 실효 의미론)
# ============================================================

def test_cap_beats_floor():
    # 코너 floor 7.2를 slow cap 5.0이 이긴다
    assert arbitrate_speed(4.0, caps=[5.0], floors=[7.2]) == 5.0


def test_floor_raises_base_without_cap():
    assert arbitrate_speed(4.0, floors=[7.2]) == 7.2
    assert arbitrate_speed(9.0, floors=[7.2]) == 9.0  # 이미 높으면 그대로


def test_cap_limits_base():
    assert arbitrate_speed(10.0, caps=[6.0]) == 6.0
    assert arbitrate_speed(4.0, caps=[6.0]) == 4.0  # cap 밑이면 그대로


def test_multiple_caps_and_floors():
    # floor는 max, cap은 min이 실효값
    assert arbitrate_speed(4.0, floors=[5.0, 7.0, 6.0]) == 7.0
    assert arbitrate_speed(10.0, caps=[8.0, 6.0, 9.0]) == 6.0
    assert arbitrate_speed(4.0, caps=[8.0, 6.0], floors=[5.0, 7.0]) == 6.0


# ============================================================
# arbitrate_speed — fixed는 cap에 종속
# ============================================================

def test_fixed_overrides_base_and_floor():
    assert arbitrate_speed(2.0, floors=[6.0], fixed=3.0) == 3.0
    assert arbitrate_speed(10.0, fixed=3.0) == 3.0


def test_fixed_still_capped():
    assert arbitrate_speed(4.0, caps=[5.0], fixed=12.0) == 5.0
    assert arbitrate_speed(4.0, caps=[5.0], fixed=3.0) == 3.0  # cap 밑 fixed는 그대로


def test_fixed_nonpositive_ignored():
    assert arbitrate_speed(4.0, fixed=0.0) == 4.0
    assert arbitrate_speed(4.0, fixed=-1.0) == 4.0


# ============================================================
# arbitrate_speed — 무시 규칙 / 빈 입력 / 비정상 값
# ============================================================

def test_nonpositive_caps_floors_ignored():
    assert arbitrate_speed(4.0, caps=[0.0, -1.0], floors=[0.0, -2.0]) == 4.0


def test_empty_inputs_return_base():
    assert arbitrate_speed(4.0) == 4.0
    assert arbitrate_speed(4.0, caps=[], floors=[]) == 4.0
    assert arbitrate_speed(4.0, caps=None, floors=None) == 4.0


def test_nan_entries_ignored():
    nan = float("nan")
    inf = float("inf")
    assert arbitrate_speed(4.0, caps=[nan], floors=[nan]) == 4.0
    assert arbitrate_speed(4.0, caps=[inf], floors=[inf]) == 4.0  # inf cap/floor 무시
    assert arbitrate_speed(nan) == 0.0  # base 자체가 비정상이면 0
    assert arbitrate_speed(4.0, fixed=nan) == 4.0


# ============================================================
# detect_transition
# ============================================================

def test_transition_on_owner_change():
    assert detect_transition(0, 1, 0, 0) is True
    assert detect_transition(1, 3, 5, 5) is True


def test_transition_on_block_id_change():
    assert detect_transition(1, 1, 2, 3) is True
    assert detect_transition(0, 0, 0, 1) is True


def test_no_transition_when_same():
    assert detect_transition(0, 0, 0, 0) is False
    assert detect_transition(1, 1, 7, 7) is False


def test_transition_float_inputs_from_wire():
    # Float32MultiArray로 온 값 (1.0 vs 1)도 동일 취급
    assert detect_transition(1.0, 1, 2.0, 2) is False
    assert detect_transition(0.0, 1.0, 0.0, 0.0) is True


# ============================================================
# should_run_block — 원샷
# ============================================================

def test_block_zero_never_runs():
    assert should_run_block(0, set()) is False
    assert should_run_block(-1, set()) is False


def test_new_block_runs_once():
    done = set()
    assert should_run_block(1, done) is True
    done.add(1)  # 실행 측이 완료 마킹
    assert should_run_block(1, done) is False  # 같은 id 재수신 무시
    assert should_run_block(2, done) is True   # 다음 블록은 실행


def test_should_run_block_does_not_mutate():
    done = {1, 2}
    should_run_block(3, done)
    assert done == {1, 2}


def test_should_run_block_float_id_from_wire():
    assert should_run_block(3.0, {1, 2}) is True
    assert should_run_block(2.0, {1, 2}) is False
