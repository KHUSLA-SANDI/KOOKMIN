"""track_drive.lib.path_gen 회피 오프셋 생성 테스트 (rclpy 불필요)."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from track_drive.lib.path_gen import PathGen  # noqa: E402

RAMP = 0.06     # 기본 ramp_step
MARGIN = 0.45   # 기본 clear_margin


def straight_ref(n=36):
    """직진 참조 경로: x 0.5~4.0 m (0.1 m 간격), y=0."""
    xs = np.linspace(0.5, 4.0, n)
    return xs, np.zeros_like(xs)


def converge(gen, xs, ys, obstacles, n=30):
    out = None
    for _ in range(n):
        out = gen.generate(xs, ys, obstacles)
    return out


# ============================================================
# 정면 장애물 → 여유 큰 쪽 회피
# ============================================================

def test_front_obstacle_avoid_to_larger_gap_side():
    g = PathGen()
    xs, ys = straight_ref()
    obs = [(2.0, 0.15, 0.2)]        # 정면에서 살짝 좌측 → 우(-y)측 여유가 큼

    out = converge(g, xs, ys, obs)
    assert out["offset"] < -0.2                       # 우측(-)으로 회피
    assert out["min_clear"] >= MARGIN - 1e-9          # 여유 확보
    np.testing.assert_allclose(out["ys"], ys + out["offset"])


def test_obstacle_dict_input_accepted():
    g = PathGen()
    xs, ys = straight_ref()
    out = converge(g, xs, ys, [{"x": 2.0, "y": 0.15, "r": 0.2}])
    assert out["offset"] < -0.2                       # 튜플 입력과 동일 동작


# ============================================================
# 장애물 소멸 → 0 복귀
# ============================================================

def test_return_to_zero_after_obstacle_gone():
    g = PathGen()
    xs, ys = straight_ref()
    out = converge(g, xs, ys, [(2.0, 0.15, 0.2)])
    assert abs(out["offset"]) > 0.3                   # 일단 회피 중

    prev = abs(out["offset"])
    for _ in range(15):
        out = g.generate(xs, ys, [])
        assert abs(out["offset"]) <= prev + 1e-12     # |d| 단조 감소
        prev = abs(out["offset"])
    assert out["offset"] == 0.0                        # 정확히 0 복귀
    assert out["min_clear"] == float("inf")


# ============================================================
# 램프 상한 — 경로 점프 금지
# ============================================================

def test_ramp_limits_offset_change_per_call():
    g = PathGen()
    xs, ys = straight_ref()
    obs = [(2.0, 0.0, 0.3)]         # 정면 정중앙 — 즉시 회피 필요

    out = g.generate(xs, ys, obs)
    assert abs(out["offset"]) <= RAMP + 1e-12          # 첫 호출도 상한 준수
    assert abs(out["offset"]) > 0.0

    prev = out["offset"]
    for _ in range(20):
        out = g.generate(xs, ys, obs)
        assert abs(out["offset"] - prev) <= RAMP + 1e-12
        prev = out["offset"]


# ============================================================
# 히스테리시스 — 비용이 근소하게 엎치락뒤치락해도 선택 유지
# ============================================================

def test_hysteresis_prevents_flip_flop():
    # 후보를 ±0.6 둘로 제한, 연속성 비용 제거 → 히스테리시스만으로 유지되는지 검증
    g = PathGen({"offsets": [-0.6, 0.6], "w_cont": 0.0})
    xs, ys = straight_ref()

    sel_hist = []
    argmin_hist = []
    for i in range(20):
        y_o = 0.02 if i % 2 == 0 else -0.02   # 장애물이 좌우로 살짝 흔들림
        out = g.generate(xs, ys, [(2.0, y_o, 0.2)])
        sel_hist.append(out["debug"]["sel_offset"])
        argmin_hist.append(int(np.argmin(out["debug"]["costs"])))

    # 매 프레임 최저비용 후보는 실제로 엎치락뒤치락하지만...
    assert len(set(argmin_hist)) == 2
    # ...선택은 첫 프레임 이후 한 번도 안 바뀐다
    assert len(set(sel_hist)) == 1
    assert out["offset"] < 0.0                 # 처음 고른 쪽(-0.6) 유지


# ============================================================
# 좌측 상한 — 조향 좌측 여유 부족(비대칭) 반영
# ============================================================

def test_left_offset_capped_at_max():
    g = PathGen()
    xs, ys = straight_ref()
    # 우측~중앙을 벽처럼 막아 좌회피 강제 (최적은 +1.2 지만 후보에서 제외돼야 함)
    obs = [(2.0, -0.9, 0.3), (2.0, -0.3, 0.3), (2.0, 0.3, 0.3)]

    out = converge(g, xs, ys, obs, n=40)
    cand = out["debug"]["offsets"]
    assert cand.max() <= 0.9 + 1e-9            # +1.2 후보 자체가 잘림
    assert cand.min() <= -1.2 + 1e-9           # 우측은 -1.2 까지 그대로 (비대칭)
    assert abs(out["offset"] - 0.9) < 1e-9     # 허용 최대 좌측으로 수렴
    # 상한 탓에 여유는 부족하지만 관통은 아님
    assert 0.0 < out["min_clear"] < MARGIN


# ============================================================
# 빈 참조 경로 — 상태 동결 없이 0 복귀 지속
# ============================================================

def test_empty_ref_decays_without_crash():
    g = PathGen()
    xs, ys = straight_ref()
    converge(g, xs, ys, [(2.0, 0.15, 0.2)])    # 오프셋 생성
    out = g.generate([], [], [])
    assert out["xs"].size == 0
    prev = abs(out["offset"])
    for _ in range(15):
        out = g.generate([], [], [])
        assert abs(out["offset"]) <= prev + 1e-12
        prev = abs(out["offset"])
    assert out["offset"] == 0.0
