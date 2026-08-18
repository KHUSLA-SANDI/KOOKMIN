"""track_drive.lib.cluster 합성 스캔 클러스터링/트래커 테스트 (rclpy 불필요)."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from track_drive.lib.cluster import ClusterTracker, cluster_scan  # noqa: E402

# 실차 /scan 스펙: 500빔, angle_min=-π, incr=2π/500 (≈0.72°)
N = 500
AMIN = -np.pi
INC = 2.0 * np.pi / N


def make_ranges(beams):
    """{빔 인덱스: 거리} → ranges 배열 (나머지는 inf=무효)."""
    r = np.full(N, np.inf)
    for i, v in beams.items():
        r[i] = v
    return r


def run(ranges, **kw):
    return cluster_scan(ranges, AMIN, INC, **kw)


# ============================================================
# cluster_scan — 기본 분할 / min_pts / 거리 필터
# ============================================================

def test_two_objects_and_min_pts():
    beams = {}
    for i in range(248, 253):   # 물체 A: 정면(각도 0 부근) r=2.0, 5점
        beams[i] = 2.0
    for i in range(370, 374):   # 물체 B: 좌측 ~86° r=1.5, 4점
        beams[i] = 1.5
    beams[100] = 3.0            # 고립 단일점 → min_pts=2 로 제거돼야 함
    out = run(make_ranges(beams))

    assert len(out) == 2
    a = min(out, key=lambda c: abs(c["y"]))   # 정면 물체
    b = max(out, key=lambda c: abs(c["y"]))
    assert abs(a["x"] - 2.0) < 0.05 and abs(a["y"]) < 0.05
    assert a["n"] == 5
    assert abs(b["y"] - 1.5) < 0.05
    assert b["n"] == 4
    # 반경 = 중심~최원점 + padding — 5점 스팬(±0.05m)보다 크고 과대하지 않음
    assert 0.05 < a["r"] < 0.2


def test_range_min_max_filter():
    beams = {10: 0.03, 11: 0.03,      # r_min=0.05 미만 → 무효 (차체 반사)
             200: 9.5, 201: 9.5,      # r_max=8.0 초과 → 무효
             300: 1.0, 301: 1.0}      # 유효
    out = run(make_ranges(beams))
    assert len(out) == 1
    assert out[0]["n"] == 2


def test_empty_and_all_invalid():
    assert run(np.full(N, np.inf)) == []
    assert cluster_scan(np.array([]), AMIN, INC) == []


# ============================================================
# cluster_scan — 스캔 시작/끝 랩 병합
# ============================================================

def test_wrap_merge_across_pi_boundary():
    beams = {}
    for i in (497, 498, 499, 0, 1, 2):   # -π/+π 경계에 걸친 후방 물체 r=1.0
        beams[i] = 1.0
    for i in range(248, 253):            # 중간(정면)에 다른 물체 → 시퀀스 분리 유발
        beams[i] = 2.0
    out = run(make_ranges(beams))

    # 랩 병합이 없으면 후방 물체가 둘로 쪼개져 3개가 됨
    assert len(out) == 2
    rear = min(out, key=lambda c: c["x"])
    assert abs(rear["x"] + 1.0) < 0.02      # x ≈ -1 (후방)
    assert abs(rear["y"]) < 0.05
    assert rear["n"] == 6                    # 3+3 병합 확인


# ============================================================
# cluster_scan — 거리 비례 gap 허용
# ============================================================

def test_distance_proportional_gap_merges_far_object():
    # r=6.0 물체가 6빔 무효 구간을 사이에 두고 갈라져 보임.
    # 끊긴 구간 현길이 ≈ 2*6*sin(7*INC/2) ≈ 0.53 m:
    #   기본 gap 0.35 만으론 분할, 거리 비례 허용(0.35+0.05*6=0.65)으론 병합.
    beams = {}
    for i in (238, 239, 240, 247, 248, 249):
        beams[i] = 6.0
    r = make_ranges(beams)

    merged = run(r)                    # 기본 gap_per_m=0.05
    assert len(merged) == 1
    assert merged[0]["n"] == 6

    split = run(r, gap_per_m=0.0)      # 비례항 제거하면 둘로 갈라짐
    assert len(split) == 2


def test_proportional_gap_does_not_overmerge_near():
    # r=1.0 에서 40빔(현 ≈0.50 m) 떨어진 두 쌍 — 허용치 0.35+0.05*1=0.40 초과 → 분할 유지
    beams = {100: 1.0, 101: 1.0, 141: 1.0, 142: 1.0}
    out = run(make_ranges(beams))
    assert len(out) == 2


# ============================================================
# ClusterTracker — k-of-n 확인
# ============================================================

def c(x, y, r=0.1, n=5):
    return {"x": x, "y": y, "r": r, "n": n}


def test_tracker_k_of_n_confirm():
    tr = ClusterTracker(k=2, n=3, match_dist=0.5)

    out1 = tr.update([c(2.0, 0.0)])
    assert len(out1) == 1
    assert out1[0]["confirmed"] is False       # 1프레임 관측 < k=2
    assert out1[0]["age"] == 0

    out2 = tr.update([c(2.0, 0.05)])
    assert len(out2) == 1
    assert out2[0]["confirmed"] is True        # 2/3 관측 → 확인
    assert out2[0]["id"] == out1[0]["id"]      # 같은 트랙 유지
    assert out2[0]["age"] == 1


def test_tracker_coast_then_die():
    tr = ClusterTracker(k=2, n=3, match_dist=0.5)
    tr.update([c(2.0, 0.0)])
    tr.update([c(2.0, 0.0)])

    out3 = tr.update([])                       # 미관측 1
    assert len(out3) == 1
    assert out3[0]["seen"] is False            # coasting (마지막 위치 유지)
    assert abs(out3[0]["x"] - 2.0) < 1e-9

    out4 = tr.update([])                       # 미관측 2 — 창=[1,0,0] 아직 생존
    assert len(out4) == 1
    out5 = tr.update([])                       # 미관측 3 — 창=[0,0,0] 제거
    assert out5 == []


def test_tracker_velocity_static_vs_moving():
    tr = ClusterTracker(k=2, n=3, match_dist=0.5)
    tr.update([c(2.0, 0.0), c(1.0, 1.0)])
    out = tr.update([c(2.1, 0.0), c(1.0, 1.0)])   # 하나만 +0.1 전진

    moving = next(t for t in out if t["x"] > 1.5)
    static = next(t for t in out if t["x"] < 1.5)
    assert abs(moving["vel_xy"][0] - 0.1) < 1e-9  # dt 미지정 → m/frame
    assert abs(moving["vel_xy"][1]) < 1e-9
    assert abs(static["vel_xy"][0]) < 1e-9        # 정적 재료: 변위 ~0

    # dt 지정 시 m/s 환산
    out2 = tr.update([c(2.2, 0.0), c(1.0, 1.0)], dt=0.5)
    moving2 = next(t for t in out2 if t["x"] > 1.5)
    assert abs(moving2["vel_xy"][0] - 0.2) < 1e-9


def test_tracker_far_jump_makes_new_track():
    tr = ClusterTracker(k=2, n=3, match_dist=0.5)
    out1 = tr.update([c(2.0, 0.0)])
    out2 = tr.update([c(4.0, 0.0)])            # match_dist=0.5 초과 → 새 트랙

    assert len(out2) == 2
    old = next(t for t in out2 if t["id"] == out1[0]["id"])
    new = next(t for t in out2 if t["id"] != out1[0]["id"])
    assert old["seen"] is False and abs(old["x"] - 2.0) < 1e-9
    assert new["seen"] is True and abs(new["x"] - 4.0) < 1e-9
    assert new["age"] == 0 and new["confirmed"] is False
