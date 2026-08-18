"""라이다 gap 클러스터링 + 프레임간 트래커 — 순수 numpy.

역할: /scan ranges 1회분을 장애물 클러스터 목록으로 변환(cluster_scan)하고,
  ClusterTracker 로 k-of-n 연속 관측 확인 + 프레임간 위치변화(정적/동적 판별
  재료)를 제공.
입력: cluster_scan(ranges, angle_min, angle_inc, ...) — LaserScan 필드 그대로.
  실차 /scan: 500빔, angle_min=-π, incr≈0.72°, BEST_EFFORT.
출력: [{"x","y","r","n"}] (라이다 좌표계 x전방/y좌 m, r=반경, n=점 개수).
  트래커 update()는 +id/age/vel_xy/confirmed/seen 을 붙여 반환.
예선 대비 변경점: integration_node.cluster_scan_obstacles(gap 분할) 골격 계승.
  min_pts 1→2(단일점 노이즈 제거), 스캔 시작/끝(-π/+π 경계) 랩 병합,
  거리 비례 gap 허용(먼 물체일수록 빔 간 간격이 벌어짐), ClusterTracker 신규.
주의: 외부팀(콘 통로 모듈)도 이 파일 하나만 import 한다 — 패키지 내 다른
  모듈/rclpy 에 의존하지 말 것.
"""

import numpy as np

# 기본값 출처 메모 (노드에서 declare_parameter 로 노출됨):
#   gap_m=0.35    # [SIM] 예선 CLUSTER_GAP 계승
#   padding=0.05  # [SIM] 예선 CLUSTER_PADDING 계승
#   gap_per_m, min_pts, r_min/r_max 는 실차 신규 값


def cluster_scan(ranges, angle_min, angle_inc,
                 r_min=0.05, r_max=8.0, gap_m=0.35, min_pts=2,
                 gap_per_m=0.05, padding=0.05):
    """스캔 1회분을 gap 기준으로 클러스터링.

    인접 유효 점 i-1,i 사이 거리가 gap_m + gap_per_m*min(r[i-1],r[i]) 를
    넘으면 분할 (거리 비례 gap 허용 — 먼 물체는 빔 간 호 길이가 커짐).
    분할 후 첫/끝 클러스터의 끝점이 같은 허용치 안이면 랩 병합
    (-π/+π 경계에 걸친 물체가 둘로 쪼개지는 것 방지).

    반환: [{"x","y","r","n"}] — 중심(x전방/y좌 m), r=중심~최원점+padding, n=점수.
    """
    ranges = np.asarray(ranges, dtype=np.float64).ravel()
    m = ranges.size
    if m == 0:
        return []

    angles = float(angle_min) + np.arange(m) * float(angle_inc)
    valid = np.isfinite(ranges) & (ranges > r_min) & (ranges < r_max)
    if not valid.any():
        return []

    r = ranges[valid]
    a = angles[valid]
    pts = np.column_stack([r * np.cos(a), r * np.sin(a)])  # x전방 / y좌

    if pts.shape[0] == 1:
        idx_groups = [np.array([0])]
    else:
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        allow = gap_m + gap_per_m * np.minimum(r[:-1], r[1:])
        splits = np.where(d > allow)[0] + 1
        idx_groups = np.split(np.arange(pts.shape[0]), splits)

        # 스캔 시작/끝 랩 병합 — 마지막 점과 첫 점이 물리적으로 이어져 있으면
        # 끝 그룹을 첫 그룹 앞에 붙인다 (2그룹 이상일 때만 의미 있음)
        if len(idx_groups) >= 2:
            d_wrap = float(np.linalg.norm(pts[-1] - pts[0]))
            allow_wrap = gap_m + gap_per_m * min(r[-1], r[0])
            if d_wrap <= allow_wrap:
                idx_groups[0] = np.concatenate([idx_groups[-1], idx_groups[0]])
                idx_groups.pop()

    out = []
    for g in idx_groups:
        if g.size < min_pts:  # 단일점 노이즈 제거 (예선 min_pts=1 → 2)
            continue
        gp = pts[g]
        cx, cy = gp.mean(axis=0)
        rad = float(np.linalg.norm(gp - (cx, cy), axis=1).max()) + padding
        out.append({"x": float(cx), "y": float(cy), "r": rad, "n": int(g.size)})
    return out


def _cluster_xyrn(c):
    """dict{x,y,r[,n]} 또는 (x,y,r[,n]) 튜플 허용."""
    if isinstance(c, dict):
        return float(c["x"]), float(c["y"]), float(c.get("r", 0.0)), int(c.get("n", 1))
    x, y = float(c[0]), float(c[1])
    rr = float(c[2]) if len(c) > 2 else 0.0
    nn = int(c[3]) if len(c) > 3 else 1
    return x, y, rr, nn


class ClusterTracker:
    """프레임간 최근접 매칭 + k-of-n 연속 관측 확인 트래커.

    k/n: 최근 n프레임 중 k프레임 이상 관측되면 confirmed.
    match_dist: 프레임간 같은 물체로 볼 최대 이동 거리 (m).
    vel_xy: 프레임간 위치변화량 (dt 미지정 시 m/frame, dt 주면 m/s) —
      정적(라바콘)/동적(차량) 구분 재료. 필터링은 호출측 책임.
    """

    def __init__(self, k=2, n=3, match_dist=0.5):
        if not (1 <= k <= n):
            raise ValueError("1 <= k <= n 이어야 함")
        self.k = int(k)
        self.n = int(n)
        self.match_dist = float(match_dist)
        self._tracks = []
        self._next_id = 1

    def reset(self):
        self._tracks = []
        self._next_id = 1

    def update(self, clusters, dt=None):
        """새 프레임의 클러스터 목록으로 트랙 갱신.

        반환: 살아있는 모든 트랙의 dict 목록 —
          {"x","y","r","n","id","age","vel_xy","confirmed","seen"}
          seen=False 트랙은 이번 프레임 미관측(마지막 위치 유지, coasting).
          최근 n프레임 전부 미관측이면 트랙 제거.
        """
        step = float(dt) if (dt is not None and dt > 0.0) else 1.0
        cl = [_cluster_xyrn(c) for c in clusters]

        # ---- 최근접 그리디 매칭 (match_dist 이내만) ----
        t2c = {}
        if cl and self._tracks:
            txy = np.array([[t["x"], t["y"]] for t in self._tracks])
            cxy = np.array([[c[0], c[1]] for c in cl])
            dmat = np.linalg.norm(txy[:, None, :] - cxy[None, :, :], axis=2)
            used_c = set()
            flat = np.argsort(dmat, axis=None)
            for f in flat:
                ti, ci = divmod(int(f), dmat.shape[1])
                if dmat[ti, ci] > self.match_dist:
                    break  # 정렬돼 있으므로 이후는 전부 초과
                if ti in t2c or ci in used_c:
                    continue
                t2c[ti] = ci
                used_c.add(ci)
        else:
            used_c = set()

        # ---- 기존 트랙 갱신 ----
        for ti, tr in enumerate(self._tracks):
            tr["age"] += 1
            ci = t2c.get(ti)
            if ci is None:
                tr["hits"].append(0)
                tr["seen"] = False
            else:
                x, y, rr, nn = cl[ci]
                tr["vel"] = ((x - tr["x"]) / step, (y - tr["y"]) / step)
                tr["x"], tr["y"], tr["r"], tr["n"] = x, y, rr, nn
                tr["hits"].append(1)
                tr["seen"] = True
            del tr["hits"][:-self.n]  # 최근 n프레임 창 유지

        # ---- 미매칭 클러스터 → 새 트랙 ----
        for ci, (x, y, rr, nn) in enumerate(cl):
            if ci in used_c:
                continue
            self._tracks.append({
                "id": self._next_id, "x": x, "y": y, "r": rr, "n": nn,
                "age": 0, "hits": [1], "vel": (0.0, 0.0), "seen": True,
            })
            self._next_id += 1

        # ---- 전멸(최근 n프레임 모두 미관측) 트랙 제거 ----
        self._tracks = [t for t in self._tracks if sum(t["hits"]) > 0]

        return [{
            "x": t["x"], "y": t["y"], "r": t["r"], "n": t["n"],
            "id": t["id"], "age": t["age"],
            "vel_xy": t["vel"],
            "confirmed": sum(t["hits"]) >= self.k,
            "seen": t["seen"],
        } for t in self._tracks]
