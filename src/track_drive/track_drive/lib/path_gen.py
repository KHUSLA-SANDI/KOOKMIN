"""횡 오프셋 회피 경로 생성기 — 순수 numpy.

역할: 참조 경로(중앙선 등)를 y(좌+) 방향으로 평행이동한 오프셋 후보들의
  비용(장애물 여유/차선 복귀/연속성)을 평가해 최적 오프셋을 고르고,
  램프 상한 + 히스테리시스로 경로 점프 없이 적용한다.
입력: generate(ref_xs, ref_ys, obstacles) — 라이다 좌표계 x전방/y좌 (m),
  obstacles = [(x, y, r)] 또는 cluster_scan 이 주는 dict{x,y,r} 목록.
출력: dict(xs, ys, offset, min_clear, debug).
예선 대비 변경점: 신규 파일 — 예선엔 없던 격자 오프셋 비용 최적화.
  오프셋은 내부 상태로만 보존하고 호출당 ramp_step 이상 움직이지 않는다
  (예선 버그 1 '스무딩 상태 직접 오염' 유형 재발 방지). 장애물이 없으면
  d→0 으로 수렴. 좌측(+y)은 조향 스톱 여유 부족(-32)으로 후보 자체를
  left_offset_max 로 잘라 비대칭 운용.
"""

import numpy as np

# cfg 기본값 — 노드에서 declare_parameter 로 노출됨. 전부 실차 신규 설계값.
_DEFAULT_CFG = {
    "offsets": None,          # None이면 -1.2 ~ +1.2, 0.3 간격 격자
    "w_obs": 3.0,             # 여유(clear_margin) 침해량 벌점 가중
    "w_lane": 0.4,            # |d| 벌점 — 차선 중앙 복귀 유도
    "w_cont": 1.5,            # |d - 이전 적용 오프셋| 벌점 — 연속성
    "w_collision": 100.0,     # 경로가 장애물 원판을 관통(clearance<=0)하면 추가 벌점
    "clear_margin": 0.45,     # 차폭/2 + 여유 (m)
    "ramp_step": 0.06,        # 호출당 오프셋 변화 상한 (m) — 경로 점프 금지
    "left_offset_max": 0.9,   # 좌(+y) 오프셋 상한 — 좌 조향 여유 부족 반영(비대칭)
    "hysteresis": 0.15,       # 이전 선택 대비 비용 개선이 이 미만이면 유지
}


def _obs_array(obstacles):
    """[(x,y,r)] 또는 dict{x,y,r} 목록 → (N,3) ndarray."""
    rows = []
    for o in (obstacles or []):
        if isinstance(o, dict):
            rows.append((float(o["x"]), float(o["y"]), float(o.get("r", 0.0))))
        else:
            rows.append((float(o[0]), float(o[1]), float(o[2])))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 3)


class PathGen:
    """오프셋 격자 비용 최적화 기반 회피 경로 생성기 (상태: 적용/선택 오프셋)."""

    def __init__(self, cfg=None):
        c = dict(_DEFAULT_CFG)
        if cfg:
            c.update(cfg)
        if c["offsets"] is None:
            c["offsets"] = np.arange(-1.2, 1.2 + 1e-9, 0.3)
        offs = np.sort(np.asarray(c["offsets"], dtype=np.float64).ravel())
        # 좌(+y) 후보 컷 — 좌 조향 스톱 여유 부족으로 큰 좌측 회피는 못 따라감
        offs = offs[offs <= c["left_offset_max"] + 1e-9]
        if offs.size == 0:
            raise ValueError("left_offset_max 컷 이후 오프셋 후보가 없음")
        self.cfg = c
        self._offsets = offs
        self._applied = 0.0   # 실제 적용 중인 오프셋 (램프 결과)
        self._sel = 0.0       # 마지막 선택 오프셋 (히스테리시스 기준)

    def reset(self):
        self._applied = 0.0
        self._sel = 0.0

    # ------------------------------------------------------------
    def _ramp_toward(self, target):
        """이전 적용 오프셋에서 ramp_step 이내로만 이동 — 경로 점프 금지."""
        step = self.cfg["ramp_step"]
        return float(self._applied + np.clip(target - self._applied, -step, step))

    def _candidate_costs(self, xs, ys, obs):
        """후보 오프셋별 비용. obs: (N,3) [x,y,r]."""
        c = self.cfg
        dx = xs[:, None] - obs[None, :, 0]          # (pts, obs)
        dy0 = ys[:, None] - obs[None, :, 1]
        costs = np.empty(self._offsets.size)
        for i, d in enumerate(self._offsets):
            dy = dy0 + d
            # 장애물별 경로 최근접 clearance (원판 표면 기준)
            clear = np.sqrt(dx * dx + dy * dy).min(axis=0) - obs[:, 2]
            viol = np.clip(c["clear_margin"] - clear, 0.0, None)
            cost = c["w_obs"] * viol.sum()
            cost += c["w_collision"] * float((clear <= 0.0).sum())  # 관통 = 큰 벌점
            cost += c["w_lane"] * abs(d)
            cost += c["w_cont"] * abs(d - self._applied)
            costs[i] = cost
        return costs

    # ------------------------------------------------------------
    def generate(self, ref_xs, ref_ys, obstacles):
        """회피 오프셋 적용 경로 생성.

        반환 dict:
          xs, ys     — 적용 오프셋으로 평행이동된 경로 (ys = ref_ys + offset)
          offset     — 실제 적용 오프셋 (m, 좌+)
          min_clear  — 적용 경로와 장애물 표면 간 최소 여유 (장애물 없으면 inf)
          debug      — offsets/costs/sel_offset/n_obs (사람 확인용)
        """
        xs = np.asarray(ref_xs, dtype=np.float64).ravel()
        ys = np.asarray(ref_ys, dtype=np.float64).ravel()
        obs = _obs_array(obstacles)
        c = self.cfg

        costs = None
        if xs.size == 0:
            # 참조 경로 없음 — 새 선택은 못 하지만 램프/0 복귀는 계속 진행
            # (early-return 상태 동결(예선 버그 6) 금지)
            self._sel = 0.0
        elif obs.shape[0] == 0:
            self._sel = 0.0   # 장애물 없으면 중앙(d=0) 복귀
        else:
            costs = self._candidate_costs(xs, ys, obs)
            i_best = int(np.argmin(costs))
            i_prev = int(np.argmin(np.abs(self._offsets - self._sel)))
            # 히스테리시스: 이전 선택 대비 비용 개선이 문턱 미만이면 유지
            if costs[i_best] < costs[i_prev] - c["hysteresis"]:
                i_sel = i_best
            else:
                i_sel = i_prev
            self._sel = float(self._offsets[i_sel])

        self._applied = self._ramp_toward(self._sel)
        out_ys = ys + self._applied

        min_clear = float("inf")
        if xs.size and obs.shape[0]:
            dx = xs[:, None] - obs[None, :, 0]
            dy = out_ys[:, None] - obs[None, :, 1]
            min_clear = float(
                (np.sqrt(dx * dx + dy * dy).min(axis=0) - obs[:, 2]).min())

        return {
            "xs": xs.copy(),
            "ys": out_ys,
            "offset": self._applied,
            "min_clear": min_clear,
            "debug": {
                "offsets": self._offsets.copy(),
                "costs": None if costs is None else costs.copy(),
                "sel_offset": self._sel,
                "n_obs": int(obs.shape[0]),
            },
        }
