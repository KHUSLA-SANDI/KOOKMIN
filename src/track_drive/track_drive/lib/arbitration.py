"""속도 중재 + owner/블록 전이 판정 — 순수 함수 모음.

역할: motion이 매 주기 호출하는 속도 중재(arbitrate_speed)와
  owner/블록 전이 감지(detect_transition), 블록 원샷 게이트(should_run_block)
입력: 명령단위 속도값들 (v_base, caps/floors 리스트, fixed, stop)
출력: 최종 속도 명령 float / 전이 여부 bool
예선 대비 변경점: 예선의 '무조건 대입'(버그 2)·'0.0 수신으로만 cap 해제'(버그 3)를
  스냅샷 기반 min/max 중재로 대체. cap이 floor를 이긴다 (예선 실효 의미론:
  slow cap 5.0 > 코너 floor 7.2). 블록은 id 기반 원샷 (버그 9 대응).
  rclpy import 금지.
"""

import math


def _valid_entries(values):
    """<=0 이거나 NaN/inf 인 cap/floor는 '없음'으로 무시."""
    if values is None:
        return []
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and f > 0.0:
            out.append(f)
    return out


def arbitrate_speed(v_base, caps=(), floors=(), fixed=0.0, stop=False):
    """최종 속도 명령 산출.

    규칙 (우선순위 순):
      1. stop이면 무조건 0.0
      2. v = max(v_base, *유효 floor)
      3. fixed > 0 이면 v = fixed  (floor 결과를 덮어씀)
      4. v = min(v, *유효 cap)     — cap이 floor/fixed를 이긴다
    <=0 또는 비정상(NaN/inf) cap/floor/fixed는 없음으로 취급. 빈 리스트 안전.
    """
    if stop:
        return 0.0

    try:
        v = float(v_base)
    except (TypeError, ValueError):
        v = 0.0
    if not math.isfinite(v):
        v = 0.0

    for f in _valid_entries(floors):
        if f > v:
            v = f

    try:
        fixed_f = float(fixed)
    except (TypeError, ValueError):
        fixed_f = 0.0
    if math.isfinite(fixed_f) and fixed_f > 0.0:
        v = fixed_f

    for c in _valid_entries(caps):
        if c < v:
            v = c

    return v


def _as_id(v):
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return 0


def detect_transition(prev_owner, owner, prev_block_id, block_id):
    """owner가 바뀌었거나 block_id가 바뀌면 True.

    motion은 True일 때 스무딩/하위 상태를 리셋한다 — early-return으로
    이전 상태가 동결된 채 이월되는 것(버그 6) 방지.
    """
    return (_as_id(prev_owner) != _as_id(owner)
            or _as_id(prev_block_id) != _as_id(block_id))


def should_run_block(block_id, done_ids):
    """블록 원샷 게이트 — 같은 id 재수신은 실행하지 않는다 (버그 9 대응).

    block_id 0(없음)·음수는 항상 False. done_ids는 수정하지 않는다 —
    실행을 시작한 쪽이 완료 시점에 done_ids.add(id)로 마킹할 것.
    """
    bid = _as_id(block_id)
    if bid <= 0:
        return False
    return bid not in done_ids
