#!/usr/bin/env python3
"""
[친구(sm) 원본 — 수정 금지] 차선 주행 plan() 함수 + 피팅 헬퍼.

path_planner_node.py의 LANE 모드에서 plan()을 import해서 호출.
이 파일은 친구 C:\\motion_node\\motion_node\\path_planner_node.py 와 동일.

핵심: plan() → _fit_yellow_sliding() → 노란선(cls_id=8) 2차 피팅 → center_path.
lane_detect_node가 OpenCV HSV로 노란/흰 분리해서 z=8/6 넣어줌.

경로계획 노드 — YOLO 클래스(MID/LANE/CHILD_LANE)를 demux해 차선 선택 주행.

구독:
  /fused/lane        (PoseArray)  — 차선 점, position.z = YOLO cls_id
                                     (MID 8 = 노랑 중앙선, LANE 6 = 흰색 외곽선,
                                      CHILD_LANE 2 = 학교영역 라인 → white로 취급)
  /fused/obstacles   (PoseArray)  — 라이다 클러스터

발행:
  /target      (PointStamped)
  /center_path (PoseArray)
  /lane_left   (PoseArray) — 선택된 좌측 경계
  /lane_right  (PoseArray) — 선택된 우측 경계
  /lane_fits   (PoseArray) — 검출된 모든 fit (yellow + white_l + white_r) flat

PREFERRED_LANE:
  "left"  → 좌측 차선 (white_left ↔ yellow_center) 사이로 주행
  "right" → 우측 차선 (yellow_center ↔ white_right) 사이로 주행
"""

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray, PointStamped

# ======================== 파라미터 ========================

# YOLO 클래스 → 차선 색상 매핑. CHILD_LANE은 학교영역 노면 표시여서 외곽 흰선
# 역할로 묶는다. integration_node가 position.z에 cls_id를 채워 보낸다.
YELLOW_CLS_IDS = {8}            # MID = 노랑 중앙선
YELLOW_DASH_CLS_IDS = {16}      # 어린이 보호구역용 노란 점선 조각
WHITE_CLS_IDS = {6, 2}          # LANE = 흰 외곽선, CHILD_LANE = 학교영역 마킹

PLAN_HZ = 20

# 어느 쪽 차선으로 갈지
PREFERRED_LANE = "right"   # 가까운 차선 동률일 때 fallback

# 주행 목표
LANE_WIDTH = 3.00
LANE_HALF_WIDTH = 1.50          # 노랑 중앙선에서 차선 중심까지 거리 (±이값 시프트)
# ★ TARGET_X 3.0 은 시뮬 값이었다. 실차 노란테이프는 BEV 점이 0.28~1.58m 에만
#   잡혀서, 3.0m 목표는 데이터 없는 곳을 2배 외삽하는 꼴이라 목표점이 심하게
#   흔들렸다 (실측: 3.0m → 0.22m 요동 / 1.0m → 0.076m). 데이터 안쪽으로 당겼다.
TARGET_X = 1.0
TARGET_Y_LIMIT = 4.0

# 피팅 ROI — 실차에서 실제로 점이 잡히는 구간에 맞췄다 (2026-08-12 실측).
# 옛 값 0.5~12.0 은 시뮬 기준이라 12m 까지 봤는데, 실차 점은 1.6m 안쪽이라
# 대부분이 빈 구간이었고 0.5m 로 자르는 바람에 가장 가까운 점들까지 버렸다.
FIT_X_MIN = 0.25
FIT_X_MAX = 2.50
# ★ 아래 셋은 예선 시뮬 값이 실차 노란테이프엔 2배쯤 빡세서 내렸다 (2026-08-12 실측).
#   실차 BEV 점은 중앙 16~17개 / x 폭 0.98m 인데, 하위 25% 가 각각 12~13개 / 0.76m 라
#   옛 문턱(12점, 0.75m)에 그대로 걸렸다. 240장 중 24% 가 점 개수에서 죽었다.
#   연속 400장으로 잰 결과 (성공률 / 목표점 떨림 중앙 / 경로 최장 끊김):
#       12점 0.75m 곡률1.30 → 66% / 0.025m / 3.2초   ← 3.2초 공백이 치명적
#        8점 0.60m          → 74% / 0.029m / 1.3초
#        6점 0.40m 곡률2.50 → 80% / 0.027m / 1.3초   ← 채택
#        4점 0.30m          → 82% / 0.028m / 1.4초   (더 내려도 끊김이 안 줄어 이득 없음)
#   떨림이 거의 안 늘면서 끊김만 3.2초→1.3초로 줄어 공짜에 가깝다.
#   ※ 근본 해결은 문턱이 아니라 x 폭이다. 점이 0.3~1.3m 밖에 안 뻗는 게 진짜 원인이라
#     라벨을 더 멀리 칠해 재학습하면 이 값들은 다시 올려도 된다.
FIT_MIN_POINTS = 6
FIT_MIN_X_SPAN = 0.40
FIT_BAND_WIDTH = 0.65
FIT_MAX_RMSE = 0.32
FIT_CURVE_MAX = 2.50
FIT_SLOPE_MAX = 2.80
FIT_BIN_WIDTH = 0.45
FIT_MIN_BINS = 5
FIT_REFINE_ITERS = 2
FIT_INLIER_BAND = 0.28
FIT_TARGET_MAX_JUMP = 0.65
FIT_Y_ABS_MAX = 4.5
FIT_EXPAND_BAND = 0.55
FIT_ANCHOR_WINDOWS = ((0.5, 3.5), (3.0, 7.5), (6.5, 12.0))
YELLOW_MAX_MISS = 6
YELLOW_MAX_JUMP_M = 0.55
YELLOW_FIT_SMOOTH_ALPHA = 0.35
YELLOW_DIRECTION_OUTLIER_BAND_M = 0.70
YELLOW_DIRECTION_FILTER_MIN_POINTS = FIT_MIN_POINTS
YELLOW_DIRECTION_SELECT_BIN_M = 0.45
CHILD_DASH_MIN_POINTS = 8
CHILD_DASH_MIN_X_SPAN = 1.15
CHILD_DASH_BAND_M = 0.38
CHILD_DASH_BIN_M = 0.35
CHILD_DASH_MIN_BINS = 4
CHILD_DASH_MIN_GAP_M = 0.65
CHILD_DASH_MAX_OCCUPANCY = 0.82
CHILD_DASH_MAX_RMSE = 0.40
CHILD_DASH_MAX_MISS = 8
CHILD_DASH_MAX_JUMP_M = 0.85
CHILD_DASH_SMOOTH_ALPHA = 0.35

MIN_LANE_WIDTH = 2.40
MAX_LANE_WIDTH = 3.60
WHITE_FALLBACK_X_MAX = 10.0
WHITE_FALLBACK_Y_ABS_MAX = 5.5
LANE_WIDTH_ACCEPT_MIN = 2.40
LANE_WIDTH_ACCEPT_MAX = 3.60
LANE_SIDE_SWITCH_MARGIN = 1.0

# 흰색 좌/우 dedup
WHITE_DEDUP_DIST = 0.30
WHITE_YELLOW_OFFSET_TOL = 0.35

# 샘플 — 실제로 본 구간만 발행한다. 옛 값 1.0~11.0 은 데이터가 1.6m 까지밖에
# 없는데 11m 까지 2차식을 외삽해 뿌리는 것이라, motion 이 그 가짜 먼 점을 보고
# 조향했다. motion 쪽 lookahead/focus 도 이 범위에 맞춰 줄여야 한다
# (config/lane_only.yaml 참고).
SAMPLE_X_MIN = 0.3
SAMPLE_X_MAX = 2.5
SAMPLE_COUNT = 13

# 타겟 스무딩
TARGET_SMOOTH_ALPHA = 0.27
TARGET_MAX_STEP_M = 0.175
TARGET_DEADBAND_M = 0.020
LANE_LOST_HOLD_FRAMES = 8
LANE_PLAN_ALPHA = 0.34
LANE_MEMORY_FRAMES = 18

# 장애물 회피
AVOID_X_MIN = 0.3
AVOID_X_MAX = 3.5
AVOID_LATERAL = 0.7
AVOID_SHIFT = 0.8

# 모듈 상태
_prev_target_y = 0.0
_lost_lane_frames = 0
_prev_left_fit = None
_prev_right_fit = None
_prev_yellow_fit = None
_prev_white_left_fit = None
_prev_white_right_fit = None
_FIT_SMOOTH_ALPHA = 0.30
_prev_lane_plan = None
_lane_memory_frames = 0
_prev_lane_side = None
_yellow_miss = 0
_prev_child_yellow_dash_fit = None
_child_yellow_dash_miss = 0

# YOLO seg 출력은 노이즈가 적어서 RANSAC 없이 단순 polyfit으로 충분.
# 실제 fit에 들어간 점들을 캐싱해 /fit/yellow_inliers로 노출 (디버그용).
_yellow_inlier_xs = np.array([], dtype=np.float64)
_yellow_inlier_ys = np.array([], dtype=np.float64)


# ======================== 단일 2차식 피팅 ========================

def _score_poly2(xs, ys, min_points=FIT_MIN_POINTS):
    if xs.size < min_points:
        return None
    x_span = float(np.max(xs) - np.min(xs))
    if x_span < FIT_MIN_X_SPAN:
        return None
    try:
        coef = np.polyfit(xs, ys, 2)
    except Exception:
        return None
    a, b, _ = coef
    if abs(a) > FIT_CURVE_MAX or abs(b) > FIT_SLOPE_MAX:
        return None
    pred = np.polyval(coef, xs)
    err = ys - pred
    rmse = float(np.sqrt(np.mean(err * err)))

    inliers = np.abs(err) < max(0.15, min(0.40, rmse * 2.5))
    if min_points <= int(np.count_nonzero(inliers)) < xs.size:
        xs2 = xs[inliers]; ys2 = ys[inliers]
        if float(np.max(xs2) - np.min(xs2)) >= FIT_MIN_X_SPAN:
            try:
                coef = np.polyfit(xs2, ys2, 2)
            except Exception:
                return None
            a, b, _ = coef
            if abs(a) > FIT_CURVE_MAX or abs(b) > FIT_SLOPE_MAX:
                return None
            pred = np.polyval(coef, xs2)
            rmse = float(np.sqrt(np.mean((ys2 - pred) ** 2)))
            xs = xs2; ys = ys2; x_span = float(np.max(xs2) - np.min(xs2))

    if rmse > FIT_MAX_RMSE:
        return None

    near_x = float(np.min(xs))
    score = rmse - 0.08 * x_span - 0.0015 * xs.size + 0.03 * max(0.0, near_x - 2.0)
    return score, coef


def _reference_y(ref, xs):
    arr = np.asarray(ref, dtype=np.float64)
    if arr.shape == (3,):
        return np.polyval(arr, xs)
    return np.full_like(xs, float(arr))


def _binned_points(xs, ys):
    """x 방향으로 점을 균등화해 한 구간 잡음이 fit을 끌고 가지 않게 한다."""
    if xs.size == 0:
        return xs, ys
    bins = np.floor((xs - FIT_X_MIN) / FIT_BIN_WIDTH).astype(np.int32)
    bx, by = [], []
    for b in np.unique(bins):
        m = bins == b
        if int(np.count_nonzero(m)) < 2:
            continue
        bx.append(float(np.median(xs[m])))
        by.append(float(np.median(ys[m])))
    if len(bx) < FIT_MIN_BINS:
        return np.array([]), np.array([])
    return np.array(bx, dtype=np.float64), np.array(by, dtype=np.float64)


def _fit_poly2_candidate(xs, ys):
    if xs.size < FIT_MIN_POINTS:
        return None

    bx, by = _binned_points(xs, ys)
    if bx.size >= FIT_MIN_BINS:
        cand = _score_poly2(bx, by, min_points=FIT_MIN_BINS)
    else:
        cand = _score_poly2(xs, ys)
    if cand is None:
        return None

    coef = cand[1]
    for _ in range(FIT_REFINE_ITERS):
        err = np.abs(ys - np.polyval(coef, xs))
        inliers = err <= FIT_INLIER_BAND
        if int(np.count_nonzero(inliers)) < FIT_MIN_POINTS:
            break
        refined = _score_poly2(xs[inliers], ys[inliers])
        if refined is None:
            break
        coef = refined[1]

    scored = _score_poly2(xs[np.abs(ys - np.polyval(coef, xs)) <= FIT_INLIER_BAND],
                          ys[np.abs(ys - np.polyval(coef, xs)) <= FIT_INLIER_BAND])
    if scored is None:
        return cand[0], coef
    return scored[0], coef


def _fit_jump_ok(prev, new):
    if prev is None or new is None:
        return True
    check_xs = np.array([2.0, TARGET_X, 7.0], dtype=np.float64)
    jump = np.max(np.abs(np.polyval(new, check_xs) - np.polyval(prev, check_xs)))
    return bool(jump <= FIT_TARGET_MAX_JUMP)


def _best_seeded_fit(xs, ys, seeds, allow_global=True):
    """예상 y/곡선 주변 점만 골라 robust 2차식 fit."""
    candidates = []
    for s in seeds:
        band = np.abs(ys - _reference_y(s, xs)) <= FIT_BAND_WIDTH
        if int(np.count_nonzero(band)) < FIT_MIN_POINTS:
            continue
        cand = _fit_poly2_candidate(xs[band], ys[band])
        if cand is not None:
            candidates.append(cand)
    if candidates:
        candidates.sort(key=lambda c: c[0])
        return candidates[0][1]
    if not allow_global:
        return None
    cand = _fit_poly2_candidate(xs, ys)
    return cand[1] if cand is not None else None


def _fit_yellow(xs, ys):
    """노랑은 중앙선 1개라고 가정 — 단일 fit."""
    if xs is None or ys is None or xs.size < FIT_MIN_POINTS:
        return None
    roi = (xs > FIT_X_MIN) & (xs < FIT_X_MAX) & (np.abs(ys) < 2.5)
    xs = xs[roi]; ys = ys[roi]
    if xs.size < FIT_MIN_POINTS:
        return None

    if _prev_yellow_fit is not None:
        seeds = [_prev_yellow_fit, 0.0, float(np.median(ys))]
    else:
        seeds = [0.0, float(np.median(ys))]
    near = xs < 3.5
    if int(np.count_nonzero(near)) >= 6:
        seeds.append(float(np.median(ys[near])))
    fit = _best_seeded_fit(xs, ys, seeds)
    return fit if _fit_jump_ok(_prev_yellow_fit, fit) else None


def _fit_whites(xs, ys, yellow_fit):
    """흰색은 좌/우 외곽선 2개로 분리해 각각 fit."""
    if xs is None or ys is None or xs.size < FIT_MIN_POINTS:
        return None, None
    roi = (xs > FIT_X_MIN) & (xs < FIT_X_MAX) & (np.abs(ys) < 2.5)
    xs = xs[roi]; ys = ys[roi]
    if xs.size < FIT_MIN_POINTS:
        return None, None

    # 노랑 기준 좌/우 분리. 흰색 배경 물체가 fit에 섞이지 않도록
    # 차선폭 근처에 있는 흰 점만 후보로 쓴다.
    if yellow_fit is not None:
        yellow_y = np.polyval(yellow_fit, xs)
        left_offset = ys - yellow_y
        right_offset = yellow_y - ys
        left_mask = ((left_offset >= MIN_LANE_WIDTH)
                     & (left_offset <= MAX_LANE_WIDTH)
                     & (np.abs(left_offset - LANE_WIDTH) <= WHITE_YELLOW_OFFSET_TOL))
        right_mask = ((right_offset >= MIN_LANE_WIDTH)
                      & (right_offset <= MAX_LANE_WIDTH)
                      & (np.abs(right_offset - LANE_WIDTH) <= WHITE_YELLOW_OFFSET_TOL))
        expected_left = _shifted(yellow_fit, +LANE_WIDTH)
        expected_right = _shifted(yellow_fit, -LANE_WIDTH)
    else:
        left_mask = ((ys > 0.10)
                     & (np.abs(ys - LANE_WIDTH) <= WHITE_YELLOW_OFFSET_TOL))
        right_mask = ((ys < -0.10)
                      & (np.abs(ys + LANE_WIDTH) <= WHITE_YELLOW_OFFSET_TOL))
        expected_left = LANE_WIDTH
        expected_right = -LANE_WIDTH

    prefer_left = expected_left if _prev_white_left_fit is None else _prev_white_left_fit
    prefer_right = expected_right if _prev_white_right_fit is None else _prev_white_right_fit

    white_left = None
    if int(np.count_nonzero(left_mask)) >= FIT_MIN_POINTS:
        white_left = _best_seeded_fit(
            xs[left_mask], ys[left_mask],
            [prefer_left, expected_left, LANE_WIDTH],
            allow_global=False,
        )
        if not _fit_jump_ok(_prev_white_left_fit, white_left):
            white_left = None

    white_right = None
    if int(np.count_nonzero(right_mask)) >= FIT_MIN_POINTS:
        white_right = _best_seeded_fit(
            xs[right_mask], ys[right_mask],
            [prefer_right, expected_right, -LANE_WIDTH],
            allow_global=False,
        )
        if not _fit_jump_ok(_prev_white_right_fit, white_right):
            white_right = None

    # 동일 차선 dedup (TARGET_X에서 너무 가까우면 더 신뢰 가능한 쪽만 유지)
    if white_left is not None and white_right is not None:
        ly = float(np.polyval(white_left, TARGET_X))
        ry = float(np.polyval(white_right, TARGET_X))
        if abs(ly - ry) < WHITE_DEDUP_DIST:
            # 둘 중 prev에 더 가까운 쪽만 살림
            if _prev_white_left_fit is not None and ly >= 0:
                white_right = None
            elif _prev_white_right_fit is not None and ry < 0:
                white_left = None
            else:
                # prev 없으면 y 부호 우선
                if ly + ry >= 0:
                    white_right = None
                else:
                    white_left = None

    return white_left, white_right


def _shifted(coef, dy):
    out = np.array(coef, dtype=np.float64).copy()
    out[2] += dy
    return out


def _smooth_fit(prev, new):
    if new is None:
        return prev
    if prev is None:
        return np.asarray(new, dtype=np.float64)
    return _FIT_SMOOTH_ALPHA * np.asarray(new, dtype=np.float64) + (1.0 - _FIT_SMOOTH_ALPHA) * prev


# ======================== 차선 선택 ========================

def _select_lane(yellow_fit, white_left_fit, white_right_fit, preferred=PREFERRED_LANE):
    """노랑 + 좌/우 흰색에서 PREFERRED_LANE 사이드 경계 두 개 선택.

    preferred="left"  → (white_left, yellow_center)
    preferred="right" → (yellow_center, white_right)
    """
    if preferred == "left":
        # 좌측 차선 = white_left(좌) ~ yellow(우)
        if yellow_fit is not None and white_left_fit is not None:
            return white_left_fit, yellow_fit
        if yellow_fit is not None:
            return _shifted(yellow_fit, +LANE_WIDTH), yellow_fit
        if white_left_fit is not None:
            return white_left_fit, _shifted(white_left_fit, -LANE_WIDTH)
        if white_right_fit is not None:
            # 흰색 우측만 보이면 차선이 LANE_WIDTH*2 떨어진 좌측 흰색을 가정
            wl = _shifted(white_right_fit, +LANE_WIDTH * 2)
            yc = _shifted(white_right_fit, +LANE_WIDTH)
            return wl, yc
        return None, None

    # preferred == "right"
    # 우측 차선 = yellow(좌) ~ white_right(우)
    if yellow_fit is not None and white_right_fit is not None:
        return yellow_fit, white_right_fit
    if yellow_fit is not None:
        return yellow_fit, _shifted(yellow_fit, -LANE_WIDTH)
    if white_right_fit is not None:
        return _shifted(white_right_fit, +LANE_WIDTH), white_right_fit
    if white_left_fit is not None:
        yc = _shifted(white_left_fit, -LANE_WIDTH)
        wr = _shifted(white_left_fit, -LANE_WIDTH * 2)
        return yc, wr
    return None, None


def _pair_width_ok(left_fit, right_fit):
    if left_fit is None or right_fit is None:
        return False
    for x in (1.5, TARGET_X, 7.5):
        lw = float(np.polyval(left_fit, x)) - float(np.polyval(right_fit, x))
        if not np.isfinite(lw) or lw < MIN_LANE_WIDTH or lw > MAX_LANE_WIDTH:
            return False
    return True


# ======================== 샘플 / 스무딩 ========================

def _sample_xs():
    return np.linspace(
        max(FIT_X_MIN, SAMPLE_X_MIN),
        min(FIT_X_MAX, SAMPLE_X_MAX),
        SAMPLE_COUNT,
    )


def _polyval_clipped(coef, xs):
    return np.clip(np.polyval(coef, xs), -TARGET_Y_LIMIT, TARGET_Y_LIMIT)


def _smooth_target_y(target_y, lane_seen):
    global _prev_target_y, _lost_lane_frames
    if lane_seen:
        _lost_lane_frames = 0
    else:
        _lost_lane_frames += 1
        if _lost_lane_frames <= LANE_LOST_HOLD_FRAMES:
            return _prev_target_y

    delta = float(target_y - _prev_target_y)
    if abs(delta) < TARGET_DEADBAND_M:
        target_y = _prev_target_y
    else:
        target_y = _prev_target_y + TARGET_SMOOTH_ALPHA * delta

    step = float(np.clip(target_y - _prev_target_y, -TARGET_MAX_STEP_M, TARGET_MAX_STEP_M))
    target_y = float(np.clip(_prev_target_y + step, -TARGET_Y_LIMIT, TARGET_Y_LIMIT))
    _prev_target_y = target_y
    return target_y


def _tv_score_poly2_candidate(xs, ys, prefer_y=None):
    if xs.size < FIT_MIN_POINTS:
        return None

    x_span = float(np.max(xs) - np.min(xs))
    if x_span < FIT_MIN_X_SPAN:
        return None

    try:
        coef = np.polyfit(xs, ys, 2)
    except Exception:
        return None

    a, b, _c = coef
    if abs(a) > FIT_CURVE_MAX or abs(b) > FIT_SLOPE_MAX:
        return None

    pred = np.polyval(coef, xs)
    err = ys - pred
    rmse = float(np.sqrt(np.mean(err * err)))

    inliers = np.abs(err) < max(0.18, min(0.45, rmse * 2.5))
    if np.count_nonzero(inliers) >= FIT_MIN_POINTS and np.count_nonzero(inliers) < xs.size:
        xs2 = xs[inliers]
        ys2 = ys[inliers]
        x_span2 = float(np.max(xs2) - np.min(xs2))
        if x_span2 >= FIT_MIN_X_SPAN:
            try:
                coef = np.polyfit(xs2, ys2, 2)
            except Exception:
                return None
            a, b, _c = coef
            if abs(a) > FIT_CURVE_MAX or abs(b) > FIT_SLOPE_MAX:
                return None
            pred = np.polyval(coef, xs2)
            err = ys2 - pred
            rmse = float(np.sqrt(np.mean(err * err)))
            xs = xs2
            ys = ys2
            x_span = x_span2

    if rmse > FIT_MAX_RMSE:
        return None

    near_x = float(np.min(xs))
    y_at_target = float(np.polyval(coef, TARGET_X))
    prefer_penalty = 0.0 if prefer_y is None else 0.25 * abs(y_at_target - prefer_y)
    score = rmse - 0.08 * x_span - 0.0015 * xs.size + 0.03 * max(0.0, near_x - 2.0) + prefer_penalty
    return score, coef


def _append_fit_candidate(candidates, xs, ys, mask, prefer_y=None):
    if np.count_nonzero(mask) < FIT_MIN_POINTS:
        return None
    cand = _tv_score_poly2_candidate(xs[mask], ys[mask], prefer_y)
    if cand is not None:
        candidates.append(cand)
        return cand[1]
    return None


def _window_seed_values(xs, ys):
    seeds = []
    ranges = [(FIT_X_MIN, FIT_X_MAX)]
    ranges.extend(FIT_ANCHOR_WINDOWS)
    for x0, x1 in ranges:
        m = (xs >= x0) & (xs <= x1)
        if np.count_nonzero(m) < FIT_MIN_POINTS:
            continue
        local_ys = ys[m]
        seeds.extend(np.linspace(
            float(np.percentile(local_ys, 5)),
            float(np.percentile(local_ys, 95)),
            7,
        ).tolist())
        seeds.extend(np.percentile(local_ys, [15, 35, 50, 65, 85]).tolist())
    return seeds


def _add_anchor_window_candidates(candidates, xs, ys, prefer_y=None):
    for x0, x1 in FIT_ANCHOR_WINDOWS:
        xmask = (xs >= x0) & (xs <= x1)
        if np.count_nonzero(xmask) < FIT_MIN_POINTS:
            continue

        local_xs = xs[xmask]
        local_ys = ys[xmask]
        local_seeds = np.percentile(local_ys, [10, 25, 40, 50, 60, 75, 90])
        if prefer_y is not None:
            local_seeds = np.append(local_seeds, float(prefer_y))

        for seed in local_seeds:
            local_band = xmask & (np.abs(ys - seed) <= FIT_BAND_WIDTH)
            local_coef = _append_fit_candidate(
                candidates, xs, ys, local_band, prefer_y)
            if local_coef is None:
                continue

            expanded = np.abs(ys - np.polyval(local_coef, xs)) <= FIT_EXPAND_BAND
            _append_fit_candidate(candidates, xs, ys, expanded, prefer_y)


def _tv_fit_poly2_from_points(xs, ys, prefer_y=None):
    if xs is None or ys is None:
        return None

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < FIT_MIN_POINTS:
        return None

    roi = (
        (xs > FIT_X_MIN) &
        (xs < FIT_X_MAX) &
        (np.abs(ys) < FIT_Y_ABS_MAX)
    )
    xs = xs[roi]
    ys = ys[roi]
    if xs.size < FIT_MIN_POINTS:
        return None

    candidates = []
    seeds = _window_seed_values(xs, ys)
    if prefer_y is not None:
        seeds.append(float(prefer_y))

    for seed in dict.fromkeys(round(float(s), 3) for s in seeds):
        band = np.abs(ys - seed) <= FIT_BAND_WIDTH
        _append_fit_candidate(candidates, xs, ys, band, prefer_y)

    _add_anchor_window_candidates(candidates, xs, ys, prefer_y)

    if not candidates:
        cand = _tv_score_poly2_candidate(xs, ys, prefer_y)
        if cand is not None:
            candidates.append(cand)

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _yellow_fit_jump_ok(prev_fit, new_fit):
    if prev_fit is None or new_fit is None:
        return True
    check_xs = np.array([1.8, TARGET_X, 7.5], dtype=np.float64)
    jump = np.max(np.abs(np.polyval(new_fit, check_xs) -
                         np.polyval(prev_fit, check_xs)))
    return bool(jump <= YELLOW_MAX_JUMP_M)


def _filter_yellow_direction_outliers(xs, ys):
    if _prev_yellow_fit is None or xs.size < YELLOW_DIRECTION_FILTER_MIN_POINTS:
        return xs, ys

    a, b, _ = np.asarray(_prev_yellow_fit, dtype=np.float64)
    ref_y = np.polyval(_prev_yellow_fit, xs)
    ref_slope = 2.0 * a * xs + b
    normal_err = np.abs(ys - ref_y) / np.sqrt(1.0 + ref_slope * ref_slope)
    keep = normal_err <= YELLOW_DIRECTION_OUTLIER_BAND_M
    return xs[keep], ys[keep]


def _select_yellow_closest_to_previous_fit(xs, ys):
    if _prev_yellow_fit is None or xs.size < YELLOW_DIRECTION_FILTER_MIN_POINTS:
        return xs, ys

    a, b, _ = np.asarray(_prev_yellow_fit, dtype=np.float64)
    ref_y = np.polyval(_prev_yellow_fit, xs)
    ref_slope = 2.0 * a * xs + b
    normal_err = np.abs(ys - ref_y) / np.sqrt(1.0 + ref_slope * ref_slope)
    bins = np.floor((xs - FIT_X_MIN) / YELLOW_DIRECTION_SELECT_BIN_M).astype(np.int32)

    selected = []
    for bid in np.unique(bins):
        idx = np.flatnonzero(bins == bid)
        if idx.size == 0:
            continue
        selected.append(idx[int(np.argmin(normal_err[idx]))])

    if len(selected) < YELLOW_DIRECTION_FILTER_MIN_POINTS:
        return xs[selected], ys[selected]
    selected = np.array(selected, dtype=np.int64)
    order = np.argsort(xs[selected])
    selected = selected[order]
    return xs[selected], ys[selected]


def _fit_yellow_sliding(xs, ys):
    """YOLO 노랑 점 → 단순 np.polyfit(2차) + jump 가드 + EMA smoothing.

    YOLO seg가 직접 클래스를 골라주므로 RANSAC 같은 outlier 제거가 불필요.
    ROI/계수 sanity·prev-fit 점프 가드·실패 시 hold만 유지.
    """
    global _prev_yellow_fit, _yellow_miss
    global _yellow_inlier_xs, _yellow_inlier_ys

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    roi = (xs > FIT_X_MIN) & (xs < FIT_X_MAX) & (np.abs(ys) < FIT_Y_ABS_MAX)
    xs = xs[roi]
    ys = ys[roi]
    xs, ys = _filter_yellow_direction_outliers(xs, ys)
    xs, ys = _select_yellow_closest_to_previous_fit(xs, ys)

    fit = None
    if (xs.size >= FIT_MIN_POINTS
            and float(np.max(xs) - np.min(xs)) >= FIT_MIN_X_SPAN):
        try:
            # 호모그래피 비선형성으로 멀리 픽셀 한 점 = 가까운 안정점 한 점이
            # 동일 가중되면 곡률(a)이 튄다. 1/(1+x²) 가중으로 가까운 점 우선.
            w = 1.0 / (1.0 + xs * xs)
            coef = np.polyfit(xs, ys, 2, w=w)
        except (np.linalg.LinAlgError, ValueError):
            coef = None
        if coef is not None and np.all(np.isfinite(coef)):
            a, b, _ = coef
            if (abs(a) <= FIT_CURVE_MAX
                    and abs(b) <= FIT_SLOPE_MAX
                    and _yellow_fit_jump_ok(_prev_yellow_fit, coef)):
                fit = coef

    if fit is None:
        _yellow_miss += 1
        if _prev_yellow_fit is not None and _yellow_miss <= YELLOW_MAX_MISS:
            return _prev_yellow_fit
        _prev_yellow_fit = None
        _yellow_inlier_xs = np.array([], dtype=np.float64)
        _yellow_inlier_ys = np.array([], dtype=np.float64)
        return None

    if _prev_yellow_fit is None:
        _prev_yellow_fit = np.asarray(fit, dtype=np.float64)
    else:
        _prev_yellow_fit = (
            (1.0 - YELLOW_FIT_SMOOTH_ALPHA) * _prev_yellow_fit +
            YELLOW_FIT_SMOOTH_ALPHA * np.asarray(fit, dtype=np.float64)
        )
    _yellow_miss = 0
    _yellow_inlier_xs = xs
    _yellow_inlier_ys = ys
    return _prev_yellow_fit


def _build_lane_plan_from_bounds(left_coef, right_coef, mode):
    xs = np.linspace(FIT_X_MIN, FIT_X_MAX, 60)
    left_coef = np.asarray(left_coef, dtype=np.float64)
    right_coef = np.asarray(right_coef, dtype=np.float64)
    center_coef = 0.5 * (left_coef + right_coef)
    left_y = np.polyval(left_coef, xs)
    right_y = np.polyval(right_coef, xs)
    center_y = np.polyval(center_coef, xs)
    target_left_y = float(np.polyval(left_coef, TARGET_X))
    target_right_y = float(np.polyval(right_coef, TARGET_X))
    raw_target_y = float(np.clip(0.5 * (target_left_y + target_right_y),
                                 -TARGET_Y_LIMIT, TARGET_Y_LIMIT))
    return {
        "mode": mode,
        "left_coef": left_coef,
        "right_coef": right_coef,
        "center_coef": center_coef,
        "xs": xs,
        "left_y": left_y,
        "right_y": right_y,
        "center_y": center_y,
        "target_x": TARGET_X,
        "target_left_y": target_left_y,
        "target_right_y": target_right_y,
        "raw_target_y": raw_target_y,
        "memory_age": 0,
    }


def _fit_white_on_yellow_side(white_xs, white_ys, yellow_coef, side):
    if white_xs is None or white_ys is None:
        return None
    white_xs = np.asarray(white_xs, dtype=np.float64)
    white_ys = np.asarray(white_ys, dtype=np.float64)
    if white_xs.size < FIT_MIN_POINTS:
        return None

    yellow_at_x = np.polyval(yellow_coef, white_xs)
    prefer_y = float(np.polyval(yellow_coef, TARGET_X))
    if side == "right":
        offset = yellow_at_x - white_ys
        side_mask = ((offset >= LANE_WIDTH_ACCEPT_MIN) &
                     (offset <= LANE_WIDTH_ACCEPT_MAX))
        prefer_y -= LANE_WIDTH
    else:
        offset = white_ys - yellow_at_x
        side_mask = ((offset >= LANE_WIDTH_ACCEPT_MIN) &
                     (offset <= LANE_WIDTH_ACCEPT_MAX))
        prefer_y += LANE_WIDTH

    return _tv_fit_poly2_from_points(white_xs[side_mask], white_ys[side_mask],
                                     prefer_y=prefer_y)


def _choose_nearest_lane_side(yellow_coef):
    """노랑 점선 기준 좌/우 차선 center 중 ego(y=0)에 가까운 쪽 선택."""
    global _prev_lane_side

    yellow_y = float(np.polyval(yellow_coef, TARGET_X))
    half_width = 0.5 * LANE_WIDTH
    costs = {
        "left": abs(yellow_y + half_width),
        "right": abs(yellow_y - half_width),
    }

    side = "left" if costs["left"] < costs["right"] else "right"
    if abs(costs["left"] - costs["right"]) < LANE_SIDE_SWITCH_MARGIN:
        side = _prev_lane_side or (
            PREFERRED_LANE if PREFERRED_LANE in ("left", "right") else "right"
        )
    elif _prev_lane_side in costs:
        # 이전 차선이 크게 불리해지기 전까지는 유지해서 좌우 선택 떨림을 줄인다.
        other = "left" if _prev_lane_side == "right" else "right"
        if costs[_prev_lane_side] <= costs[other] + LANE_SIDE_SWITCH_MARGIN:
            side = _prev_lane_side

    _prev_lane_side = side
    return side


def _choose_nearest_lane_plan(candidates):
    """검출된 lane pair 중 ego(y=0)에 가까운 lane center 선택."""
    global _prev_lane_side

    if not candidates:
        return None

    scored = []
    for side, plan in candidates:
        center_y = float(np.polyval(plan["center_coef"], TARGET_X))
        scored.append((abs(center_y), side, plan))
    scored.sort(key=lambda item: item[0])

    best_cost, best_side, best_plan = scored[0]
    if _prev_lane_side is not None:
        for cost, side, plan in scored:
            if side == _prev_lane_side and cost <= best_cost + LANE_SIDE_SWITCH_MARGIN:
                _prev_lane_side = side
                return plan

    _prev_lane_side = best_side
    return best_plan


def _child_dash_stats(xs):
    if xs.size < CHILD_DASH_MIN_POINTS:
        return None
    order = np.sort(np.asarray(xs, dtype=np.float64))
    x_span = float(order[-1] - order[0])
    if x_span < CHILD_DASH_MIN_X_SPAN:
        return None

    bin_ids = np.unique(
        np.floor((order - FIT_X_MIN) / CHILD_DASH_BIN_M).astype(np.int32))
    if bin_ids.size < CHILD_DASH_MIN_BINS:
        return None
    bin_span = max(1, int(bin_ids[-1] - bin_ids[0] + 1))
    occupancy = float(bin_ids.size / bin_span)

    x_gap = float(np.max(np.diff(order))) if order.size >= 2 else 0.0
    if bin_ids.size >= 2:
        bin_gap = float(np.max(np.diff(bin_ids) - 1) * CHILD_DASH_BIN_M)
    else:
        bin_gap = 0.0
    max_gap = max(x_gap, bin_gap)
    dash_like = (
        max_gap >= CHILD_DASH_MIN_GAP_M
        or occupancy <= CHILD_DASH_MAX_OCCUPANCY
    )
    return {
        "x_span": x_span,
        "occupancy": occupancy,
        "max_gap": max_gap,
        "dash_like": dash_like,
    }


def _fit_child_dash_candidate(xs, ys):
    if xs.size < CHILD_DASH_MIN_POINTS:
        return None
    x_span = float(np.max(xs) - np.min(xs))
    if x_span < CHILD_DASH_MIN_X_SPAN:
        return None
    try:
        w = 1.0 / (1.0 + xs * xs)
        coef = np.polyfit(xs, ys, 2, w=w)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.all(np.isfinite(coef)):
        return None
    a, b, _ = coef
    if abs(a) > FIT_CURVE_MAX or abs(b) > FIT_SLOPE_MAX:
        return None

    err = np.abs(ys - np.polyval(coef, xs))
    inliers = err <= CHILD_DASH_BAND_M
    if int(np.count_nonzero(inliers)) >= CHILD_DASH_MIN_POINTS:
        try:
            w = 1.0 / (1.0 + xs[inliers] * xs[inliers])
            refined = np.polyfit(xs[inliers], ys[inliers], 2, w=w)
        except (np.linalg.LinAlgError, ValueError):
            refined = None
        if refined is not None and np.all(np.isfinite(refined)):
            a, b, _ = refined
            if abs(a) <= FIT_CURVE_MAX and abs(b) <= FIT_SLOPE_MAX:
                coef = refined
                err = np.abs(ys - np.polyval(coef, xs))
                inliers = err <= CHILD_DASH_BAND_M

    if int(np.count_nonzero(inliers)) < CHILD_DASH_MIN_POINTS:
        return None
    rmse = float(np.sqrt(np.mean(err[inliers] * err[inliers])))
    if rmse > CHILD_DASH_MAX_RMSE:
        return None
    return coef, rmse


def _child_dash_seed_values(xs, ys):
    seeds = []
    if _prev_child_yellow_dash_fit is not None:
        seeds.append(float(np.polyval(_prev_child_yellow_dash_fit, TARGET_X)))

    band = ((xs >= max(FIT_X_MIN, TARGET_X - 2.5))
            & (xs <= min(FIT_X_MAX, TARGET_X + 3.5)))
    y_ref = ys[band] if int(np.count_nonzero(band)) >= CHILD_DASH_MIN_POINTS else ys
    if y_ref.size >= CHILD_DASH_MIN_POINTS:
        y_sorted = np.sort(y_ref)
        splits = np.where(np.diff(y_sorted) > 0.45)[0] + 1
        for group in np.split(y_sorted, splits):
            if group.size >= CHILD_DASH_MIN_POINTS:
                seeds.append(float(np.median(group)))

    seeds.extend(_window_seed_values(xs, ys))
    return list(dict.fromkeys(round(float(s), 3) for s in seeds))


def _child_dash_jump_ok(prev, new):
    if prev is None or new is None:
        return True
    check_xs = np.array([1.8, TARGET_X, 7.5], dtype=np.float64)
    jump = np.max(np.abs(np.polyval(new, check_xs) - np.polyval(prev, check_xs)))
    return bool(jump <= CHILD_DASH_MAX_JUMP_M)


def _fit_yellow_child_dash(xs, ys):
    """어린이 보호구역에서는 연속 실선보다 x 방향으로 끊긴 노란 점선을 우선한다."""
    global _prev_child_yellow_dash_fit, _child_yellow_dash_miss
    global _yellow_inlier_xs, _yellow_inlier_ys

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    roi = (
        (xs > FIT_X_MIN) &
        (xs < FIT_X_MAX) &
        (np.abs(ys) < FIT_Y_ABS_MAX)
    )
    xs = xs[roi]
    ys = ys[roi]
    if xs.size < CHILD_DASH_MIN_POINTS:
        _child_yellow_dash_miss += 1
        if (_prev_child_yellow_dash_fit is not None
                and _child_yellow_dash_miss <= CHILD_DASH_MAX_MISS):
            return _prev_child_yellow_dash_fit
        return None

    candidates = []
    seeds = _child_dash_seed_values(xs, ys)
    if not seeds:
        seeds = [float(np.median(ys))]

    for seed in seeds:
        mask = np.abs(ys - seed) <= max(FIT_BAND_WIDTH, CHILD_DASH_BAND_M)
        cand = _fit_child_dash_candidate(xs[mask], ys[mask])
        if cand is None:
            continue
        coef, rmse = cand
        if not _child_dash_jump_ok(_prev_child_yellow_dash_fit, coef):
            continue
        inlier = np.abs(ys - np.polyval(coef, xs)) <= CHILD_DASH_BAND_M
        stats = _child_dash_stats(xs[inlier])
        if stats is None or not stats["dash_like"]:
            continue
        prev_penalty = 0.0
        if _prev_child_yellow_dash_fit is not None:
            prev_penalty = 0.18 * abs(
                float(np.polyval(coef, TARGET_X))
                - float(np.polyval(_prev_child_yellow_dash_fit, TARGET_X))
            )
        score = (
            rmse
            - 0.035 * stats["x_span"]
            - 0.45 * stats["max_gap"]
            - 0.55 * (1.0 - stats["occupancy"])
            + prev_penalty
        )
        candidates.append((score, coef, inlier))

    if not candidates:
        _child_yellow_dash_miss += 1
        if (_prev_child_yellow_dash_fit is not None
                and _child_yellow_dash_miss <= CHILD_DASH_MAX_MISS):
            return _prev_child_yellow_dash_fit
        return None

    candidates.sort(key=lambda item: item[0])
    fit = np.asarray(candidates[0][1], dtype=np.float64)
    inlier = candidates[0][2]

    if _prev_child_yellow_dash_fit is None:
        _prev_child_yellow_dash_fit = fit
    else:
        _prev_child_yellow_dash_fit = (
            (1.0 - CHILD_DASH_SMOOTH_ALPHA) * _prev_child_yellow_dash_fit
            + CHILD_DASH_SMOOTH_ALPHA * fit
        )
    _child_yellow_dash_miss = 0
    _yellow_inlier_xs = xs[inlier]
    _yellow_inlier_ys = ys[inlier]
    return _prev_child_yellow_dash_fit


def _choose_lane_plan_from_yellow(yellow_coef, white_xs, white_ys):
    candidates = []

    white_left = _fit_white_on_yellow_side(white_xs, white_ys, yellow_coef, "left")
    if white_left is not None:
        plan = _build_lane_plan_from_bounds(white_left, yellow_coef, "yellow_white_left")
        if _pair_width_ok(plan["left_coef"], plan["right_coef"]):
            candidates.append(("left", plan))

    white_right = _fit_white_on_yellow_side(white_xs, white_ys, yellow_coef, "right")
    if white_right is not None:
        plan = _build_lane_plan_from_bounds(yellow_coef, white_right, "yellow_white_right")
        if _pair_width_ok(plan["left_coef"], plan["right_coef"]):
            candidates.append(("right", plan))

    selected = _choose_nearest_lane_plan(candidates)
    if selected is not None:
        return selected

    side = _choose_nearest_lane_side(yellow_coef)

    if side == "right":
        left_coef = yellow_coef
        right_coef = _shifted(yellow_coef, -LANE_WIDTH)
        mode = "yellow_synth_right"
    else:
        right_coef = yellow_coef
        left_coef = _shifted(yellow_coef, LANE_WIDTH)
        mode = "yellow_synth_left"

    return _build_lane_plan_from_bounds(left_coef, right_coef, mode)


def _choose_lane_plan_from_white(white_xs, white_ys):
    if white_xs is None or white_ys is None:
        return None
    white_xs = np.asarray(white_xs, dtype=np.float64)
    white_ys = np.asarray(white_ys, dtype=np.float64)
    roi = (
        (white_xs > FIT_X_MIN) &
        (white_xs < WHITE_FALLBACK_X_MAX) &
        (np.abs(white_ys) < WHITE_FALLBACK_Y_ABS_MAX)
    )
    if np.count_nonzero(roi) < FIT_MIN_POINTS:
        return None

    white_xs = white_xs[roi]
    white_ys = white_ys[roi]
    if PREFERRED_LANE == "right":
        side_mask = white_ys < 0.2
        prefer_y = -LANE_WIDTH * 0.5
    else:
        side_mask = white_ys > -0.2
        prefer_y = LANE_WIDTH * 0.5

    white_coef = _tv_fit_poly2_from_points(white_xs[side_mask], white_ys[side_mask],
                                           prefer_y=prefer_y)
    if white_coef is None:
        white_coef = _tv_fit_poly2_from_points(white_xs, white_ys,
                                               prefer_y=prefer_y)
    if white_coef is None:
        return None

    white_y = float(np.polyval(white_coef, TARGET_X))
    if white_y > 0:
        left_coef = white_coef
        right_coef = _shifted(white_coef, -LANE_WIDTH)
        mode = "white_synth_from_left"
    else:
        right_coef = white_coef
        left_coef = _shifted(white_coef, LANE_WIDTH)
        mode = "white_synth_from_right"

    return _build_lane_plan_from_bounds(left_coef, right_coef, mode)


def _stabilize_lane_plan(lane_plan):
    global _prev_lane_plan, _lane_memory_frames

    if lane_plan is None:
        if _prev_lane_plan is None or _lane_memory_frames >= LANE_MEMORY_FRAMES:
            return None
        _lane_memory_frames += 1
        mem = _build_lane_plan_from_bounds(
            _prev_lane_plan["left_coef"],
            _prev_lane_plan["right_coef"],
            f"{_prev_lane_plan['mode']}_memory{_lane_memory_frames}",
        )
        mem["memory_age"] = _lane_memory_frames
        _prev_lane_plan = mem
        return mem

    if _prev_lane_plan is not None:
        left_coef = (
            (1.0 - LANE_PLAN_ALPHA) * _prev_lane_plan["left_coef"] +
            LANE_PLAN_ALPHA * lane_plan["left_coef"]
        )
        right_coef = (
            (1.0 - LANE_PLAN_ALPHA) * _prev_lane_plan["right_coef"] +
            LANE_PLAN_ALPHA * lane_plan["right_coef"]
        )
        lane_plan = _build_lane_plan_from_bounds(left_coef, right_coef, lane_plan["mode"])

    _lane_memory_frames = 0
    _prev_lane_plan = lane_plan
    return lane_plan


# ======================== ROS 노드 ========================

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__("path_planner_node")

        self._yellow_xs = np.array([])
        self._yellow_ys = np.array([])
        self._white_xs = np.array([])
        self._white_ys = np.array([])
        self._obstacles = []
        self._has_lane = False

        self.create_subscription(PoseArray, "/fused/lane", self._on_lane, 10)
        self.create_subscription(PoseArray, "/fused/obstacles", self._on_obs, 10)

        self._target_pub = self.create_publisher(PointStamped, "/target", 10)
        self._path_pub = self.create_publisher(PoseArray, "/center_path", 10)
        self._lane_left_pub = self.create_publisher(PoseArray, "/lane_left", 10)
        self._lane_right_pub = self.create_publisher(PoseArray, "/lane_right", 10)
        self._lane_fits_pub = self.create_publisher(PoseArray, "/lane_fits", 10)
        self._yellow_inliers_pub = self.create_publisher(
            PoseArray, "/fit/yellow_inliers", 10)
        self.create_timer(1.0 / PLAN_HZ, self._tick)

        self.get_logger().info("LANEPLAN")

    def _on_lane(self, msg: PoseArray):
        self._has_lane = True
        if not msg.poses:
            self._yellow_xs = np.array([], dtype=np.float64)
            self._yellow_ys = np.array([], dtype=np.float64)
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
        wm = np.isin(cls, list(WHITE_CLS_IDS))
        self._yellow_xs = xs[ym]
        self._yellow_ys = ys[ym]
        self._white_xs = xs[wm]
        self._white_ys = ys[wm]

    def _on_obs(self, msg: PoseArray):
        self._obstacles = [
            (p.position.x, p.position.y, p.position.z) for p in msg.poses
        ]

    def _tick(self):
        if not self._has_lane:
            return

        result = plan(self._yellow_xs, self._yellow_ys,
                      self._white_xs, self._white_ys,
                      self._obstacles)
        now = self.get_clock().now().to_msg()

        # 항상 발행: lane_plan 실패 프레임에도 viewer에서 fit 입력 상태를 볼 수 있게.
        self._yellow_inliers_pub.publish(
            self._make_pose_array(now, _yellow_inlier_xs, _yellow_inlier_ys))

        if result is None:
            return

        target_x, target_y = result["target"]
        sample_xs = result["sample_xs"]

        tgt = PointStamped()
        tgt.header.stamp = now
        tgt.header.frame_id = "lidar_frame"
        tgt.point.x = float(target_x)
        tgt.point.y = float(target_y)
        self._target_pub.publish(tgt)

        self._path_pub.publish(self._make_pose_array(now, sample_xs, result["sample_ys"]))
        self._lane_left_pub.publish(self._make_pose_array(now, sample_xs, result["left_ys"]))
        self._lane_right_pub.publish(self._make_pose_array(now, sample_xs, result["right_ys"]))

        all_fits = result.get("all_fits", [])
        if all_fits:
            flat = PoseArray()
            flat.header.stamp = now
            flat.header.frame_id = "lidar_frame"
            for coef in all_fits:
                ys = _polyval_clipped(coef, sample_xs)
                for x, y in zip(sample_xs, ys):
                    p = Pose()
                    p.position.x = float(x)
                    p.position.y = float(y)
                    flat.poses.append(p)
            self._lane_fits_pub.publish(flat)

    @staticmethod
    def _make_pose_array(stamp, xs, ys):
        out = PoseArray()
        out.header.stamp = stamp
        out.header.frame_id = "lidar_frame"
        for x, y in zip(xs, ys):
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            out.poses.append(p)
        return out


# ======================== 경로계획 ========================

def plan(yellow_xs, yellow_ys, white_xs, white_ys, obstacles,
         child_zone=False, yellow_dash_xs=None, yellow_dash_ys=None):
    """노랑(MID) 2차식 자체를 center로 사용 — 노란 중앙선을 직접 따라간다.

    흰선 점은 안 쓴다 (signature는 호환 위해 유지).
    """
    yellow_xs = np.asarray(yellow_xs, dtype=np.float64)
    yellow_ys = np.asarray(yellow_ys, dtype=np.float64)

    if child_zone:
        dash_xs = np.asarray(
            yellow_dash_xs if yellow_dash_xs is not None else yellow_xs,
            dtype=np.float64)
        dash_ys = np.asarray(
            yellow_dash_ys if yellow_dash_ys is not None else yellow_ys,
            dtype=np.float64)
        yellow_fit = _fit_yellow_child_dash(dash_xs, dash_ys)
        if yellow_fit is None:
            yellow_fit = _fit_yellow_sliding(yellow_xs, yellow_ys)
    else:
        yellow_fit = _fit_yellow_sliding(yellow_xs, yellow_ys)
    lane_seen = yellow_fit is not None
    if yellow_fit is None:
        return None

    yellow_fit = np.asarray(yellow_fit, dtype=np.float64)
    center_coef = yellow_fit.copy()

    sample_xs = _sample_xs()
    sample_ys = _polyval_clipped(center_coef, sample_xs)

    raw_target_y = float(np.clip(np.polyval(center_coef, TARGET_X),
                                 -TARGET_Y_LIMIT, TARGET_Y_LIMIT))
    target_y = _smooth_target_y(raw_target_y, lane_seen)

    # 장애물 회피
    for ox, oy, _r in obstacles or []:
        if AVOID_X_MIN < ox < AVOID_X_MAX and abs(oy - target_y) < AVOID_LATERAL:
            if oy >= target_y:
                target_y -= AVOID_SHIFT
            else:
                target_y += AVOID_SHIFT
            target_y = float(np.clip(target_y, -TARGET_Y_LIMIT, TARGET_Y_LIMIT))
            sample_ys = np.clip(sample_ys + (target_y - raw_target_y),
                                -TARGET_Y_LIMIT, TARGET_Y_LIMIT)
            break

    # 시각화용 좌/우 — 노랑을 따라가니까 ±LANE_HALF_WIDTH로 lane bounds만 표시.
    left_coef = yellow_fit.copy(); left_coef[2] += LANE_HALF_WIDTH
    right_coef = yellow_fit.copy(); right_coef[2] -= LANE_HALF_WIDTH

    return {
        "target": (TARGET_X, target_y),
        "sample_xs": sample_xs,
        "sample_ys": sample_ys,
        "left_ys": _polyval_clipped(left_coef, sample_xs),
        "right_ys": _polyval_clipped(right_coef, sample_xs),
        "all_fits": [yellow_fit, left_coef, right_coef],
        "mode": "child_yellow_dash" if child_zone else "yellow_center",
    }


# ======================== main ========================

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
