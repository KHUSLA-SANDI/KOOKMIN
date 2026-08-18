#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""녹화 사진을 실시간 카메라처럼 재생하며 인지→경로 전 과정을 한 화면에 본다.

왜 필요한가
  국민대 트랙은 왕복 3~4시간이라 자주 못 간다. 그래서 거기서 딴 녹화를 노트북에서
  돌려보며 "이 그림에서 차선이 어떻게 잡히고, 그 점들로 어떤 경로가 나오는지"를
  확인한다. 차도 ROS 도 필요 없다 — 이 파일 하나로 돈다.

화면 (좌 → 우)
  1. 원본      : 녹화된 프레임 그대로
  2. YOLO      : 검출 마스크/박스 오버레이 (실제 주행에서 모델이 보는 것)
  3. BEV+경로  : H 로 편 지면 좌표에 차선점 + lane_planner 가 맞춘 곡선 + 목표점

실행 (WSL 에서, 창은 WSLg 로 뜬다)
  python3 tools/replay_viewer.py \
      --dir "/mnt/d/Desktop/국민대 자율주행/녹화_20260805/photo1" \
      --model "/mnt/d/Desktop/국민대 자율주행/best_640.pt"

  여러 폴더를 이어서 보려면 --dir 를 여러 번 준다.

조작
  space 재생/일시정지     . / ,  한 장 앞/뒤 (일시정지 중)
  + / - 재생 속도         [ / ]  검출 임계값 conf 내림/올림
  u     왜곡보정 on/off   b      모델 입력 색순서 BGR/RGB
  g     BEV 격자 on/off   s      현재 화면 저장
  q     종료

주의
  · 이 모델(best_640.pt)은 640x480 으로 학습됐는데 녹화는 1920x1080 이다.
    640 모드는 센서를 잘라낸 것(화각 가로 66.7%)이라 화각이 다르다. 그래도
    검출이 되는지 보려는 것이므로 기본은 프레임 전체를 넣는다.
  · BEV 좌표는 config/camera.yaml 의 H 를 쓴다. 이 H 는 1920x1080 보정영상
    기준이라, 녹화 해상도가 다르면 좌표가 틀어진다 (기동 시 확인해 경고한다).
"""
import argparse
import glob
import importlib.util
import os
import sys
import time

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.realpath(__file__))
_PKG = os.path.dirname(_HERE)          # .../track_drive
sys.path.insert(0, _PKG)

from track_drive.lib import bev as B                      # noqa: E402


def _stub_ros():
    """노트북에는 ROS 가 없다. lane_planner 는 rclpy 를 임포트하지만 실제로는
    파일 끝의 PathPlannerNode 클래스와 main() 에서만 쓴다 — plan() 자체는
    순수 numpy 다. 그래서 껍데기 모듈만 끼워 넣어 파일을 읽을 수 있게 한다.
    (ROS 를 설치하지 않고 노트북에서 같은 경로 계산을 재현하기 위한 것)
    """
    import types
    if "rclpy" not in sys.modules:
        try:
            import rclpy       # noqa: F401 — 있으면 그대로 쓴다
        except ImportError:
            rclpy = types.ModuleType("rclpy")
            rclpy.init = lambda *a, **k: None
            rclpy.spin = lambda *a, **k: None
            rclpy.shutdown = lambda *a, **k: None
            node_mod = types.ModuleType("rclpy.node")

            class _Node(object):
                def __init__(self, *a, **k):
                    pass
            node_mod.Node = _Node
            rclpy.node = node_mod
            sys.modules["rclpy"] = rclpy
            sys.modules["rclpy.node"] = node_mod
    if "geometry_msgs.msg" not in sys.modules:
        try:
            import geometry_msgs.msg    # noqa: F401
        except ImportError:
            g = types.ModuleType("geometry_msgs")
            gm = types.ModuleType("geometry_msgs.msg")
            for nm in ("Pose", "PoseArray", "PointStamped"):
                setattr(gm, nm, type(nm, (object,), {}))
            g.msg = gm
            sys.modules["geometry_msgs"] = g
            sys.modules["geometry_msgs.msg"] = gm


_stub_ros()

# lane_planner 는 패키지 상대임포트가 없어 파일에서 직접 읽어 쓴다
_lp_spec = importlib.util.spec_from_file_location(
    "lane_planner", os.path.join(_PKG, "track_drive", "lane_planner.py"))
lane_planner = importlib.util.module_from_spec(_lp_spec)
sys.modules["lane_planner"] = lane_planner
_lp_spec.loader.exec_module(lane_planner)

# ── BEV 표시 범위 (보기용. lib/bev.py 의 런타임 캔버스와는 별개) ──
BEV_X0, BEV_X1 = -0.5, 5.5      # 전방 m (아래 → 위)
BEV_Y0, BEV_Y1 = -2.5, 2.5      # 좌우 m (오른쪽 → 왼쪽)

# 이 모델의 클래스 이름 → 우리 파이프라인에서의 역할.
# ★ 이름으로 찾는다. best_640.pt 는 4클래스라 예선 lane.pt(12클래스)와 번호가 다르다.
ROLE_YELLOW = ("mid",)              # 노란 중앙선 — plan() 이 이걸로 경로를 만든다
ROLE_WHITE = ("lane",)              # 흰 외곽선 — 그리기만 (plan 은 안 씀)

C_YELLOW = (0, 220, 255)
C_WHITE = (230, 230, 230)
C_CENTER = (80, 255, 80)
C_LEFT = (255, 180, 60)
C_RIGHT = (255, 180, 60)
C_TARGET = (0, 0, 255)
# 세 가지 피팅 (BGR)
C_FITY = (60, 255, 60)        # 노랑 점만        — 초록
C_FITW = (255, 230, 0)        # 흰 점만          — 하늘
C_FITB = (255, 90, 255)       # 노랑 + 흰 합침   — 자홍
C_GRID = (60, 60, 60)
C_AXIS = (110, 110, 110)


def _load_text():
    """tools/tuner.py 의 Text 클래스를 빌려 쓴다 (한글 폰트 탐색 + PIL 렌더).

    cv2.putText 는 한글을 못 그려 '???' 가 된다. tuner.py 가 이미 리눅스/WSL/윈도우
    폰트 경로를 훑는 Text 를 갖고 있으므로 중복 구현하지 않고 파일에서 불러온다.
    (tools 에 __init__.py 가 없어 일반 import 는 안 된다 — importlib 로 읽는다)
    """
    p = os.path.join(_HERE, "tuner.py")
    if not os.path.isfile(p):
        return None
    try:
        spec = importlib.util.spec_from_file_location("_tuner_text", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_tuner_text"] = mod
        spec.loader.exec_module(mod)
        t = mod.Text()
        return t if getattr(t, "ok", False) else None
    except Exception:      # noqa: BLE001 — 폰트가 없어도 뷰어는 떠야 한다
        return None


_TEXT = _load_text()


def put(img, text, org, scale=0.5, color=(255, 255, 255), th=1):
    """한글 가능하면 PIL 로, 아니면 cv2 로. org 는 baseline 이 아니라 좌상단 기준."""
    if _TEXT is not None:
        px = max(11, int(round(scale * 30)))
        _TEXT.draw(img, [(text, org[0], org[1] - px, px, color)])
        return
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, th, cv2.LINE_AA)


class BevCanvas:
    """지면 좌표(m) → 화면 픽셀. x 전방(위), y 좌측(왼쪽) — REP-103."""

    def __init__(self, height):
        self.h = int(height)
        self.ppm = self.h / (BEV_X1 - BEV_X0)
        self.w = int(round((BEV_Y1 - BEV_Y0) * self.ppm))

    def px(self, x, y):
        c = int(round((BEV_Y1 - y) * self.ppm))
        r = int(round((BEV_X1 - x) * self.ppm))
        return c, r

    def blank(self, grid=True):
        img = np.zeros((self.h, self.w, 3), np.uint8)
        if not grid:
            return img
        gx = int(np.floor(BEV_X0)) + 1
        while gx <= BEV_X1:
            _, r = self.px(gx, 0)
            cv2.line(img, (0, r), (self.w - 1, r), C_GRID, 1)
            put(img, "%dm" % gx, (4, r - 3), 0.38, (120, 120, 120))
            gx += 1
        gy = int(np.floor(BEV_Y0)) + 1
        while gy <= BEV_Y1:
            c, _ = self.px(0, gy)
            cv2.line(img, (c, 0), (c, self.h - 1), C_GRID, 1)
            gy += 1
        # 원점(라이다 지면 투영) 십자
        c0, r0 = self.px(0.0, 0.0)
        cv2.line(img, (c0 - 8, r0), (c0 + 8, r0), C_AXIS, 1)
        cv2.line(img, (c0, r0 - 8), (c0, r0 + 8), C_AXIS, 1)
        # y=0 중심선
        cv2.line(img, (c0, 0), (c0, self.h - 1), (45, 45, 45), 1)
        return img

    def dots(self, img, xs, ys, color, r=2):
        for x, y in zip(np.ravel(xs), np.ravel(ys)):
            if not (BEV_X0 <= x <= BEV_X1 and BEV_Y0 <= y <= BEV_Y1):
                continue
            c, rr = self.px(float(x), float(y))
            cv2.circle(img, (c, rr), r, color, -1, cv2.LINE_AA)

    def curve(self, img, xs, ys, color, th=2):
        pts = []
        for x, y in zip(np.ravel(xs), np.ravel(ys)):
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            pts.append(self.px(float(x), float(y)))
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, color, th, cv2.LINE_AA)


def lp_reset():
    """lane_planner 의 프레임 간 상태를 지운다.

    plan() 은 직전 피팅/목표를 기억해 EMA·점프가드를 건다. 파라미터를 바꿨는데
    이 상태가 남아 있으면 이전 설정의 값이 섞여 비교가 오염된다.
    """
    for nm, val in (("_prev_target_y", 0.0), ("_lost_lane_frames", 0),
                    ("_prev_left_fit", None), ("_prev_right_fit", None),
                    ("_prev_yellow_fit", None), ("_prev_white_left_fit", None),
                    ("_prev_white_right_fit", None), ("_prev_lane_plan", None),
                    ("_lane_memory_frames", 0), ("_prev_lane_side", None),
                    ("_yellow_miss", 0), ("_prev_child_yellow_dash_fit", None),
                    ("_child_yellow_dash_miss", 0)):
        if hasattr(lane_planner, nm):
            setattr(lane_planner, nm, val)
    for nm in ("_yellow_inlier_xs", "_yellow_inlier_ys"):
        if hasattr(lane_planner, nm):
            setattr(lane_planner, nm, np.array([], dtype=np.float64))


def lp_set(fit_min, fit_max, samp_min, samp_max, target_x):
    """lane_planner 의 범위 상수를 바꿔 끼운다 (모듈 전역이라 대입만으로 적용된다)."""
    lane_planner.FIT_X_MIN = float(fit_min)
    lane_planner.FIT_X_MAX = float(fit_max)
    lane_planner.SAMPLE_X_MIN = float(samp_min)
    lane_planner.SAMPLE_X_MAX = float(samp_max)
    lane_planner.TARGET_X = float(target_x)
    lp_reset()


def fit2(xs, ys, fmin, fmax):
    """2차식 피팅. lane_planner._fit_yellow_sliding 의 핵심만 떼어낸 것.

    같은 가중(1/(1+x²) — 가까운 점 우선)과 같은 ROI/최소조건을 쓰되, 프레임 간
    상태(점프가드·EMA)는 뺐다. 노랑/흰/합침 세 조합을 같은 잣대로 비교하기 위해서다.
    반환: (계수, 쓴 점 수, 실패사유)
    """
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    if xs.size == 0:
        return None, 0, "점 없음"
    roi = ((xs > fmin) & (xs < fmax)
           & (np.abs(ys) < lane_planner.FIT_Y_ABS_MAX))
    xs, ys = xs[roi], ys[roi]
    if xs.size < lane_planner.FIT_MIN_POINTS:
        return None, int(xs.size), "점 %d<%d" % (xs.size, lane_planner.FIT_MIN_POINTS)
    span = float(xs.max() - xs.min())
    if span < lane_planner.FIT_MIN_X_SPAN:
        return None, int(xs.size), "스팬 %.2f<%.2f" % (span, lane_planner.FIT_MIN_X_SPAN)
    try:
        w = 1.0 / (1.0 + xs * xs)
        coef = np.polyfit(xs, ys, 2, w=w)
    except (np.linalg.LinAlgError, ValueError):
        return None, int(xs.size), "피팅 실패"
    if not np.all(np.isfinite(coef)):
        return None, int(xs.size), "계수 비정상"
    return coef, int(xs.size), ""


def yellow_gate(xs, ys):
    """경로가 왜 안 나오는지 — lane_planner 의 관문을 순서대로 재현해 보여준다.

    화면에 "점 N개 최소 M 필요" 만 찍으면 점은 충분한데 x 폭이나 곡률에서
    죽은 프레임을 오해하게 된다. 실제로 막은 곳을 짚어준다.
    반환: (표시할 문구, 통과했나)
    """
    lp = lane_planner
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    n = xs.size
    if n < lp.FIT_MIN_POINTS:
        return "점 %d개 < %d" % (n, lp.FIT_MIN_POINTS), False
    roi = (xs > lp.FIT_X_MIN) & (xs < lp.FIT_X_MAX) & (np.abs(ys) < 2.5)
    xs, ys = xs[roi], ys[roi]
    if xs.size < lp.FIT_MIN_POINTS:
        return "ROI 안 %d개 < %d" % (xs.size, lp.FIT_MIN_POINTS), False
    span = float(np.max(xs) - np.min(xs))
    if span < lp.FIT_MIN_X_SPAN:
        return "x폭 %.2f < %.2fm" % (span, lp.FIT_MIN_X_SPAN), False
    try:
        coef = np.polyfit(xs, ys, 2)
    except Exception:                       # noqa: BLE001
        return "polyfit 실패", False
    a, b, _ = coef
    if abs(a) > lp.FIT_CURVE_MAX:
        return "곡률 %.2f > %.2f" % (abs(a), lp.FIT_CURVE_MAX), False
    if abs(b) > lp.FIT_SLOPE_MAX:
        return "기울기 %.2f > %.2f" % (abs(b), lp.FIT_SLOPE_MAX), False
    rmse = float(np.sqrt(np.mean((ys - np.polyval(coef, xs)) ** 2)))
    if rmse > lp.FIT_MAX_RMSE:
        return "RMSE %.3f > %.2f" % (rmse, lp.FIT_MAX_RMSE), False
    return "통과 (%d점 x폭%.2fm)" % (xs.size, span), True


def white_spread(xs, ys, fmin, fmax):
    """흰 점이 한 줄인지 두 줄인지 가늠. 중간 x 구간에서 y 의 폭을 잰다.

    좌·우 흰선이 한 마스크로 합쳐져 있어, 폭이 크면 두 줄(양쪽 다 보임),
    작으면 한 줄(한쪽만 보임)이다. 한 줄만 보이는데 그걸로 피팅하면
    중앙이 아니라 그 선을 따라가므로 반 차선 폭만큼 치우친다.
    """
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    if xs.size < 4:
        return float("nan")
    lo, hi = fmin + 0.25 * (fmax - fmin), fmin + 0.75 * (fmax - fmin)
    sel = (xs >= lo) & (xs <= hi)
    if sel.sum() < 4:
        sel = np.ones_like(xs, bool)
    yy = ys[sel]
    return float(np.percentile(yy, 90) - np.percentile(yy, 10))


def bev_points(mask, H, res, sub, binm, minpx):
    """마스크 → BEV 워프 → x구간별 중심점. lib/bev.py 의 계산을 파라미터화한 것.

    왜 lib 을 그대로 안 쓰나: lib 의 상수(BEV_RES 0.05 / SUBSAMPLE 3 / BIN 0.18)는
    예선 시뮬 기준이라 실제 트랙의 얇은 테이프 선에서는 점이 1~5개밖에 안 나온다
    (plan() 은 12개 필요). 어느 값이 맞는지 눈으로 보고 정하려고 조절 가능하게 했다.
    정해지면 lib/bev.py 에 반영한다.
    """
    w = int(round((2 * B.BEV_Y_HALF) / res))
    h = int(round((B.BEV_X_MAX - B.BEV_X_MIN) / res))
    mgrid = np.array([[0.0, -1.0 / res, B.BEV_Y_HALF / res],
                      [-1.0 / res, 0.0, B.BEV_X_MAX / res],
                      [0.0, 0.0, 1.0]], np.float64)
    bm = cv2.warpPerspective(mask, mgrid @ np.asarray(H, np.float64), (w, h),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    vs, us = np.where(bm > 0)
    if vs.size == 0:
        return (np.array([], np.float32), np.array([], np.float32), 0)
    raw_px = int(vs.size)
    vs, us = vs[::sub], us[::sub]
    xs = B.BEV_X_MAX - vs.astype(np.float64) * res
    ys = -(us.astype(np.float64) * res - B.BEV_Y_HALF)
    bins = np.floor(xs / binm).astype(np.int64)
    ox, oy = [], []
    for b in np.unique(bins):
        sel = bins == b
        if int(sel.sum()) < minpx:
            continue
        ox.append(float(np.median(xs[sel])))
        oy.append(float(np.median(ys[sel])))
    if not ox:
        return (np.array([], np.float32), np.array([], np.float32), raw_px)
    ox = np.asarray(ox, np.float32)
    oy = np.asarray(oy, np.float32)
    order = np.argsort(ox)
    return ox[order], oy[order], raw_px


def build_masks(result, names, want_names, shape):
    """YOLO 결과에서 지정한 이름의 클래스만 골라 이진 마스크 합성."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    n = 0
    if result is None or result.masks is None or result.boxes is None:
        return m, n
    if len(result.boxes) == 0:
        return m, n
    cls = result.boxes.cls.cpu().numpy().astype(int)
    data = result.masks.data.cpu().numpy()
    want_ids = {i for i, nm in names.items() if nm in want_names}
    for i, c in enumerate(cls):
        if int(c) not in want_ids:
            continue
        mi = data[i]
        if mi.shape != (h, w):
            mi = cv2.resize(mi, (w, h), interpolation=cv2.INTER_NEAREST)
        m[mi > 0.5] = 255
        n += 1
    return m, n


def overlay(frame, result, names, alpha=0.45):
    """검출 마스크/박스를 원본 위에 그린다."""
    out = frame.copy()
    if result is None or result.boxes is None or len(result.boxes) == 0:
        put(out, "검출 없음", (12, 30), 0.7, (0, 0, 255), 2)
        return out
    h, w = frame.shape[:2]
    palette = {"mid": (0, 220, 255), "lane": (230, 230, 230),
               "stop": (0, 0, 255), "bonsun-1": (255, 120, 255)}
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    tint = np.zeros_like(out)
    if result.masks is not None:
        data = result.masks.data.cpu().numpy()
        for i, c in enumerate(cls):
            nm = names.get(int(c), str(c))
            col = palette.get(nm, (0, 255, 0))
            mi = data[i]
            if mi.shape != (h, w):
                mi = cv2.resize(mi, (w, h), interpolation=cv2.INTER_NEAREST)
            tint[mi > 0.5] = col
        out = cv2.addWeighted(out, 1.0, tint, alpha, 0)
    for i, (c, cf) in enumerate(zip(cls, conf)):
        nm = names.get(int(c), str(c))
        col = palette.get(nm, (0, 255, 0))
        x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy()[i].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        put(out, "%s %.2f" % (nm, cf), (x1, max(18, y1 - 6)), 0.6, col, 2)
    return out


def fit_h(img, h):
    """세로 h 에 맞춰 비율 유지 축소."""
    s = h / float(img.shape[0])
    return cv2.resize(img, (int(round(img.shape[1] * s)), h),
                      interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True,
                    help="사진 폴더 (여러 번 지정 가능)")
    ap.add_argument("--model", required=True, help=".pt 경로")
    ap.add_argument("--camera-yaml",
                    default=os.path.join(_PKG, "config", "camera.yaml"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--fps", type=float, default=5.0, help="재생 속도")
    ap.add_argument("--panel-h", type=int, default=380, help="패널 세로 픽셀")
    ap.add_argument("--rgb", action="store_true", default=True,
                    help="모델 입력을 RGB 로 (기본). --no-rgb 로 BGR")
    ap.add_argument("--no-rgb", dest="rgb", action="store_false")
    ap.add_argument("--no-undistort", dest="undistort", action="store_false",
                    default=True)
    ap.add_argument("--save-dir", default=os.path.expanduser("~/replay_shots"))
    # ── 점 추출 파라미터 (기본값은 실측으로 고른 것 — 아래 주석 참고) ──
    # lib/bev.py 기본(0.05 / 3 / 0.18 / 2)은 예선 시뮬 기준이라 실제 트랙의 얇은
    # 테이프에서 점이 중앙값 1개밖에 안 나왔다(24프레임 전부 경로 실패).
    # 아래 조합은 같은 표본에서 점 20개 / 83% 성공. 워프 비용은 0.14→0.82ms 로
    # YOLO 추론 45ms 대비 무시할 수준이다.
    ap.add_argument("--bev-res", type=float, default=0.02)
    ap.add_argument("--sub", type=int, default=1)
    ap.add_argument("--bin", dest="binm", type=float, default=0.05)
    ap.add_argument("--minpx", type=int, default=2)
    # ── 피팅 범위 / 목표점 (기본값 근거는 실측) ──
    # 차선점의 99%가 x<2.63m 안에 있고, 차체 마스크가 x=0.25m 를 하한으로 만든다.
    # TARGET_X 는 3.0(예선값)에서 프레임간 목표 흔들림이 0.22m(p95 1.06m)였는데
    # 1.0 으로 줄이면 0.076m(p95 0.26m)로 3분의 1이 된다.
    ap.add_argument("--fit-x-min", type=float, default=0.25)
    ap.add_argument("--fit-x-max", type=float, default=2.5)
    ap.add_argument("--sample-x-min", type=float, default=0.3)
    ap.add_argument("--sample-x-max", type=float, default=2.5)
    ap.add_argument("--target-x", type=float, default=1.0)
    ap.add_argument("--no-car-mask", dest="car_mask", action="store_false",
                    default=True, help="차체 가림선을 적용하지 않는다(비교용)")
    a = ap.parse_args()

    files = []
    for d in a.dir:
        got = sorted(glob.glob(os.path.join(d, "*.jpg")))
        files += got
        print("  %-50s %d장" % (os.path.basename(d.rstrip("/")), len(got)))
    files = [f for f in files if os.path.getsize(f) > 0]
    if not files:
        print("★ 사진이 없다"); return 1
    print("총 %d장" % len(files))

    cam = B.load_camera_config(a.camera_yaml)
    H = np.asarray(cam["H"], np.float64)
    calib_wh = tuple(cam["image_size"] or (0, 0))
    print("camera.yaml: %dx%d  h_calibrated=%s  %s"
          % (calib_wh[0], calib_wh[1], cam["h_calibrated"], cam["note"] or ""))

    probe = cv2.imread(files[0])
    if probe is None:
        print("★ 첫 장을 못 읽는다"); return 1
    ih, iw = probe.shape[:2]
    if (iw, ih) != calib_wh:
        print("▲ 경고: 녹화 %dx%d 가 캘리브 기준 %dx%d 와 다르다 — BEV 좌표가 틀어진다"
              % (iw, ih, calib_wh[0], calib_wh[1]))

    undist = B.Undistorter.from_config(cam, (iw, ih))
    print("undistort: %s (%s)" % (undist.enabled, undist.model))

    from ultralytics import YOLO
    model = YOLO(a.model)
    names = dict(model.names)
    print("model: task=%s names=%s" % (model.task, names))
    if not any(n in names.values() for n in ROLE_YELLOW):
        print("▲ 경고: 이 모델에 %s 클래스가 없다 — 경로 생성이 안 된다" % (ROLE_YELLOW,))
    if model.task != "segment":
        print("▲ 경고: segment 모델이 아니다 — 마스크가 없어 BEV 점이 안 나온다")

    os.makedirs(a.save_dir, exist_ok=True)
    canvas = BevCanvas(a.panel_h)
    WIN = "replay — 원본 | YOLO | BEV+경로"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    idx, playing, conf = 0, True, float(a.conf)
    use_undist, use_rgb, show_grid = a.undistort, a.rgb, True
    fps = float(a.fps)
    res, sub, binm, minpx = a.bev_res, a.sub, a.binm, a.minpx
    fmin, fmax = a.fit_x_min, a.fit_x_max
    smin, smax, tgtx = a.sample_x_min, a.sample_x_max, a.target_x
    use_cmask = a.car_mask
    cmv = cam["car_mask_v"]
    if cmv is None:
        print("▲ camera.yaml 에 car_mask_v 가 없다 — 차체 가림선을 적용하지 않는다")
        use_cmask = False
    else:
        q = H @ np.array([float(iw) / 2.0, float(cmv), 1.0])
        print("차체 가림선 car_mask_v=%.0f → 지면 x≈%.2f m" % (cmv, q[0] / q[2]))
    lp_set(fmin, fmax, smin, smax, tgtx)

    last = 0.0
    cache = {}
    hist = []          # 최근 프레임의 경로 생성 성공 여부 (성공률 표시용)
    tys = []           # 최근 목표 y — 프레임간 흔들림 측정용

    while True:
        now = time.time()
        if playing and (now - last) >= (1.0 / max(0.1, fps)):
            idx = (idx + 1) % len(files)
            last = now

        # YOLO 결과는 검출 파라미터에만 의존하므로 그것만 캐시한다
        # (점 추출 파라미터를 돌릴 때 추론을 다시 하지 않게)
        ykey = (idx, conf, use_undist, use_rgb)
        if ykey in cache:
            raw, frame, det, my, mw, ny, nw, dt = cache[ykey]
        else:
            raw = cv2.imread(files[idx])
            if raw is None:
                idx = (idx + 1) % len(files)
                continue
            frame = undist.apply(raw) if use_undist else raw
            src = frame[:, :, ::-1] if use_rgb else frame
            t0 = time.perf_counter()
            r0 = model.predict(np.ascontiguousarray(src), conf=conf,
                               imgsz=a.imgsz, device="cpu", verbose=False)[0]
            dt = (time.perf_counter() - t0) * 1000
            det = overlay(frame, r0, names)
            my, ny = build_masks(r0, names, ROLE_YELLOW, frame.shape[:2])
            mw, nw = build_masks(r0, names, ROLE_WHITE, frame.shape[:2])
            if len(cache) > 25:
                cache.clear()
            cache[ykey] = (raw, frame, det, my, mw, ny, nw, dt)

        # ★ 런타임(yolo_node)과 동일하게 차체 가림선을 먼저 적용한다.
        #   안 하면 차체 화소가 x≈0.2m 의 가짜 점이 되어 피팅을 끌어당긴다.
        myc, mwc = my, mw
        if use_cmask:
            myc = B.apply_car_mask(my.copy(), cmv, calib_h=frame.shape[0])
            mwc = B.apply_car_mask(mw.copy(), cmv, calib_h=frame.shape[0])
        yx, yy, rawpx = bev_points(myc, H, res, sub, binm, minpx)
        wx, wy, _ = bev_points(mwc, H, res, sub, binm, minpx)

        # ── 경로 생성 ──
        bevimg = canvas.blank(show_grid)
        # 흰 점은 위치 감만 잡으라고 흐리게 남긴다 (피팅에는 안 쓴다)
        canvas.dots(bevimg, wx, wy, (70, 70, 70), 1)
        canvas.dots(bevimg, yx, yy, C_YELLOW, 3)
        plan_txt = "-"
        try:
            pl = lane_planner.plan(yx, yy, wx, wy, [])
        except Exception as e:            # noqa: BLE001 — 뷰어가 죽지 않게
            pl, plan_txt = None, "plan 예외: %s" % type(e).__name__
        # ── 노랑 기반 피팅 하나만 그린다 ──
        #   흰선/합침 피팅은 뺐다. plan() 이 흰선을 안 쓰므로(lane_planner:1209)
        #   화면에 그려봤자 실제 주행과 상관없는 선이라 판단만 흐린다.
        bx = np.linspace(max(0.05, fmin), fmax, 40)
        ycoef, ynpt, ywhy = fit2(yx, yy, fmin, fmax)
        legend = []
        if ycoef is not None:
            canvas.curve(bevimg, bx, np.polyval(ycoef, bx), C_FITY, 2)
            legend.append(("노랑 피팅", "%d점" % ynpt, C_FITY))
        else:
            legend.append(("노랑 피팅", ywhy, (90, 90, 90)))
        gate_msg, gate_ok = yellow_gate(yx, yy)
        legend.append(("관문", gate_msg, C_FITY if gate_ok else (110, 110, 255)))

        if pl is not None:
            sx = np.asarray(pl["sample_xs"], float)
            canvas.curve(bevimg, sx, np.asarray(pl["left_ys"], float), C_LEFT, 1)
            canvas.curve(bevimg, sx, np.asarray(pl["right_ys"], float), C_RIGHT, 1)
            tx, ty = pl["target"]
            c, r = canvas.px(float(tx), float(ty))
            cv2.circle(bevimg, (c, r), 6, C_TARGET, -1, cv2.LINE_AA)
            cv2.circle(bevimg, (c, r), 10, C_TARGET, 1, cv2.LINE_AA)
            plan_txt = "목표 x=%.2f y=%+.2f  (%s)" % (tx, ty, pl.get("mode", "-"))
        elif plan_txt == "-":
            # ★ 예전엔 무조건 "점 N개, 최소 M 필요" 라고만 찍어서, 점은 충분한데
            #   x 폭이나 곡률에서 죽은 프레임까지 점 부족처럼 보였다. 실제로 막은
            #   관문을 그대로 보여준다.
            plan_txt = "경로 없음 — %s" % gate_msg

        hist.append(pl is not None)
        if len(hist) > 120:
            hist.pop(0)
        rate = 100.0 * sum(hist) / max(1, len(hist))

        # 목표점이 프레임마다 얼마나 튀는가 = 조향이 얼마나 떨릴까
        tys.append(float(pl["target"][1]) if pl is not None else np.nan)
        if len(tys) > 120:
            tys.pop(0)
        arr = np.array(tys, float)
        d = np.abs(np.diff(arr[np.isfinite(arr)])) if np.isfinite(arr).sum() > 2 else None
        jit_med = float(np.median(d)) if d is not None and d.size else float("nan")
        jit_p95 = float(np.percentile(d, 95)) if d is not None and d.size else float("nan")

        # 피팅에 실제로 쓰인 구간을 BEV 에 표시 (ROI 밖 점은 버려진다)
        for xline, col in ((fmin, (90, 90, 160)), (fmax, (90, 90, 160))):
            _, rr = canvas.px(xline, 0)
            if 0 <= rr < canvas.h:
                cv2.line(bevimg, (0, rr), (canvas.w - 1, rr), col, 1)

        # ── 합성 ──
        p1 = fit_h(raw, a.panel_h)
        p2 = fit_h(det, a.panel_h)
        for img, t in ((p1, "1. 원본"), (p2, "2. YOLO  %d검출  %.0fms" % (ny + nw, dt))):
            put(img, t, (10, 24), 0.62, (255, 255, 255), 2)
        put(bevimg, "3. BEV + 경로", (10, 24), 0.62, (255, 255, 255), 2)
        for li, (nm, info, col) in enumerate(legend):
            y = 46 + li * 17
            cv2.line(bevimg, (12, y - 4), (30, y - 4), col, 2)
            put(bevimg, "%s %s" % (nm, info), (36, y), 0.40, col, 1)

        sep = np.full((a.panel_h, 2, 3), 70, np.uint8)
        top = np.hstack([p1, sep, p2, sep, bevimg])

        bar = np.zeros((110, top.shape[1], 3), np.uint8)
        put(bar, "[%d/%d] %s   |  %s  conf %.2f  %s  %s  차체마스크%s  %.1ffps"
            % (idx + 1, len(files), os.path.basename(files[idx]),
               "▶재생" if playing else "❚❚정지", conf,
               "보정ON" if use_undist else "보정OFF",
               "RGB" if use_rgb else "BGR", "ON" if use_cmask else "OFF", fps),
            (10, 20), 0.5, (200, 200, 200))
        yspan = (float(np.max(yx) - np.min(yx))) if len(yx) else 0.0
        put(bar, "★노란점 %d개  (x폭 %.2fm, 검출 %d덩이→BEV화소 %d)   |  %s"
            % (len(yx), yspan, ny, rawpx, plan_txt),
            (10, 40), 0.5, (120, 255, 255) if len(yx) >= lane_planner.FIT_MIN_POINTS
            else (120, 120, 255))
        put(bar, "점추출  RES %.3f [1/2]  SUB %d [3/4]  BIN %.3f [5/6]  MIN %d [7/8]"
            % (res, sub, binm, minpx), (10, 60), 0.48, (255, 220, 120))
        put(bar, "범위  FIT %.2f~%.2f [z/x, c/v]   TARGET_X %.2f [n/m]   "
            "SAMPLE %.1f~%.1f" % (fmin, fmax, tgtx, smin, smax),
            (10, 80), 0.48, (255, 180, 220))
        put(bar, "최근%d  경로성공 %.0f%%   목표흔들림 중앙 %.3fm p95 %.3fm"
            "     [k]차체마스크 [0]예선값 [9]권장값 [s]저장 [q]종료"
            % (len(hist), rate, jit_med, jit_p95),
            (10, 100), 0.46, (150, 220, 255))
        cv2.imshow(WIN, np.vstack([top, bar]))

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q") or k == 27:
            break
        elif k == ord(" "):
            playing = not playing
        elif k == ord("."):
            idx = (idx + 1) % len(files); playing = False
        elif k == ord(","):
            idx = (idx - 1) % len(files); playing = False
        elif k in (ord("+"), ord("=")):
            fps = min(60.0, fps * 1.5)
        elif k == ord("-"):
            fps = max(0.2, fps / 1.5)
        elif k == ord("["):
            conf = max(0.01, round(conf - 0.05, 2))
        elif k == ord("]"):
            conf = min(0.95, round(conf + 0.05, 2))
        elif k == ord("u"):
            use_undist = not use_undist
        elif k == ord("b"):
            use_rgb = not use_rgb
        elif k == ord("g"):
            show_grid = not show_grid
        # ── 점 추출 파라미터 (바꾸면 성공률 집계를 리셋한다) ──
        elif k == ord("1"):
            res = max(0.005, round(res - 0.005, 3)); hist.clear()
        elif k == ord("2"):
            res = min(0.10, round(res + 0.005, 3)); hist.clear()
        elif k == ord("3"):
            sub = max(1, sub - 1); hist.clear()
        elif k == ord("4"):
            sub = min(10, sub + 1); hist.clear()
        elif k == ord("5"):
            binm = max(0.01, round(binm - 0.01, 3)); hist.clear()
        elif k == ord("6"):
            binm = min(0.50, round(binm + 0.01, 3)); hist.clear()
        elif k == ord("7"):
            minpx = max(1, minpx - 1); hist.clear()
        elif k == ord("8"):
            minpx = min(20, minpx + 1); hist.clear()
        # ── 피팅 범위 / 목표점 ──
        elif k in (ord("z"), ord("x"), ord("c"), ord("v"), ord("n"), ord("m")):
            if k == ord("z"):
                fmin = max(0.0, round(fmin - 0.05, 2))
            elif k == ord("x"):
                fmin = min(fmax - 0.1, round(fmin + 0.05, 2))
            elif k == ord("c"):
                fmax = max(fmin + 0.1, round(fmax - 0.25, 2))
            elif k == ord("v"):
                fmax = min(12.0, round(fmax + 0.25, 2))
            elif k == ord("n"):
                tgtx = max(0.3, round(tgtx - 0.1, 2))
            elif k == ord("m"):
                tgtx = min(11.0, round(tgtx + 0.1, 2))
            smin, smax = max(0.1, fmin), fmax
            lp_set(fmin, fmax, smin, smax, tgtx)
            hist.clear(); tys.clear()
        elif k == ord("k"):
            use_cmask = (not use_cmask) and (cmv is not None)
            hist.clear(); tys.clear()
        elif k == ord("0"):
            # 예선 시뮬 기준값으로 되돌려 비교
            res, sub, binm, minpx = B.BEV_RES, B.LANE_PIX_SUBSAMPLE, \
                B.LANE_CENTERLINE_BIN_M, 2
            fmin, fmax, smin, smax, tgtx = 0.5, 12.0, 1.0, 11.0, 3.0
            lp_set(fmin, fmax, smin, smax, tgtx)
            hist.clear(); tys.clear()
        elif k == ord("9"):
            # 실측으로 고른 권장값
            res, sub, binm, minpx = 0.02, 1, 0.05, 2
            fmin, fmax, smin, smax, tgtx = 0.25, 2.5, 0.3, 2.5, 1.0
            lp_set(fmin, fmax, smin, smax, tgtx)
            hist.clear(); tys.clear()
        elif k == ord("s"):
            p = os.path.join(a.save_dir, "replay_%06d.jpg" % idx)
            cv2.imwrite(p, np.vstack([top, bar]), [cv2.IMWRITE_JPEG_QUALITY, 92])
            print("저장: %s" % p)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
