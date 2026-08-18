#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
캘리브 수동 튜너 — 왜곡보정(K/D) 과 지면 호모그래피(H) 를 손으로 돌려 맞춘다.

역할 : 파라미터를 키보드로 바꾸면 [보정영상 | BEV] 가 즉시 다시 그려진다.
       자동 최적화 없음. 눈으로 보고 맞추는 도구다.
입력 : config/camera.yaml (초기값), 정지 이미지(기본) 또는 /image_raw (--live)
출력 : [s] 로 camera.yaml 갱신 (기존 파일은 .bak 으로 백업)

▣ 왜 new_K 를 얼리나
   보정영상의 픽셀 좌표계는 new_K 가 정한다. k1 을 건드릴 때마다 new_K 를 다시
   계산하면 화면이 통째로 움직여서 H 도 같이 무효가 된다. 그래서 시작할 때 한 번
   계산한 new_K 를 고정한다. K/D 를 크게 바꿔 화면 가장자리에 검은 여백이 생기면
   [N] 으로 다시 계산할 것 (그 순간 H 는 다시 맞춰야 한다).

▣ 화면
   왼쪽  = 보정영상. 초록 격자 = 현재 H 로 지면 격자를 되쏜 것.
           → 초록선이 실제 줄눈/테이프 위에 얹히면 H 가 맞다.
   오른쪽= BEV. 노랑 = 타일 피치 격자, 초록 = 0.5 m 눈금.
           → 타일이 정사각형이고 선이 곧으면 맞다.

▣ 조작
   TAB       그룹 전환 (왜곡 K/D  ↔  지면 H)
   ↑↓ / w x  파라미터 선택          ←→ / a d   값 -, +
   [ ]       스텝 ÷10, ×10          0          선택 항목만 초기값으로
   R         전부 초기값으로        m          H 모드 전환 (물리 6개 ↔ 행렬 9개)
   L         자로 재기 모드 — 보정영상에서 선 양끝을 2번 클릭하면
             그 구간의 "직선에서 벗어난 최대/RMS px" 를 계속 표시한다
   C         잰 선 전부 지우기      g  격자 on/off      e  대비강화 on/off
   N         new_K 재계산(H 무효)   n  다음 이미지      SPACE 라이브 프레임 갱신
   s         camera.yaml 저장       q / ESC  종료

실행 (차 PC 터미널):
   cd ~/xycar_ws/src/track_drive
   python3 tools/tuner.py                       # 저장된 사진으로
   python3 tools/tuner.py --live                # 실시간 카메라로
"""
import argparse
import os
import shutil
import sys
import threading
import time

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.realpath(__file__))
_PKG = os.path.dirname(_HERE)
sys.path.insert(0, _PKG)
from track_drive.lib import bev as B                # noqa: E402

try:
    import yaml
except ImportError:
    yaml = None


# ============================ 파라미터 정의 ============================
# (이름, 설명, 기본 스텝)  — 설명은 화면 하단에 그대로 뜬다.
# 감도 숫자는 실측 검증값이다 (640x480, new_K 고정). "스텝 1칸" = 방향키 한 번.
# ★ 코너의 정규화 반지름이 r=1.08~1.20 (>1) 이라 고차항이 코너에서 더 크게 작동한다 —
#   코너 영향 서열이 k3 > k2 > k1 로 직관과 반대다. 화면 중앙 가로줄에서만 k1 > k2 > k3.
DIST_PARAMS = [
    ("k1", "1차 방사왜곡 ★먼저 이걸로. 더 음수=배럴 더 세게 편다. 1칸=코너 3.3px 좌우끝 1.6px", 0.005),
    ("k2", "2차 방사왜곡. k1 보다 가장자리에 집중(집중비 3.5배). 1칸=코너 4.4px 좌우끝 1.2px", 0.005),
    ("k3", "3차 방사왜곡 ▲코너 최강. 1칸=코너 4~8px. 맨 마지막에 아주 조금만", 0.002),
    ("cx", "왜곡 중심 u. 좌우 휨 정도가 다를 때. 1칸=좌우 비대칭 보정", 0.5),
    ("cy", "왜곡 중심 v. 위아래 휨 정도가 다를 때", 0.5),
    ("p2", "접선왜곡(가로) ▲D 중 유일하게 좌우 부호가 같다 = 좌우 쏠림 잡는 노브. 1칸 0.7px", 0.0005),
    ("p1", "접선왜곡(세로). 코너 0.5px, 가로중앙선에선 p2 의 1/3만 작동", 0.0005),
    ("fx", "초점거리 px(가로) ▲건드리지 마. k 로 상쇄가 안 됨(58% 잔존)", 1.0),
    ("fy", "초점거리 px(세로) ▲건드리지 마. cy 중심 세로배율이라 화면 위쪽만 움직임", 1.0),
]
PHYS_PARAMS = [
    ("pitch",  "카메라 내림각 deg ★. 크게 = 먼 곳이 더 당겨짐(BEV 세로 압축)", 0.1),
    ("height", "카메라 지면높이 m ★. 크게 = BEV 전체가 균일하게 커짐", 0.002),
    ("yaw",    "좌우 틀어짐 deg. 크게 = BEV 가 시계방향으로 돈다", 0.1),
    ("roll",   "좌우 기울기 deg. 좌우 한쪽이 들리는 변형", 0.1),
    ("cam_x",  "원점→카메라 전방 오프셋 m. BEV 를 통째로 앞뒤로 민다", 0.005),
    ("cam_y",  "원점→카메라 좌 오프셋 m. BEV 를 통째로 좌우로 민다", 0.005),
]
# ▲ 부호 주의: 이 H 는 작업영역에서 W<0 (1m 지점 W=-0.243) 이라 직관과 반대다.
#   1~2행(h00~h12)을 키우면 그 좌표가 줄고, 3행(h20~h22)을 키우면 x,y 가 함께 늘어난다.
# ▲ 어떤 원소도 "거리와 무관한 평행이동"이 아니다 — 전부 1/W 를 공통인자로 가져서
#   먼 곳에서 효과가 커진다 (2m/1m 배율: h01·h11 1.7배, h00·h02·h10·h12 1.9배,
#   h21 3.4배, h22·h20 3.8배).
# (이름, H 안의 저장 위치 인덱스(행*3+열), 설명) — 화면 순서는 튜닝 우선순위대로다.
H_ELEM_DOC = [
    ("h01", 1, "v→x ★세로(전방) 축척 주항. +면 거리가 줄어든다. 1칸 -12mm@1m (-21mm@2m)"),
    ("h10", 3, "u→y ★가로(좌우) 축척 주항. +면 y 가 줄어든다. 1칸 -13mm@1m"),
    ("h02", 2, "x 배율(상수항이지만 평행이동 아님). 1칸 -4.1mm@1m, -7.8mm@2m ≒ 거리의 -0.4%"),
    ("h12", 5, "y 배율. 1칸 -4.1mm@1m. h02 와 같은 성질"),
    ("h21", 7, "v 원근항 ★. 1칸 +6.0mm@1m, +20mm@2m — 근거리 맞은 뒤 원거리만 틀릴 때"),
    ("h11", 4, "v→y. 순수 전단(회전:스큐 1:1). +면 y 가 줄어든다. 1칸 -12mm@1m"),
    ("h00", 0, "u→x. 순수 전단. +면 x 가 줄어든다. 1칸 -13mm@1m"),
    ("h20", 6, "u 원근항. 좌우 비대칭 원근. 1칸 +6.6mm@1m, +25mm@2m (원거리 증폭 최대)"),
    ("h22", 8, "동차항. 1.0 고정 권장(게이지). 단독 변경은 no-op 아님 — 1칸 +4.1mm@1m +16mm@2m"),
]

# 지면 격자 오버레이 범위 (m)
GRID_X = (0.0, 3.2)
GRID_Y = (-1.6, 1.6)


# ============================ 한글 텍스트 렌더 ============================
# cv2.putText 는 ASCII 만 그린다 (한글이 전부 '?' 로 깨진다). PIL + CJK 폰트로 그린다.
_FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",   # 차 PC (Ubuntu)
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/mnt/c/Windows/Fonts/malgun.ttf",      # WSL — 윈도우 맑은 고딕
    "/mnt/c/Windows/Fonts/gulim.ttc",
    "C:/Windows/Fonts/malgun.ttf",          # 윈도우 네이티브 파이썬
]


class Text:
    """한글 가능한 폰트를 찾아 캐시. 못 찾으면 cv2.putText 로 폴백."""

    def __init__(self):
        self.ok = False
        self._cache = {}
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return
        for path in _FONTS:
            if not os.path.isfile(path):
                continue
            for idx in range(6):
                try:
                    f = ImageFont.truetype(path, 16, index=idx)
                except (OSError, ValueError, IndexError):
                    break
                # 한글 글리프가 실제로 있는지: '한글' 과 사용자영역 문자를 비교
                def ink(s):
                    im = Image.new("L", (60, 24), 0)
                    ImageDraw.Draw(im).text((1, 1), s, font=f, fill=255)
                    return np.asarray(im, np.uint8)
                a, b = ink("한글"), ink("")
                if a.sum() > 200 and np.abs(a.astype(int) - b.astype(int)).mean() > 4:
                    self.path, self.index, self.ok = path, idx, True
                    break
            if self.ok:
                break

    def _font(self, size):
        f = self._cache.get(size)
        if f is None:
            from PIL import ImageFont
            f = ImageFont.truetype(self.path, size, index=self.index)
            self._cache[size] = f
        return f

    def draw(self, img, items):
        """items = [(문자열, x, y_top, 크기px, (B,G,R))] — img 를 제자리에서 수정."""
        if not items:
            return img
        if not self.ok:
            for t, x, y, s, c in items:
                cv2.putText(img, t, (int(x), int(y + s)), cv2.FONT_HERSHEY_SIMPLEX,
                            s / 30.0, c, 1, cv2.LINE_AA)
            return img
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for t, x, y, s, c in items:
            d.text((int(x), int(y)), str(t), font=self._font(int(s)),
                   fill=(int(c[2]), int(c[1]), int(c[0])))
        img[:, :, :] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        return img


TXT = Text()


# ============================ H ↔ 물리 파라미터 ============================

def build_H(Kn, height, pitch_deg, yaw_deg, roll_deg, cam_x, cam_y):
    """물리 파라미터 → H (보정 이미지 픽셀 → 지면 미터, x전방/y좌).

    지면축(x전방,y좌,z상) → 카메라축(x우,y하,z전) 기본회전 R0 에
    pitch(카메라 x축), yaw(지면 z축), roll(광축) 을 얹는다.
    """
    th, ps, rl = np.radians([pitch_deg, yaw_deg, roll_deg])
    R0 = np.array([[0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [1.0, 0.0, 0.0]])
    Rp = np.array([[1, 0, 0],
                   [0, np.cos(th), -np.sin(th)],
                   [0, np.sin(th), np.cos(th)]])
    Ry = np.array([[np.cos(ps), -np.sin(ps), 0],
                   [np.sin(ps), np.cos(ps), 0],
                   [0, 0, 1]])
    Rr = np.array([[np.cos(rl), -np.sin(rl), 0],
                   [np.sin(rl), np.cos(rl), 0],
                   [0, 0, 1]])
    R = Rr @ Rp @ R0 @ Ry
    t = -R @ np.array([cam_x, cam_y, height], dtype=np.float64)
    Hg2i = np.asarray(Kn, np.float64) @ np.column_stack([R[:, 0], R[:, 1], t])
    return np.linalg.inv(Hg2i)


def fit_phys_to_H(H_target, Kn, x0=None):
    """현재 H 에 가장 가까운 물리 6파라미터를 찾는다 (시작값 용도).

    반환 (params_dict, 상대오차). scipy 없으면 (기본값, None).
    """
    default = dict(height=0.18, pitch=15.0, yaw=0.0, roll=0.0, cam_x=0.0, cam_y=0.0)
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return default, None

    Ht = np.asarray(H_target, np.float64)
    Ht = Ht / np.linalg.norm(Ht)

    def res(p):
        try:
            Hm = build_H(Kn, p[0], p[1], p[2], p[3], p[4], p[5])
        except np.linalg.LinAlgError:
            return np.full(9, 1e3)
        n = np.linalg.norm(Hm)
        if not np.isfinite(n) or n < 1e-12:
            return np.full(9, 1e3)
        Hm = Hm / n
        # 동차행렬은 부호 자유도가 있다 — 가까운 쪽으로
        a = (Hm - Ht).ravel()
        b = (-Hm - Ht).ravel()
        return a if np.sum(a * a) <= np.sum(b * b) else b

    best, best_cost = None, np.inf
    seeds = [x0] if x0 is not None else []
    seeds += [[0.18, 15.0, 0.0, 0.0, 0.0, 0.0],
              [0.18, 25.0, 0.0, 0.0, 0.0, 0.0],
              [0.25, 10.0, 0.0, 0.0, 0.0, 0.0],
              [0.15, 20.0, 5.0, 0.0, 0.0, 0.0],
              [0.15, 20.0, -5.0, 0.0, 0.0, 0.0]]
    lo = [0.02, -40.0, -60.0, -30.0, -2.0, -2.0]
    hi = [1.50, 80.0, 60.0, 30.0, 2.0, 2.0]
    for s in seeds:
        s = [float(np.clip(v, lo[i] + 1e-6, hi[i] - 1e-6)) for i, v in enumerate(s)]
        try:
            sol = least_squares(res, s, bounds=(lo, hi), max_nfev=4000)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if sol.cost < best_cost:
            best_cost, best = sol.cost, sol.x
    if best is None:
        return default, None
    return (dict(height=float(best[0]), pitch=float(best[1]), yaw=float(best[2]),
                 roll=float(best[3]), cam_x=float(best[4]), cam_y=float(best[5])),
            float(np.sqrt(2.0 * best_cost)))


# ============================ 표시 보조 ============================

def enhance(img):
    """대비 강화 (표시 전용, 기하 불변)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 3)
    return cv2.addWeighted(out, 1.6, blur, -0.6, 0)


def darkmap(bgr):
    """어두운 선일수록 값이 큰 맵 (자로 재기용)."""
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32),
                         (0, 0), 0.9)
    return cv2.GaussianBlur(g, (0, 0), 21) - g


# ============================ H 정확도 (실측 대응점 기준) ============================

def distort_norm(xn, yn, D):
    """정규화 이상좌표 → 왜곡좌표 (plumb_bob 정방향). D=[k1,k2,p1,p2,k3]."""
    k1, k2, p1, p2, k3 = [float(t) for t in np.ravel(D)[:5]]
    r2 = xn * xn + yn * yn
    rad = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = xn * rad + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
    yd = yn * rad + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
    return xd, yd


def h_points_to_raw(pts, K_ref, D_ref, nK_ref):
    """보정좌표 기준 h_points → **원본(raw) 픽셀** 기준으로 되돌린다.

    h_points 는 캘리브 당시의 K/D/new_K 로 보정한 픽셀 좌표다. 그 좌표계에
    묶여 있으면 K/D 를 갈아끼울 때(체스보드 캘리브 등) 대응이 통째로 깨진다.
    원본 픽셀은 어떤 캘리브에도 의존하지 않으므로 이쪽이 진실원이다.
    반환 [[u_raw, v_raw, x, y], ...]
    """
    out = []
    if not pts:
        return out
    nKi = np.linalg.inv(np.asarray(nK_ref, np.float64))
    K_ref = np.asarray(K_ref, np.float64)
    for row in pts:
        try:
            u, v, x, y = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        except (TypeError, ValueError, IndexError):
            continue
        n = nKi @ np.array([u, v, 1.0])
        xd, yd = distort_norm(n[0] / n[2], n[1] / n[2], D_ref)
        p = K_ref @ np.array([xd, yd, 1.0])
        out.append([float(p[0] / p[2]), float(p[1] / p[2]), x, y])
    return out


def raw_to_h_points(raw_pts, K_cur, D_cur, nK):
    """원본 픽셀 기준 대응점 → 현재 K/D/new_K 로 보정한 좌표. H 평가/적합에 쓴다."""
    if not raw_pts:
        return []
    src = np.asarray([[p[0], p[1]] for p in raw_pts], np.float64).reshape(-1, 1, 2)
    try:
        und = cv2.undistortPoints(src, np.asarray(K_cur, np.float64),
                                  np.ravel(D_cur)[:5].astype(np.float64),
                                  P=np.asarray(nK, np.float64)).reshape(-1, 2)
    except cv2.error:
        return []
    return [[float(u), float(v), float(p[2]), float(p[3])]
            for (u, v), p in zip(und, raw_pts)]


def h_metric(Hm, pts):
    """대응점 기준 지면좌표 재투영 오차 (rms_m, max_m, n). 점 없으면 (None,None,0)."""
    if not pts:
        return None, None, 0
    Hm = np.asarray(Hm, np.float64)
    e = []
    for u, v, x, y in pts:
        p = Hm @ np.array([u, v, 1.0])
        if abs(p[2]) < 1e-12:
            return None, None, len(pts)
        e.append(float(np.hypot(p[0] / p[2] - x, p[1] / p[2] - y)))
    if not e:
        return None, None, 0
    e = np.asarray(e)
    return float(np.sqrt((e ** 2).mean())), float(e.max()), len(e)


# ============ 화면 전체 직선성 점수 (원본에서 테이프/줄눈을 촘촘히 추적) ============
# 왜 필요한가: 실측 대응점 11개는 앞 0.35~1.2m 에만 몰려 있어서 화면 전체를 제약하지
# 못한다. 그 좁은 구역만 좋아지고 가장자리·원거리는 나빠지는 K/D 가 "최적" 으로
# 보이는 함정이 있다. 여기서는 원본 영상에서 어두운 선을 한 번 추적해 고정해두고,
# K/D 를 바꿀 때마다 그 점들을 undistort 해서 **얼마나 곧아지는지** 만 본다.
#   · 원본 픽셀 검출은 K/D 와 무관 → 순환논리 없음
#   · H 를 전혀 안 씀 → 왜곡만의 성적
#   · 화면 전체를 덮음 → 눈으로 보는 것과 같은 범위

def _peaks(prof, lo, thr):
    out = []
    for k in range(1, len(prof) - 1):
        if prof[k] >= prof[k - 1] and prof[k] > prof[k + 1] and prof[k] >= thr:
            a, b, c = prof[k - 1], prof[k], prof[k + 1]
            den = a - 2 * b + c
            d = 0.5 * (a - c) / den if abs(den) > 1e-6 else 0.0
            out.append(lo + k + max(-1.0, min(1.0, d)))
    return out


def _trace(dk, seed_pos, seed_cross, mode, win=7, thr=10.0, gate=2.0, maxmiss=8):
    """어두운 띠를 따라간다. 예측값에 가장 가까운 봉우리만 받아 hop 을 막는다.
    mode='h': u 를 따라 v 추적 / 'v': v 를 따라 u 추적."""
    Hh, Ww = dk.shape[:2]
    LIM = Hh if mode == "h" else Ww
    END = Ww if mode == "h" else Hh
    pts = []
    for d in (1, -1):
        hist = [(float(seed_pos), float(seed_cross))]
        pos, miss = int(seed_pos), 0
        while True:
            pos += d
            if not (2 <= pos < END - 2):
                break
            if len(hist) < 3:
                pred = hist[-1][1]
            else:
                h = np.asarray(hist[-40:])
                pred = float(np.polyval(
                    np.polyfit(h[:, 0], h[:, 1], 2 if len(h) >= 8 else 1), pos))
            lo = int(max(0, pred - win)); hi = int(min(LIM - 1, pred + win))
            if hi - lo < 4:
                break
            prof = (dk[lo:hi + 1, pos - 1:pos + 2].mean(axis=1) if mode == "h"
                    else dk[pos - 1:pos + 2, lo:hi + 1].mean(axis=0))
            cand = _peaks(prof, lo, thr)
            best = min(cand, key=lambda c: abs(c - pred)) if cand else None
            if best is None or abs(best - pred) > gate:
                miss += 1
                if miss > maxmiss:
                    break
                continue
            miss = 0
            hist.append((float(pos), best))
            pts.append((float(pos), best) if mode == "h" else (best, float(pos)))
    if not pts:
        return np.zeros((0, 2))
    P = np.asarray(pts)
    return P[np.argsort(P[:, 0 if mode == "h" else 1])]


def detect_lines(bgr, min_pts=100, min_span=220):
    """원본(왜곡 있는) 영상에서 어두운 직선들을 추적한다. [(pts Nx2, mode)] 반환."""
    dk = darkmap(bgr)
    Hh, Ww = dk.shape[:2]
    kept = []

    def dup(P, mode):
        ax, ay = (0, 1) if mode == "h" else (1, 0)
        pi = {int(r[ax]): r[ay] for r in P}
        for Q, m2 in kept:
            if m2 != mode:
                continue
            qi = {int(r[ax]): r[ay] for r in Q}
            common = set(pi) & set(qi)
            if len(common) < 20:
                continue
            if np.median([abs(pi[c] - qi[c]) for c in common]) < 3.0:
                return True
        return False

    for col in (int(Ww * 0.3), int(Ww * 0.5), int(Ww * 0.7)):
        prof = dk[:, col - 12:col + 13].mean(axis=1)
        for v in _peaks(prof, 0, 10.0):
            if v < Hh * 0.45:                 # 지평선 위(벽/천장)는 제외
                continue
            P = _trace(dk, col, v, "h")
            if len(P) >= min_pts and P[:, 0].ptp() > min_span and not dup(P, "h"):
                kept.append((P, "h"))
    for row in (int(Hh * 0.62), int(Hh * 0.78), int(Hh * 0.92)):
        prof = dk[row - 12:row + 13, :].mean(axis=0)
        for u in _peaks(prof, 0, 10.0):
            P = _trace(dk, u, row, "v")
            if len(P) >= 70 and P[:, 1].ptp() > 80 and not dup(P, "v"):
                kept.append((P, "v"))
    return kept


def straightness(lines, K, D, nK):
    """추적된 원본 점들을 현재 K/D 로 보정한 뒤 직선에서 벗어난 정도 (px).

    반환 (rms, max, 선개수, 점개수, 선별 rms 리스트).
    """
    if not lines:
        return None, None, 0, 0, []
    res, per = [], []
    for P, _m in lines:
        try:
            u = cv2.undistortPoints(P.reshape(-1, 1, 2).astype(np.float64),
                                    np.asarray(K, np.float64),
                                    np.ravel(D)[:5].astype(np.float64),
                                    P=np.asarray(nK, np.float64)).reshape(-1, 2)
        except cv2.error:
            return None, None, 0, 0, []
        c = u - u.mean(0)
        _s, _sv, Vt = np.linalg.svd(c, full_matrices=False)
        r = c @ Vt[1]                       # 최적 직선에 대한 수직 잔차
        res.append(r)
        per.append(float(np.sqrt((r ** 2).mean())))
    r = np.concatenate(res)
    return (float(np.sqrt((r ** 2).mean())), float(np.abs(r).max()),
            len(lines), int(r.size), per)


def measure_segment(dk, p0, p1, halfwin=30.0, thr=8.0):
    """사용자가 찍은 두 점 사이의 어두운 선을 훑어 '직선에서 벗어난 정도'를 잰다.

    현(chord)이 아니라 **최적 직선** 기준 잔차를 쓴다 (클릭이 선 끝에서 조금
    빗나가도 결과가 안 흔들리게).
    선 방향으로 1px 간격, 수직으로 ±halfwin 을 0.5px 간격으로 한 번에 remap 해서
    (파이썬 루프 없이) 훑는다 — 값을 돌릴 때마다 다시 재므로 속도가 중요하다.

    ▲ halfwin 이 작으면 크게 휜 선이 창을 벗어나 버려지고, 남은 구간만 재서
    숫자가 오히려 작아진다(halfwin=9 일 때 휨 20px 이 4.3px 로 보고됨 — 거짓 안심).
    그래서 기본값을 30 으로 크게 두고, **커버율**(창 안에서 잡힌 비율)을 함께
    반환해 그 함정을 눈에 보이게 한다. 커버율이 낮으면 숫자를 믿으면 안 된다.
    비스듬히 교차하는 어두운 선(줄눈/테이프)도 값을 오염시키므로, 추적 오프셋의
    변동폭(dd_span)도 같이 준다 — 이게 halfwin 에 가까우면 다른 선에 끌려간 것이다.

    반환 (최대편차 px, RMS px, 곡선점들, 커버율 0~1, dd_span px) 또는 None.
    """
    p0 = np.asarray(p0, np.float64); p1 = np.asarray(p1, np.float64)
    L = float(np.linalg.norm(p1 - p0))
    if L < 25.0:
        return None
    t = (p1 - p0) / L
    nrm = np.array([-t[1], t[0]])
    S = np.arange(0.0, L + 0.5, 1.0)                       # 선 방향
    O = np.arange(-halfwin, halfwin + 1e-9, 0.5)           # 수직 방향
    base = p0[None, :] + t[None, :] * S[:, None]           # (Ns,2)
    grid = base[:, None, :] + nrm[None, None, :] * O[None, :, None]   # (Ns,No,2)
    prof = cv2.remap(np.ascontiguousarray(dk, np.float32),
                     np.ascontiguousarray(grid[..., 0], np.float32),
                     np.ascontiguousarray(grid[..., 1], np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=-1e9)                     # (Ns,No)
    k = np.argmax(prof, axis=1)
    rows = np.arange(len(S))
    peak = prof[rows, k]
    ok = (k > 0) & (k < len(O) - 1) & (peak >= thr)
    if int(ok.sum()) < 20:
        return None
    r, kk = rows[ok], k[ok]
    a, b, c = prof[r, kk - 1], prof[r, kk], prof[r, kk + 1]
    den = a - 2 * b + c
    sub = np.where(np.abs(den) > 1e-6, 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
    dd = O[kk] + np.clip(sub, -1.0, 1.0) * 0.5
    ss = S[ok]
    resid = dd - np.polyval(np.polyfit(ss, dd, 1), ss)     # 최적 직선 대비 잔차
    pts = base[ok] + nrm[None, :] * dd[:, None]
    cover = float(ok.sum()) / float(len(S))
    dd_span = float(dd.max() - dd.min())
    return (float(np.abs(resid).max()), float(np.sqrt((resid ** 2).mean())),
            pts, cover, dd_span)


# ============================ 튜너 본체 ============================

class Tuner:
    def __init__(self, args):
        self.args = args
        self.cam = B.load_camera_config(args.camera)

        # 격자 원점: H 의 원점은 h_frame(=라이다 지면점)인데 바닥 줄눈은 A'(카메라
        # 지면점) 기준으로 그었다. 둘의 차이(h_lidar_offset_x)만큼 격자를 뒤로 민다.
        raw_yaml = {}
        if yaml is not None:
            try:
                with open(args.camera, "r", encoding="utf-8") as f:
                    raw_yaml = yaml.safe_load(f) or {}
            except (OSError, yaml.YAMLError):
                raw_yaml = {}
        if args.grid_x0 is None:
            try:
                args.grid_x0 = -float(raw_yaml.get("h_lidar_offset_x") or 0.0)
            except (TypeError, ValueError):
                args.grid_x0 = 0.0
        if args.grid_y0 is None:
            args.grid_y0 = 0.0
        print("격자 원점: x0=%+.3f m, y0=%+.3f m, 피치 %.4f m "
              "(x0 는 A'↔라이다 거리만큼 뒤로 민 값 — --grid-x0 로 바꿀 수 있다)"
              % (args.grid_x0, args.grid_y0, args.pitch))

        K0 = np.asarray(self.cam["K"], np.float64)
        D0 = np.asarray(B._coerce_D(self.cam["D"], 5)[:5], np.float64)

        # ---- 초기 프레임 ----
        self.live = bool(args.live)
        self.shots = list(args.image) if args.image else []
        self.shot_i = 0
        self._lock = threading.Lock()
        self._live_frame = None
        if self.live:
            self._start_ros()
            self.frame = self._wait_live()
        else:
            if not self.shots:
                d = os.path.join(_PKG, "dev", "shots")
                for cand in ("tape4_raw.jpg", "tape3_raw.jpg", "tape2_raw.jpg",
                             "tape_raw.jpg", "calib_raw.jpg"):
                    p = os.path.join(d, cand)
                    if os.path.isfile(p):
                        self.shots.append(p)
                if not self.shots:
                    print("사진이 없다. --image <경로> 로 지정하거나 --live 를 쓸 것.")
                    sys.exit(1)
            self.frame = self._load_shot()

        h, w = self.frame.shape[:2]
        self.size = (w, h)

        # ---- 왜곡 파라미터 ----
        self.dist0 = dict(fx=K0[0, 0], fy=K0[1, 1], cx=K0[0, 2], cy=K0[1, 2],
                          k1=D0[0], k2=D0[1], p1=D0[2], p2=D0[3], k3=D0[4])
        self.dist = dict(self.dist0)

        # ---- new_K 를 여기서 한 번 고정 (설명은 파일 상단 참조) ----
        nk = self.cam.get("new_K")
        if nk is not None:
            self.new_K = np.asarray(nk, np.float64).reshape(3, 3)
            self.newk_src = "camera.yaml"
        else:
            self.new_K = np.asarray(cv2.getOptimalNewCameraMatrix(
                self.K(), self.D(), self.size, args.alpha)[0], np.float64).reshape(3, 3)
            self.newk_src = "계산(alpha=%.2f)" % args.alpha

        # ---- H 파라미터 ----
        # ★ 기본은 반드시 행렬(elem) 모드다. 물리 6개는 8자유도 H 를 표현할 수 없어서,
        #   phys 로 기동하면 아무것도 안 건드리고 저장만 해도 H 가 열화된다
        #   (실측: 대응점 RMS 6.1mm → 200mm). phys 는 [m] 로 명시적으로 들어갈 때만.
        self.H_yaml = np.asarray(self.cam["H"], np.float64)
        self.h_mode = "elem"
        self.phys, self.fiterr = None, None
        self.phys0 = None
        self.helem = list(self.H_yaml.ravel())
        self.helem0 = list(self.helem)
        # 행렬 원소 모드의 스텝: 열 단위 크기에 맞춰 자동 설정
        col = np.abs(self.H_yaml).max(axis=0)
        self.helem_step = [max(1e-7, 0.002 * col[j]) for _ in range(3) for j in range(3)]

        # ---- 실측 대응점 (H 정확도를 mm 로 보여주는 근거) ----
        # ★ 원본(raw) 픽셀 기준으로 보관한다. 보정좌표로 들고 있으면 K/D 를 갈아끼울 때
        #   (체스보드 캘리브 등) 대응이 통째로 깨진다. 원본 픽셀은 캘리브 무관이다.
        def _rows(key):
            out = []
            for row in (raw_yaml.get(key) or []):
                try:
                    out.append([float(row[0]), float(row[1]),
                                float(row[2]), float(row[3])])
                except (TypeError, ValueError, IndexError):
                    pass
            return out

        self.h_raw = _rows("h_points_raw")
        if self.h_raw:
            self.h_src_note = "h_points_raw (원본픽셀 기준 — K/D 변경에 안전)"
        else:
            und_pts = _rows("h_points")
            nk_ref = raw_yaml.get("h_points_new_K") or self.new_K
            self.h_raw = h_points_to_raw(und_pts, K0, D0,
                                         np.asarray(nk_ref, np.float64).reshape(3, 3))
            self.h_src_note = ("h_points (yaml 의 K/D 로 원본픽셀로 환산)"
                               if und_pts else "없음")
        self.h_pts_ref = self.h_raw     # 호환용 별칭
        self._hp_cache = (None, None)   # (dist 스냅샷, 보정좌표로 옮긴 점들)

        # ---- 화면 전체 직선성 (원본에서 한 번 추적해 고정) ----
        self.lines = detect_lines(self.frame)
        self._st_cache = (None, None)
        if self.lines:
            r, m, nl, npt, _ = straightness(self.lines, K0, D0, self.new_K)
            print("직선 %d개 / 점 %d개 추적 — 지금 K/D 의 직선성 RMS %.2f px / 최대 %.2f px"
                  % (nl, npt, r, m))
        else:
            print("▲ 원본에서 직선을 못 찾았다 — 화면 전체 직선성 점수는 표시 안 됨 "
                  "([t] 로 재시도, 또는 --enhance 로 대비 올려서 확인)")
        if self.h_raw:
            # 원본픽셀 → 현재 K/D 로 보정한 좌표에서 평가해야 맞다
            r, m, n = h_metric(self.H_yaml,
                               raw_to_h_points(self.h_raw, K0, D0, self.new_K))
            if r is not None:
                print("실측 대응점 %d개 [%s] — yaml H 의 오차 RMS %.1f mm / 최대 %.1f mm"
                      % (n, self.h_src_note, 1000 * r, 1000 * m))
            else:
                print("실측 대응점 %d개 — H 평가 실패" % n)
        else:
            print("▲ camera.yaml 에 h_points 가 없다 — H 정확도를 숫자로 못 보여준다 "
                  "(격자 육안 판정만 가능)")

        # ---- UI 상태 ----
        self.group = "dist"          # dist | h
        self.sel = 0
        self.stepmul = 1.0
        self.show_grid = True
        self.do_enh = bool(args.enhance)
        self.line_mode = False
        self.pending = None
        self.segs = []               # [(p0, p1)] 보정영상 좌표
        self.msg = ""
        self.msg_t = 0.0
        self.dirty = True
        self.ui_dirty = True
        self.und_off = (0, 0)
        self.und_scale = 1.0
        self._cache = None
        self._panel = None
        self._msg_on = False
        self._dk = None

    # ---------- ROS 라이브 ----------
    def _start_ros(self):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from track_drive.lib import preprocess as PP

        rclpy.init()
        node = Node("calib_tuner")

        def cb(msg):
            bgr, _ = PP.image_msg_to_bgr(msg)
            if bgr is not None:
                with self._lock:
                    self._live_frame = bgr
        node.create_subscription(Image, self.args.topic, cb, qos_profile_sensor_data)
        self._ros_node = node
        self._ros = rclpy
        t = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
        t.start()
        print("구독 %s — 첫 프레임 대기…" % self.args.topic)

    def _wait_live(self, timeout=15.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                f = self._live_frame
            if f is not None:
                return f.copy()
            time.sleep(0.05)
        print("이미지 토픽 %s 가 안 온다. 카메라부터 켤 것." % self.args.topic)
        sys.exit(1)

    def _load_shot(self):
        p = self.shots[self.shot_i % len(self.shots)]
        img = cv2.imread(p)
        if img is None:
            print("읽기 실패:", p); sys.exit(1)
        self.shot_name = os.path.basename(p)
        return img

    # ---------- 현재 값 ----------
    def K(self):
        d = self.dist
        return np.array([[d["fx"], 0.0, d["cx"]],
                         [0.0, d["fy"], d["cy"]],
                         [0.0, 0.0, 1.0]], np.float64)

    def D(self):
        d = self.dist
        return np.array([d["k1"], d["k2"], d["p1"], d["p2"], d["k3"]], np.float64)

    def H(self):
        if self.h_mode == "phys" and self.phys is not None:
            p = self.phys
            try:
                return build_H(self.new_K, p["height"], p["pitch"], p["yaw"],
                               p["roll"], p["cam_x"], p["cam_y"])
            except np.linalg.LinAlgError:
                return self.H_yaml.copy()
        return np.array(self.helem, np.float64).reshape(3, 3)

    def dist_changed(self):
        """K/D 가 파일에서 읽은 값과 달라졌는가 (달라지면 H 는 재조정 대상)."""
        return any(abs(self.dist[k] - self.dist0[k]) > 1e-12 for k in self.dist0)

    def h_points_now(self):
        """원본픽셀 대응점을 현재 K/D 로 보정한 좌표. K/D 가 바뀌면 자동으로 따라온다."""
        key = tuple(round(self.dist[k], 12) for k in sorted(self.dist))
        if self._hp_cache[0] == key:
            return self._hp_cache[1]
        try:
            pts = raw_to_h_points(self.h_raw, self.K(), self.D(), self.new_K)
        except (cv2.error, np.linalg.LinAlgError, ValueError):
            pts = []
        self._hp_cache = (key, pts)
        return pts

    def h_accuracy(self):
        """현재 H 의 실측 대응점 재투영 오차 (rms_m, max_m, n)."""
        return h_metric(self.H(), self.h_points_now())

    def straight_score(self):
        """화면 전체 직선성 (rms_px, max_px, 선수, 점수, 선별rms). H 와 무관."""
        key = tuple(round(self.dist[k], 12) for k in sorted(self.dist))
        if self._st_cache[0] == key:
            return self._st_cache[1]
        out = straightness(self.lines, self.K(), self.D(), self.new_K)
        self._st_cache = (key, out)
        return out

    def fit_H(self):
        """현재 K/D 기준으로 H 를 실측 대응점에 최소자승 적합한다.

        ★ H 는 손으로 맞출 대상이 아니다. K/D 가 정해지면 실측점 4개 이상으로
          8자유도 H 의 최적해가 결정되므로, 눈대중은 항상 이보다 나쁘다.
          왜곡(K/D)만 손으로 잡고, H 는 이 함수에 맡기는 것이 맞는 분업이다.
        """
        pts = self.h_points_now()
        if len(pts) < 4:
            return None, "실측 대응점이 %d개뿐 — 최소 4개 필요" % len(pts)
        src = np.asarray([[p[0], p[1]] for p in pts], np.float64)
        dst = np.asarray([[p[2], p[3]] for p in pts], np.float64)
        try:
            Hm, _ = cv2.findHomography(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), 0)
        except cv2.error as e:
            return None, "적합 실패: %s" % e
        if Hm is None:
            return None, "적합 실패 (점 배치가 퇴화 — 한 줄에 몰렸는지 확인)"
        if abs(Hm[2, 2]) > 1e-12:
            Hm = Hm / Hm[2, 2]          # h22=1 로 정규화 (읽기 쉽게, 결과 동일)
        return Hm, None

    def params(self):
        """(이름, 설명, 스텝, 현재값) 목록. 행렬 모드는 화면 순서 != 저장 순서라
        저장 인덱스를 _slot() 으로 따로 얻는다."""
        if self.group == "dist":
            return [(n, doc, st, self.dist[n]) for n, doc, st in DIST_PARAMS]
        if self.h_mode == "phys":
            return [(n, doc, st, self.phys[n]) for n, doc, st in PHYS_PARAMS]
        return [(n, doc, self.helem_step[j], self.helem[j])
                for n, j, doc in H_ELEM_DOC]

    def _slot(self):
        """행렬 모드에서 선택된 줄이 H 의 몇 번째 원소인지 (행*3+열)."""
        return H_ELEM_DOC[self.sel][1]

    def bump(self, sign):
        n, _, st, _ = self.params()[self.sel]
        delta = sign * st * self.stepmul
        if self.group == "dist":
            self.dist[n] += delta
        elif self.h_mode == "phys":
            self.phys[n] += delta
        else:
            self.helem[self._slot()] += delta
        self.dirty = True

    def reset_sel(self):
        n = self.params()[self.sel][0]
        if self.group == "dist":
            self.dist[n] = self.dist0[n]
        elif self.h_mode == "phys":
            self.phys[n] = self.phys0[n]
        else:
            j = self._slot()
            self.helem[j] = self.helem0[j]
        self.dirty = True

    def reset_all(self):
        self.dist = dict(self.dist0)
        self.phys = dict(self.phys0)
        self.helem = list(self.helem0)
        self.dirty = True
        self.note("전부 초기값으로 되돌림")

    def note(self, s):
        self.msg, self.msg_t = s, time.time()
        self.ui_dirty = True
        print("  " + s)

    # ---------- 렌더 ----------
    def undistort(self):
        try:
            m1, m2 = cv2.initUndistortRectifyMap(
                self.K(), self.D(), np.eye(3), self.new_K, self.size, cv2.CV_16SC2)
            return cv2.remap(self.frame, m1, m2, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
        except cv2.error:
            return self.frame.copy()

    def bev(self, src, Hm):
        x0, x1 = self.args.range_x
        y0, y1 = self.args.range_y
        ppm = self.args.ppm
        W = max(2, int(round((y1 - y0) * ppm)))
        Hh = max(2, int(round((x1 - x0) * ppm)))
        ys = y1 - np.arange(W, dtype=np.float64) / ppm
        xs = x1 - np.arange(Hh, dtype=np.float64) / ppm
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        G = np.stack([XX, YY, np.ones_like(XX)], -1).reshape(-1, 3).T
        try:
            Hi = np.linalg.inv(Hm)
        except np.linalg.LinAlgError:
            return np.zeros((Hh, W, 3), np.uint8)
        P = Hi @ G
        wref = np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2]) or 1.0
        w = P[2]
        bad = (np.abs(w) < 1e-12) | (np.sign(w) != wref)
        w = np.where(bad, 1.0, w)
        mu = (P[0] / w).reshape(Hh, W).astype(np.float32)
        mv = (P[1] / w).reshape(Hh, W).astype(np.float32)
        mu[bad.reshape(Hh, W)] = -1
        mv[bad.reshape(Hh, W)] = -1
        return cv2.remap(src, mu, mv, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def draw_bev_grid(self, img):
        x0, x1 = self.args.range_x
        y0, y1 = self.args.range_y
        ppm = self.args.ppm
        p = self.args.pitch
        gx, gy = self.args.grid_x0, self.args.grid_y0   # 줄눈 원점 (보정영상 격자와 동일)
        k = int(np.ceil((x0 - gx) / p))
        while gx + k * p <= x1:
            r = int(round((x1 - (gx + k * p)) * ppm))
            for cx in range(0, img.shape[1], 14):
                cv2.line(img, (cx, r), (min(cx + 7, img.shape[1] - 1), r), (0, 255, 255), 1)
            k += 1
        k = int(np.ceil((y0 - gy) / p))
        while gy + k * p <= y1:
            c = int(round((y1 - (gy + k * p)) * ppm))
            for cy in range(0, img.shape[0], 14):
                cv2.line(img, (c, cy), (c, min(cy + 7, img.shape[0] - 1)), (0, 255, 255), 1)
            k += 1
        d = np.ceil(x0 / 0.5) * 0.5
        while d <= x1:
            r = int(round((x1 - d) * ppm))
            cv2.line(img, (0, r), (img.shape[1] - 1, r), (0, 190, 0), 1)
            cv2.putText(img, "x=%.1f" % d, (3, r - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (0, 255, 0), 1)
            d += 0.5
        d = np.ceil(y0 / 0.5) * 0.5
        while d <= y1:
            c = int(round((y1 - d) * ppm))
            cv2.line(img, (c, 0), (c, img.shape[0] - 1), (0, 190, 0), 1)
            cv2.putText(img, "y=%.1f" % d, (c + 3, img.shape[0] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
            d += 0.5

    def draw_ground_grid(self, img, Hm):
        """현재 H 로 지면 격자를 보정영상에 되쏜다 — 줄눈 위에 얹히면 H 가 맞다."""
        try:
            Hi = np.linalg.inv(Hm)
        except np.linalg.LinAlgError:
            return
        wref = np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2]) or 1.0
        p = self.args.pitch
        h, w = img.shape[:2]

        def proj(pts):
            G = np.column_stack([pts, np.ones(len(pts))]).T
            P = Hi @ G
            ok = (np.abs(P[2]) > 1e-12) & (np.sign(P[2]) == wref)
            u = np.where(ok, P[0] / np.where(ok, P[2], 1.0), np.nan)
            v = np.where(ok, P[1] / np.where(ok, P[2], 1.0), np.nan)
            return u, v

        def polyline(pts, color, thick=1):
            # 점마다 cv2.line 을 부르면 한 프레임에 수천 번이라 창이 버벅인다.
            # 유효 구간을 통째로 끊어 polylines 로 한 번에 그린다.
            u, v = proj(pts)
            ok = (np.isfinite(u) & np.isfinite(v)
                  & (np.abs(u) < 4000) & (np.abs(v) < 4000))
            idx = np.flatnonzero(ok)
            if idx.size < 2:
                return
            runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
            polys = [np.stack([u[r], v[r]], 1).astype(np.int32)
                     for r in runs if r.size >= 2]
            if polys:
                cv2.polylines(img, polys, False, color, thick, cv2.LINE_AA)

        # 격자 원점 오프셋: H 의 원점은 라이다 지면점인데 줄눈은 A'(카메라 지면점)
        # 에 맞춰 그었다. 라이다가 A' 보다 앞서 있는 만큼 격자를 뒤로 밀어야
        # 실제 줄눈과 겹친다 (안 그러면 H 가 맞아도 어긋나 보인다).
        x0, y0 = self.args.grid_x0, self.args.grid_y0
        ys = np.linspace(GRID_Y[0], GRID_Y[1], 160)
        xs = np.linspace(GRID_X[0], GRID_X[1], 160)
        k = int(np.ceil((GRID_X[0] - x0) / p))
        while x0 + k * p <= GRID_X[1]:                # 가로선 (전방 x)
            x = x0 + k * p
            polyline(np.column_stack([np.full_like(ys, x), ys]),
                     (0, 230, 0), 2 if k % 2 == 0 else 1)
            k += 1
        m = int(np.floor((GRID_Y[1] - y0) / p))
        for k in range(-m, m + 1):                    # 세로선 (좌우 y)
            y = y0 + k * p
            polyline(np.column_stack([xs, np.full_like(xs, y)]),
                     (255, 190, 0) if k == 0 else (0, 230, 0), 2 if k == 0 else 1)

        # 실측 대응점: 자홍 = 실제 클릭 위치(정답), 그 옆 흰 선 = 현재 H 의 예측 위치.
        # 두 점이 붙어 있으면 H 가 맞다 — 격자 육안 판정보다 훨씬 예민하다.
        pts = self.h_points_now()
        if pts:
            for u, v, x, y in pts:
                gu, gv = proj(np.array([[x, y]]))
                cv2.drawMarker(img, (int(round(u)), int(round(v))), (255, 0, 255),
                               cv2.MARKER_CROSS, 9, 1)
                if np.isfinite(gu[0]) and np.isfinite(gv[0]) and abs(gu[0]) < 4000:
                    cv2.line(img, (int(round(u)), int(round(v))),
                             (int(round(gu[0])), int(round(gv[0]))),
                             (255, 255, 255), 1, cv2.LINE_AA)

    def hud(self, width, meas):
        rows = self.params()
        lh = 19
        H = 46 + lh * len(rows) + 112
        img = np.full((H, width, 3), 24, np.uint8)
        it = []

        gtxt = "왜곡 K/D" if self.group == "dist" else \
               ("지면 H — 물리 6개" if self.h_mode == "phys" else "지면 H — 행렬 원소 9개")
        it.append(("[TAB] 그룹: %s      스텝 x%.4g      %s"
                   % (gtxt, self.stepmul, "LIVE" if self.live else self.shot_name),
                   8, 5, 15, (120, 220, 255)))

        for i, (n, doc, st, val) in enumerate(rows):
            y = 42 + i * lh
            on = (i == self.sel)
            if on:
                cv2.rectangle(img, (4, y - 1), (width - 4, y + lh - 3), (58, 58, 58), -1)
            col = (255, 255, 120) if on else (185, 185, 185)
            it.append(("▶" if on else " ", 8, y, 13, col))
            it.append((n, 26, y, 13, col))
            it.append(("%.6g" % val if abs(val) < 1e4 else "%.4e" % val,
                       92, y, 13, col))
            it.append(("±%.4g" % (st * self.stepmul), 205, y, 12, (135, 135, 135)))
            it.append((doc, 285, y, 12, (245, 245, 165) if on else (142, 142, 142)))

        y = 46 + lh * len(rows) + 6

        # --- 왜곡 지표: 잰 선. 커버율이 낮으면 숫자를 믿지 말라고 색으로 알린다 ---
        if meas:
            parts, warn = [], False
            for i, m in enumerate(meas):
                a, b, _, cov, span = m
                bad = (not np.isfinite(a)) or cov < 0.9 or span > 26.0
                warn = warn or bad
                parts.append("#%d %.2f/%.2f px%s"
                             % (i + 1, a, b,
                                (" ▲커버 %.0f%%" % (100 * cov)) if cov < 0.9 else
                                (" ▲끌림 %.0f" % span) if span > 26.0 else ""))
            it.append(("휨(최대/RMS): " + "  ".join(parts)
                       + ("   ← ▲ 표시는 다른 선에 끌렸거나 창을 넘은 것, 숫자 신뢰 금지"
                          if warn else ""),
                       8, y, 14, (100, 160, 255) if warn else (120, 255, 160)))
        else:
            it.append(("[L] 누른 뒤 보정영상에서 선 양끝을 클릭하면 휜 정도가 숫자로 나온다"
                       + ("      ◀ 클릭 대기중" if self.line_mode else ""),
                       8, y, 14, (0, 200, 255) if self.line_mode else (130, 130, 130)))

        # --- ★ 왜곡의 주 지표: 화면 전체 직선성. 이 숫자를 최소화하는 게 목표다 ---
        sr, sm, nl, npt, _per = self.straight_score()
        if sr is not None:
            b = straightness(self.lines, np.array([[self.dist0["fx"], 0, self.dist0["cx"]],
                                                   [0, self.dist0["fy"], self.dist0["cy"]],
                                                   [0, 0, 1.0]]),
                             np.array([self.dist0["k1"], self.dist0["k2"],
                                       self.dist0["p1"], self.dist0["p2"],
                                       self.dist0["k3"]]), self.new_K)[0]
            better = (b is not None and sr < b - 1e-9)
            col = ((120, 255, 160) if better else
                   (200, 200, 200) if (b is not None and abs(sr - b) < 1e-9) else
                   (100, 160, 255))
            it.append(("★ 화면전체 직선성(선 %d개/점 %d개): RMS %.3f px / 최대 %.2f px"
                       % (nl, npt, sr, sm)
                       + ("   시작값 %.3f px → %s" % (b, "개선 ▲" if better else "악화")
                          if b is not None else ""),
                       8, y + 21, 15, col))
        else:
            it.append(("화면전체 직선성: 추적된 선이 없다 ([t] 로 재시도)",
                       8, y + 21, 13, (140, 140, 140)))

        # --- H 지표: 실측 대응점. 범위가 좁다는 걸 명시한다 (전체 판정용이 아니다) ---
        rms, mx, n = self.h_accuracy()
        if n and rms is not None:
            base = h_metric(self.H_yaml, self.h_points_now())[0]
            col = ((120, 255, 160) if rms <= 0.010 else
                   (0, 220, 255) if rms <= 0.030 else (90, 130, 255))
            txt = ("H 오차(근거리 %d점, 앞 0.35~1.2m 만): RMS %.1f mm / 최대 %.1f mm"
                   % (n, 1000 * rms, 1000 * mx))
            if base is not None:
                txt += "   (원래 %.1f mm)" % (1000 * base)
            it.append((txt, 8, y + 42, 13, col))
        elif self.h_pts_ref:
            it.append(("H 오차: 계산 불가 (H 가 퇴화했는지 확인)", 8, y + 42, 13,
                       (90, 130, 255)))

        it.append(("↑↓(w x) 선택   ←→(a d) 값   [ ] 스텝   0 초기화   R 전체초기화   "
                   "★f H자동적합   t 선재추적   m H모드   L 자   C 지움   g 격자   "
                   "e 대비   N new_K재계산   n 다음사진   s 저장   q 종료",
                   8, y + 62, 12, (150, 150, 150)))

        if self.msg and time.time() - self.msg_t < 4.0:
            it.append((self.msg, 8, y + 80, 14, (120, 255, 255)))
        elif self.h_mode == "phys":
            it.append(("▲ 물리 6개 모드는 8자유도 H 를 표현 못 한다 — 저장 전에 [m] 로 "
                       "행렬 모드로 돌아갈 것", 8, y + 80, 12, (100, 160, 255)))
        elif self.dist_changed():
            it.append(("K/D 를 바꿨으면 [f] 를 눌러 H 를 다시 적합할 것 "
                       "(왜곡은 위 ★직선성으로, H 는 그 아래 mm 로 판정)",
                       8, y + 80, 12, (150, 190, 150)))
        return TXT.draw(img, it)

    def render(self):
        # 키를 안 눌렀으면 통째로 재사용한다 (매 프레임 다시 그리면 창이 버벅인다)
        if not (self.dirty or self.ui_dirty) and self._panel is not None:
            return self._panel
        if self.dirty or self._cache is None:
            und = self.undistort()
            self._dk = darkmap(und) if self.segs else None
            src = enhance(und) if self.do_enh else und
            Hm = self.H()
            bevimg = self.bev(src, Hm)
            meas = []
            for p0, p1 in self.segs:
                r = measure_segment(self._dk, p0, p1)
                if r is None:
                    r = (float("nan"), float("nan"), np.zeros((0, 2)), 0.0, 0.0)
                meas.append(r)
            self._cache = (src.copy(), bevimg, Hm, meas)
            self.dirty = False
        src, bevimg, Hm, meas = self._cache

        left = src.copy()
        if self.show_grid:
            self.draw_ground_grid(left, Hm)
        for i, ((p0, p1), m) in enumerate(zip(self.segs, meas)):
            cv2.line(left, tuple(np.int32(p0)), tuple(np.int32(p1)), (0, 255, 255), 1)
            for q in m[2]:
                cv2.circle(left, (int(q[0]), int(q[1])), 1, (0, 0, 255), -1)
            cv2.putText(left, "#%d" % (i + 1), tuple(np.int32(p0) + np.int32([4, -4])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        if self.pending is not None:
            cv2.drawMarker(left, tuple(np.int32(self.pending)), (0, 165, 255),
                           cv2.MARKER_CROSS, 14, 2)

        bv = bevimg.copy()
        if self.show_grid:
            self.draw_bev_grid(bv)

        th = self.args.view_h
        sL = th / left.shape[0]
        L = cv2.resize(left, (int(round(left.shape[1] * sL)), th))
        self.und_scale, self.und_off = sL, (0, 0)
        sB = th / max(1, bv.shape[0])
        Bp = cv2.resize(bv, (max(1, int(round(bv.shape[1] * sB))), th))
        for im, lab in ((L, "보정영상 — 초록격자가 실제 줄눈/테이프에 얹혀야 H 가 맞다"),
                        (Bp, "BEV — 타일이 정사각형이고 선이 곧아야 한다")):
            cv2.rectangle(im, (0, 0), (im.shape[1] - 1, 20), (0, 0, 0), -1)
            TXT.draw(im, [(lab, 5, 3, 14, (255, 255, 255))])
        top = cv2.hconcat([L, Bp])
        self._panel = cv2.vconcat([top, self.hud(top.shape[1], meas)])
        self._msg_on = bool(self.msg) and (time.time() - self.msg_t < 4.0)
        self.ui_dirty = False
        return self._panel

    # ---------- 마우스 ----------
    def on_mouse(self, ev, x, y, flags, _):
        if ev != cv2.EVENT_LBUTTONDOWN or not self.line_mode:
            return
        ux = (x - self.und_off[0]) / self.und_scale
        uy = (y - self.und_off[1]) / self.und_scale
        h, w = self.size[1], self.size[0]
        if not (0 <= ux < w and 0 <= uy < h):
            return
        self.ui_dirty = True
        if self.pending is None:
            self.pending = np.array([ux, uy])
        else:
            self.segs.append((self.pending, np.array([ux, uy])))
            self.pending = None
            self.line_mode = False
            self.dirty = True
            self.note("선 #%d 추가 — 파라미터 바꿀 때마다 숫자가 갱신된다" % len(self.segs))

    # ---------- 저장 ----------
    def save(self):
        if yaml is None:
            self.note("PyYAML 없음 — 저장 불가"); return
        path = self.args.camera
        try:
            with open(path, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            y = {}

        # 백업은 절대 덮어쓰지 않는다. 고정된 .bak 하나만 쓰면 두 번째 저장에서
        # 진짜 원본이 날아간다 — 시간표를 붙이고, 최초 원본은 .orig 로 따로 남긴다.
        stamp = time.strftime("%m%d-%H%M%S")
        try:
            if not os.path.exists(path + ".orig"):
                shutil.copyfile(path, path + ".orig")
            shutil.copyfile(path, "%s.bak.%s" % (path, stamp))
        except OSError:
            pass

        K, D, Hm = self.K(), self.D(), self.H()
        y["model"] = "plumb_bob"
        y["K"] = [[float(v) for v in r] for r in K]
        y["D"] = [float(v) for v in D]
        y["new_K"] = [[float(v) for v in r] for r in self.new_K]
        y["H"] = [[float(v) for v in r] for r in Hm]
        # new_K/H 는 캘리브 해상도 기준이다. 다른 크기 사진으로 열었더라도 파일에
        # 적힌 캘리브 해상도를 유지해야 한다 (현재 프레임 크기로 덮으면 거짓말이 된다).
        if not (y.get("image_width") and y.get("image_height")):
            y["image_width"], y["image_height"] = int(self.size[0]), int(self.size[1])
        y["intrinsics_calibrated"] = True
        y["h_calibrated"] = True
        y["calibrated"] = True
        y["intrinsics_source"] = "tools/tuner.py (수동 튜닝 %s)" % stamp

        # h_rms_m 은 캘리브 정확도를 주장하는 값이다. 안 갱신하면 파일이 거짓말을
        # 하게 되므로(뷰어 4곳이 이 값을 출력한다) 실측 대응점으로 다시 계산한다.
        rms, mx, n = self.h_accuracy()
        if n and rms is not None:
            y["h_rms_m"] = round(float(rms), 5)
            y["h_max_m"] = round(float(mx), 5)
            y["h_points_n"] = int(n)
            # 원본픽셀 기준으로 남긴다 — 이게 K/D 를 갈아끼워도 안 깨지는 진실원이다.
            y["h_points_raw"] = [[round(float(t), 4) for t in p] for p in self.h_raw]
            y["h_points"] = [[round(float(t), 4) for t in p]
                             for p in self.h_points_now()]
            y["h_points_new_K"] = [[float(v) for v in r] for r in self.new_K]
            y["h_points_note"] = ("h_points_raw = 원본(왜곡있는) 픽셀 기준 [u,v,x,y] — "
                                  "K/D 를 바꿔도 유효. h_points 는 위 new_K 기준 "
                                  "보정좌표 사본(참고용)")
        else:
            y.pop("h_rms_m", None)
            y.pop("h_max_m", None)
        y["h_source"] = ("tools/tuner.py 수동 튜닝 (%s 모드)"
                         % ("물리6개" if self.h_mode == "phys" else "행렬9원소"))
        if self.h_mode == "phys" and self.phys is not None:
            p = self.phys
            y["h_phys"] = {k: float(p[k]) for k in
                           ("height", "pitch", "yaw", "roll", "cam_x", "cam_y")}
        else:
            y.pop("h_phys", None)

        hdr = ("# tools/tuner.py 로 수동 튜닝한 값 (%s)\n"
               "# K/D: 렌즈 내부·왜곡.  new_K: undistort 결과 픽셀 좌표계를 정의.\n"
               "# H: undistort(new_K 기준) 픽셀 → 지면 미터 (x전방/y좌).\n"
               "#    new_K 나 K/D 가 바뀌면 H 는 통째로 무효 — 반드시 H 를 다시 맞출 것.\n"
               "# h_frame: H 출력 좌표의 원점 (lidar_frame = 라이다 지면투영점).\n"
               "# h_rms_m/h_max_m: h_points(실측 대응점) 재투영 오차. 저장 시 재계산됨.\n"
               "# 이전 파일: %s.bak.%s / 최초 원본: %s.orig\n"
               % (stamp, os.path.basename(path), stamp, os.path.basename(path)))
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(hdr)
                yaml.safe_dump(y, f, allow_unicode=True, sort_keys=True)
        except OSError as e:
            self.note("저장 실패: %s" % e)
            return

        msg = "저장 완료 (백업 .bak.%s)" % stamp
        if n and rms is not None:
            msg += " — H 오차 RMS %.1f mm" % (1000 * rms)
            if rms > 0.030:
                msg += " ▲ 30mm 초과, 아직 안 맞음"
        if self.dist_changed():
            msg += " / K/D 변경됨 → 차에 올리기 전 H 재확인"
        self.note(msg)

    # ---------- 메인 루프 ----------
    def run(self):
        win = "calib tuner"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self.on_mouse)
        print(__doc__.split("실행")[0])
        while True:
            cv2.imshow(win, self.render())
            k = cv2.waitKeyEx(30)
            if k == -1:
                continue
            c = k & 0xFF
            # 화살표 keysym(65361~65364)은 &0xFF 하면 81~84 = 'Q','R','S','T' 와
            # 겹친다. 그래서 raw 코드로만 판정하고, 대체 글자는 겹치지 않는
            # w/x/a/d 를 쓴다 ('s' 는 저장이라 방향키로 안 쓴다).
            UP = k in (65362, 2490368) or c == ord('w')
            DOWN = k in (65364, 2621440) or c == ord('x')
            LEFT = k in (65361, 2424832) or c == ord('a')
            RIGHT = k in (65363, 2555904) or c == ord('d')
            n = len(self.params())
            if c in (27, ord('q')):
                break
            elif UP:
                self.sel = (self.sel - 1) % n
            elif DOWN:
                self.sel = (self.sel + 1) % n
            elif LEFT:
                self.bump(-1)
            elif RIGHT:
                self.bump(+1)
            elif c == ord('\t') or c == 9:
                self.group = "h" if self.group == "dist" else "dist"
                self.sel = 0
            elif c == ord('['):
                self.stepmul = max(1e-4, self.stepmul / 10.0)
            elif c == ord(']'):
                self.stepmul = min(1e4, self.stepmul * 10.0)
            elif c == ord('0'):
                self.reset_sel()
            elif c == ord('R'):
                self.reset_all()
            elif c == ord('m'):
                if self.h_mode == "phys":
                    self.helem = list(self.H().ravel())      # 값 튐 없이 이어받기
                    self.h_mode = "elem"
                    self.note("행렬 원소 9개 모드 (현재 H 를 그대로 이어받음)")
                else:
                    self.h_mode = "phys"
                    x0 = ([self.phys[q] for q in ("height", "pitch", "yaw",
                                                  "roll", "cam_x", "cam_y")]
                          if self.phys is not None else None)
                    self.phys, self.fiterr = fit_phys_to_H(
                        np.array(self.helem).reshape(3, 3), self.new_K, x0=x0)
                    # 판정은 프로베니우스 잔차가 아니라 실제 지면오차(mm)로 한다 —
                    # 잔차 0.0002 인데 지면오차 200mm 인 경우가 실제로 나온다.
                    self.dirty = True
                    before = h_metric(np.array(self.helem).reshape(3, 3),
                                      self.h_points_now())[0]
                    after = self.h_accuracy()[0]
                    if before is not None and after is not None:
                        worse = 1000.0 * (after - before)
                        self.note("물리 6개 모드 — H 오차 %.1f → %.1f mm (%+.1f mm)%s"
                                  % (1000 * before, 1000 * after, worse,
                                     "  ▲ 저장 전 [m] 로 되돌릴 것" if worse > 5 else ""))
                    else:
                        self.note("물리 6개 모드 (실측점이 없어 mm 검증 불가 — 주의)")
                self.group = "h"
                self.sel = 0
                self.dirty = True
            elif c == ord('f') or c == ord('F'):
                before = self.h_accuracy()[0]
                Hm, err = self.fit_H()
                if Hm is None:
                    self.note("H 자동적합 불가 — %s" % err)
                else:
                    self.helem = list(Hm.ravel())
                    self.h_mode = "elem"
                    self.group = "h"
                    self.sel = 0
                    self.dirty = True
                    rms, mx, nn = h_metric(Hm, self.h_points_now())
                    self.note("H 자동적합 완료 (실측 %d점 최소자승): 오차 %s → RMS %.1f mm "
                              "/ 최대 %.1f mm"
                              % (nn,
                                 ("%.1f mm" % (1000 * before)) if before is not None else "?",
                                 1000 * rms, 1000 * mx))
            elif c == ord('L') or c == ord('l'):
                self.line_mode = True
                self.pending = None
                self.note("보정영상에서 선의 양 끝을 순서대로 클릭")
            elif c == ord('C') or c == ord('c'):
                self.segs, self.pending = [], None
                self.dirty = True
                self.note("잰 선 전부 지움")
            elif c == ord('g'):
                self.show_grid = not self.show_grid
            elif c == ord('e'):
                self.do_enh = not self.do_enh
                self.dirty = True
            elif c == ord('N'):
                self.new_K = np.asarray(cv2.getOptimalNewCameraMatrix(
                    self.K(), self.D(), self.size, self.args.alpha)[0],
                    np.float64).reshape(3, 3)
                self.newk_src = "재계산"
                self.segs, self.pending = [], None
                self.dirty = True
                self.note("new_K 재계산 — 좌표계가 바뀌었으니 H 를 다시 맞출 것")
            elif c == ord('n') and not self.live:
                self.shot_i += 1
                self.frame = self._load_shot()
                self.dirty = True
                self.note("사진: %s" % self.shot_name)
            elif c == ord(' ') and self.live:
                self.frame = self._wait_live(3.0)
                self.dirty = True
            elif c == ord('s'):
                self.save()
            else:
                # 백엔드(WSLg/Qt/GTK)마다 화살표 키 코드가 달라서, 못 알아먹은 키는
                # 코드를 그대로 보여준다. 화살표가 안 먹으면 w/x/a/d 로 하면 되고,
                # 이 코드를 알려주면 매핑을 추가할 수 있다.
                self.note("알 수 없는 키 (코드 %d / &0xFF=%d) — 방향키 대신 w x a d 사용 가능"
                          % (k, c))
            if c in (ord('['), ord(']')):
                self.note("스텝 배율 x%.4g" % self.stepmul)
            self.ui_dirty = True
        cv2.destroyAllWindows()
        if self.live:
            try:
                self._ros_node.destroy_node()
                if self._ros.ok():
                    self._ros.shutdown()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="왜곡보정/H 수동 튜너")
    ap.add_argument("--camera", default=os.path.join(_PKG, "config", "camera.yaml"))
    ap.add_argument("--image", nargs="*", default=None, help="튜닝에 쓸 사진(여러 개 가능)")
    ap.add_argument("--live", action="store_true", help="/image_raw 실시간")
    ap.add_argument("--topic", default="/image_raw")
    ap.add_argument("--alpha", type=float, default=0.0, help="new_K 계산 alpha")
    ap.add_argument("--pitch", type=float, default=0.453333, help="바닥 타일 피치 m")
    ap.add_argument("--grid-x0", type=float, default=None,
                    help="격자 가로선의 원점 x (m). 기본 = -h_lidar_offset_x "
                         "(A' 를 줄눈에 맞춰 뒀을 때 실제 줄눈 위치)")
    ap.add_argument("--grid-y0", type=float, default=None,
                    help="격자 세로선의 원점 y (m). 기본 0")
    ap.add_argument("--range-x", type=float, nargs=2, default=[0.0, 2.6])
    ap.add_argument("--range-y", type=float, nargs=2, default=[-1.2, 1.2])
    ap.add_argument("--ppm", type=int, default=185, help="BEV 픽셀/미터")
    ap.add_argument("--view-h", type=int, default=470, help="화면 패널 높이 px")
    ap.add_argument("--enhance", action="store_true")
    ap.add_argument("--selftest", default=None, metavar="OUT.jpg",
                    help="창 없이 한 장 그려 저장하고 종료 (원격/헤드리스 점검용)")
    args = ap.parse_args()
    if not os.path.exists(args.camera):
        print("camera.yaml 없음:", args.camera); sys.exit(1)
    t = Tuner(args)
    if args.selftest:
        # 창을 안 띄우고 모든 경로를 한 번씩 태운다 (표시 코드 오류 조기 검출)
        t.segs.append((np.array([60.0, t.size[1] * 0.72]),
                       np.array([t.size[0] - 60.0, t.size[1] * 0.72])))
        t.dirty = True
        panel = t.render()
        t.group = "h"; t.sel = 1; t.bump(+1); t.bump(-1)
        t.render()
        t.helem = list(t.H().ravel()); t.h_mode = "elem"; t.sel = 4
        t.bump(+1); t.render()
        cv2.imwrite(args.selftest, panel)
        print("selftest OK — %s (%dx%d)  파라미터 %d개"
              % (args.selftest, panel.shape[1], panel.shape[0], len(t.params())))
        return
    t.run()


if __name__ == "__main__":
    main()
