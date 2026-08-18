#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
검정 테이프 선 검출 + 직선성 측정.
테이프는 어두운 '띠' 이므로 최소값 1점이 아니라 **밝기 가중 중심(centroid)** 으로 뽑는다
→ 서브픽셀 정밀도가 크게 올라간다.
왜곡보정 영상에서 테이프가 휘어 있으면 그만큼이 잔여 왜곡이다.
"""
import sys
import numpy as np, cv2

PKG = "/home/xytron/xycar_ws/src/track_drive"
sys.path.insert(0, PKG)
from track_drive.lib import bev as B

IMG = sys.argv[1] if len(sys.argv) > 1 else PKG + "/dev/shots/tape_raw.jpg"
cam = B.load_camera_config(PKG + "/config/camera.yaml")
raw = cv2.imread(IMG)
und = B.Undistorter.from_config(cam, (raw.shape[1], raw.shape[0]))
img = und.apply(raw)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
Hh, Ww = gray.shape

# 바닥 영역에서 테이프(어두운 띠) 이진화
roi = np.zeros_like(gray, np.uint8); roi[250:, :] = 255
blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
# 지역 임계: 주변보다 충분히 어두운 곳
bg = cv2.GaussianBlur(gray, (0, 0), 25)
dark = ((bg - blur) > 18).astype(np.uint8) * 255
dark = cv2.bitwise_and(dark, roi)
dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
cv2.imwrite("/home/xytron/tape_mask.jpg", dark)
print("테이프 화소:", int((dark > 0).sum()))


def band_center(prof_dark, prof_gray, lo):
    """어두운 띠의 밝기 가중 중심 (서브픽셀)"""
    idx = np.where(prof_dark > 0)[0]
    if len(idx) < 2:
        return None, 0
    # 연속 구간 중 가장 긴 것
    splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    seg = max(splits, key=len)
    if len(seg) < 2:
        return None, 0
    w = prof_gray[seg].max() - prof_gray[seg]          # 어두울수록 큰 가중치
    if w.sum() <= 1e-6:
        return None, 0
    c = float((seg * w).sum() / w.sum())
    return lo + c, len(seg)


# ── 가로 테이프(전방 방향으로 3줄) : 각 열 u 마다 세로로 띠 중심 ──
rows = {}
for u in range(20, Ww - 20, 4):
    col_d = dark[:, max(0, u - 1):u + 2].max(axis=1)
    col_g = gray[:, max(0, u - 1):u + 2].mean(axis=1)
    idx = np.where(col_d > 0)[0]
    if len(idx) == 0:
        continue
    for seg in np.split(idx, np.where(np.diff(idx) > 3)[0] + 1):
        if len(seg) < 2 or len(seg) > 40:
            continue
        lo, hi = seg[0], seg[-1]
        c, wdt = band_center(col_d[lo:hi + 1], col_g[lo:hi + 1], lo)
        if c is None:
            continue
        key = int(round(c / 12))                        # 대략적 그룹핑
        rows.setdefault(key, []).append((u, c, wdt))

groups = [v for v in rows.values() if len(v) >= 25]
groups.sort(key=lambda g: -np.mean([p[1] for p in g]))
print(f"\n가로 테이프 후보 {len(groups)}줄 (점 25개 이상)")

vis = img.copy()
for gi, g in enumerate(groups[:6]):
    us = np.array([p[0] for p in g], float)
    vs = np.array([p[1] for p in g], float)
    wd = np.mean([p[2] for p in g])
    A = np.polyfit(us, vs, 1)
    resid = vs - np.polyval(A, us)
    A2 = np.polyfit(us, vs, 2)
    resid2 = vs - np.polyval(A2, us)
    print(f"  [{gi}] v≈{vs.mean():6.1f}  점 {len(us):3d}개  폭 {wd:4.1f}px  "
          f"u {us.min():.0f}~{us.max():.0f}")
    print(f"       직선피팅 잔차 RMS {resid.std():5.2f} px (최대 {np.abs(resid).max():5.2f})   "
          f"2차피팅 잔차 {resid2.std():5.2f} px   → 휨(2차계수) {A2[0]*1e4:+.2f}e-4")
    for u_, v_ in zip(us, vs):
        cv2.circle(vis, (int(u_), int(round(v_))), 1, (0, 0, 255), -1)
    xs = np.linspace(us.min(), us.max(), 50)
    cv2.polylines(vis, [np.stack([xs, np.polyval(A, xs)], 1).astype(np.int32)],
                  False, (0, 255, 255), 1)

cv2.imwrite("/home/xytron/tape_lines.jpg", vis)
print("\nsaved: tape_mask.jpg, tape_lines.jpg")
