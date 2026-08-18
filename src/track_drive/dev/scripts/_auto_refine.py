#!/usr/bin/env python3
"""
줄눈 자동 검출로 H 정밀화.
현재 H 를 초기추정으로 써서 줄눈 예상 위치 근처의 '어두운 골'을 서브픽셀로 찾고,
그 대응점으로 H 를 재적합한다. 사람 클릭 오차(원거리 1px=50~80mm)를 제거하는 것이 목적.
"""
import sys
import numpy as np, cv2

sys.path.insert(0, "/home/xytron/xycar_ws/본선/track_drive")
from track_drive.lib import bev as B

CFG = "/home/xytron/xycar_ws/본선/track_drive/config/camera.yaml"
P = 0.453333
MIN_CONTRAST = 8.0

cam = B.load_camera_config(CFG)
raw = cv2.imread("/home/xytron/calib_raw.jpg")
und = B.Undistorter.from_config(cam, (raw.shape[1], raw.shape[0]))
img = und.apply(raw)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
gray = cv2.GaussianBlur(gray, (0, 0), 1.1)
Himg, Wimg = gray.shape

H = np.array(cam["H"], np.float64)


def inv_sign(Hm):
    Hi = np.linalg.inv(Hm)
    return Hi, np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2])


def g2p(Hi, wref, x, y):
    q = Hi @ np.array([x, y, 1.0])
    if abs(q[2]) < 1e-12 or np.sign(q[2]) != wref:
        return None
    u, v = q[0] / q[2], q[1] / q[2]
    if not (3 <= u < Wimg - 3 and 3 <= v < Himg - 3):
        return None
    return u, v


def subpix_min(prof, lo):
    k = int(np.argmin(prof))
    if k <= 0 or k >= len(prof) - 1:
        return None, 0.0
    a, b, c = prof[k - 1], prof[k], prof[k + 1]
    den = a - 2 * b + c
    d = 0.5 * (a - c) / den if abs(den) > 1e-6 else 0.0
    return lo + k + d, float(prof.max() - prof.min())


def scan_v(u, v0, half):           # 세로로 훑어 가로 줄눈 찾기
    u = int(round(u))
    lo, hi = int(round(v0 - half)), int(round(v0 + half))
    if lo < 1 or hi >= Himg - 1 or u < 3 or u >= Wimg - 3:
        return None, 0.0
    prof = gray[lo:hi + 1, u - 2:u + 3].mean(axis=1)
    return subpix_min(prof, lo)


def scan_u(v, u0, half):           # 가로로 훑어 세로 줄눈 찾기
    v = int(round(v))
    lo, hi = int(round(u0 - half)), int(round(u0 + half))
    if lo < 1 or hi >= Wimg - 1 or v < 3 or v >= Himg - 3:
        return None, 0.0
    prof = gray[v - 2:v + 3, lo:hi + 1].mean(axis=0)
    return subpix_min(prof, lo)


I_LIST = [1, 2, 3, 4, 5, 6]
J_LIST = [-2, -1, 0, 1, 2]
Y_SAMPLES = [-0.60, -0.30, 0.0, 0.30, 0.60]
X_SAMPLES = [0.55, 0.75, 1.0, 1.3, 1.7, 2.1]

for it in range(4):
    Hi, wref = inv_sign(H)
    half_t = [16, 9, 6, 4][it]      # 반복할수록 탐색창 축소
    pix, gnd, tags = [], [], []

    # 가로 줄눈 (x = i*P)
    for i in I_LIST:
        x = i * P
        for yy in Y_SAMPLES:
            p = g2p(Hi, wref, x, yy)
            if p is None:
                continue
            vv, ct = scan_v(p[0], p[1], half_t)
            if vv is None or ct < MIN_CONTRAST:
                continue
            pix.append([p[0], vv]); gnd.append([x, yy]); tags.append(f"row i={i} y={yy:+.2f}")

    # 세로 줄눈 (y = j*P)
    for j in J_LIST:
        y = j * P
        for xx in X_SAMPLES:
            p = g2p(Hi, wref, xx, y)
            if p is None:
                continue
            uu, ct = scan_u(p[1], p[0], half_t * 2)
            if uu is None or ct < MIN_CONTRAST:
                continue
            pix.append([uu, p[1]]); gnd.append([xx, y]); tags.append(f"col j={j} x={xx:.2f}")

    if len(pix) < 8:
        print(f"[{it}] 검출 부족 ({len(pix)}) — 중단"); break
    Pm, Gm = np.array(pix), np.array(gnd)
    Hn, mask = cv2.findHomography(Pm, Gm, cv2.RANSAC, 0.02)
    if Hn is None:
        print(f"[{it}] 적합 실패"); break
    Hn = Hn / Hn[2, 2]
    ph = np.hstack([Pm, np.ones((len(Pm), 1))])
    q = (Hn @ ph.T).T; q = q[:, :2] / q[:, 2:3]
    e = np.linalg.norm(q - Gm, axis=1)
    inl = mask.ravel().astype(bool)
    print(f"[{it}차] 검출 {len(Pm)}점 / inlier {inl.sum()} → "
          f"RMS {np.sqrt((e[inl]**2).mean())*1000:6.1f} mm, 최대 {e[inl].max()*1000:6.1f} mm "
          f"(창 ±{half_t}px)")
    H = Hn

np.save("/home/xytron/H_auto.npy", H)
print("\n최종 H 저장: /home/xytron/H_auto.npy")

# 최종 검증: 예측 vs 실측 줄눈 오차
Hi, wref = inv_sign(H)
print("\n=== 최종 검증 (예측 줄눈 vs 실제 줄눈) ===")
print(f"{'i':>2} {'x[m]':>6} {'u':>5} {'예측v':>7} {'실제v':>7} {'Δx[mm]':>8}")
errs = []
for i in I_LIST:
    x = i * P
    for yy in (0.3, 0.0, -0.3):
        p = g2p(Hi, wref, x, yy)
        if p is None:
            continue
        vv, ct = scan_v(p[0], p[1], 5)
        if vv is None or ct < MIN_CONTRAST:
            continue
        p2 = g2p(Hi, wref, x + 0.02, yy)
        if p2 is None:
            continue
        dvdx = (p2[1] - p[1]) / 0.02
        dx = (vv - p[1]) / dvdx * 1000
        errs.append(abs(dx))
        print(f"{i:2d} {x:6.3f} {p[0]:5.0f} {p[1]:7.1f} {vv:7.1f} {dx:+8.1f}")
if errs:
    a = np.array(errs)
    print(f"\n평균 |Δx| = {a.mean():.1f} mm, 중앙값 {np.median(a):.1f} mm, 최대 {a.max():.1f} mm")
