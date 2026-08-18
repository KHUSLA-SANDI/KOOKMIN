#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
주점(cx, cy) 1차원 스윕 — 가로 테이프가 가장 곧아지는 값을 찾는다.

가설: 벤더 cx=305.6 이 틀렸고, 그래서 왜곡보정 중심이 어긋나 가로선이 휜다.
검증: cx 를 훑으며 '가로 테이프의 직선피팅 잔차'를 측정. 뚜렷한 최소점이 있으면 가설 확정.
파라미터 1개만 움직이므로 과적합 위험이 없다.
"""
import sys
import numpy as np, cv2

PKG = "/home/xytron/xycar_ws/src/track_drive"
sys.path.insert(0, PKG)
from track_drive.lib import bev as B

PITCH, LDX = 0.453333, 0.10
cam = B.load_camera_config(PKG + "/config/camera.yaml")
K0 = np.array(cam["K"], np.float64)
D0 = np.array(np.ravel(cam["D"])[:5], np.float64)
newK0 = np.array(cam["new_K"] if cam.get("new_K") is not None else K0, np.float64)
Hi = np.linalg.inv(np.array(cam["H"], np.float64))
wref = np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2])

raw = cv2.imread(PKG + "/dev/shots/tape2_raw.jpg")
Hh, Ww = raw.shape[:2]
g_raw = cv2.GaussianBlur(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.8)
bg_raw = cv2.GaussianBlur(g_raw, (0, 0), 25)


def cen(pg, pb, lo, thr=15.0, maxw=26):
    d = pb - pg
    idx = np.where(d > thr)[0]
    if len(idx) < 2:
        return None, 0.0
    seg = max(np.split(idx, np.where(np.diff(idx) > 1)[0] + 1), key=len)
    if len(seg) < 2 or len(seg) > maxw:
        return None, 0.0
    w = d[seg]
    return lo + float((seg * w).sum() / w.sum()), float(w.max())


def gu(x, y):
    q = Hi @ np.array([x - LDX, y, 1.0])
    if abs(q[2]) < 1e-12 or np.sign(q[2]) != wref:
        return None
    u, v = q[0] / q[2], q[1] / q[2]
    return (u, v) if (2 <= u < Ww - 2 and 2 <= v < Hh - 2) else None


def u2r(u, v):
    n = [(u - newK0[0, 2]) / newK0[0, 0], (v - newK0[1, 2]) / newK0[1, 1]]
    p, _ = cv2.projectPoints(np.array([[n[0], n[1], 1.0]]), np.zeros(3), np.zeros(3), K0, D0)
    return float(p[0, 0, 0]), float(p[0, 0, 1])


# ── RAW 영상에서 테이프 점 수집 (한 번만) ──
lines = {}
for i in (1, 2, 3):
    P = []
    for yy in np.linspace(-1.15, 1.15, 120):
        p = gu(i * PITCH, yy)
        if p is None:
            continue
        ur, vr = u2r(*p); u = int(round(ur))
        if not (3 < u < Ww - 3 and 20 < vr < Hh - 3):
            continue
        lo = max(0, int(vr - 13)); hi = min(Hh - 1, int(vr + 13))
        c, ct = cen(g_raw[lo:hi + 1, u - 1:u + 2].mean(1),
                    bg_raw[lo:hi + 1, u - 1:u + 2].mean(1), lo)
        if c is not None and ct >= 20:
            P.append([float(u), c])
    if len(P) >= 20:
        lines[f"row{i}"] = np.array(P)
for j in (-1, 1, 2):
    P = []
    for xx in np.linspace(0.40, 2.9, 130):
        p = gu(xx, j * PITCH)
        if p is None:
            continue
        ur, vr = u2r(*p); v = int(round(vr))
        if not (3 < v < Hh - 3 and 3 < ur < Ww - 3):
            continue
        lo = max(0, int(ur - 15)); hi = min(Ww - 1, int(ur + 15))
        c, ct = cen(g_raw[v - 1:v + 2, lo:hi + 1].mean(0),
                    bg_raw[v - 1:v + 2, lo:hi + 1].mean(0), lo)
        if c is not None and ct >= 20:
            P.append([c, float(v)])
    if len(P) >= 20:
        lines[f"col{j:+d}"] = np.array(P)
print("수집:", {k: len(v) for k, v in lines.items()})


def straight_rms(cx, cy, only=None):
    K = K0.copy(); K[0, 2] = cx; K[1, 2] = cy
    tot = []
    for name, P in lines.items():
        if only and not name.startswith(only):
            continue
        u = cv2.undistortPoints(P.reshape(-1, 1, 2), K, D0, P=K).reshape(-1, 2)
        c = u - u.mean(0)
        _, _, Vt = np.linalg.svd(c, full_matrices=False)
        r = c @ Vt[1]
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        m = np.abs(r - np.median(r)) < 3 * s        # 이상점 제외
        tot.append(r[m])
    if not tot:
        return np.nan
    r = np.concatenate(tot)
    return float(np.sqrt((r ** 2).mean()))


print(f"\n{'cx':>7} {'가로선 RMS':>12} {'세로선 RMS':>12} {'전체':>10}")
print("-" * 45)
best = (1e9, None)
for cx in np.arange(290, 375, 5.0):
    rr = straight_rms(cx, K0[1, 2], "row")
    rc = straight_rms(cx, K0[1, 2], "col")
    ra = straight_rms(cx, K0[1, 2])
    mark = ""
    if ra < best[0]:
        best = (ra, cx); mark = "  ←"
    print(f"{cx:7.1f} {rr:12.3f} {rc:12.3f} {ra:10.3f}{mark}")

# 미세 스윕
cxs = np.arange(best[1] - 6, best[1] + 6, 0.5)
vals = [straight_rms(c, K0[1, 2]) for c in cxs]
cx_opt = float(cxs[int(np.argmin(vals))])
print(f"\n최적 cx = {cx_opt:.1f}  (벤더값 {K0[0,2]:.1f}, 소실점 추정 341.8)")

# cy 도 같이
cys = np.arange(K0[1, 2] - 25, K0[1, 2] + 25, 1.0)
vals2 = [straight_rms(cx_opt, c) for c in cys]
cy_opt = float(cys[int(np.argmin(vals2))])
print(f"최적 cy = {cy_opt:.1f}  (벤더값 {K0[1,2]:.1f})")

print(f"\n직선성 RMS: 벤더 {straight_rms(K0[0,2], K0[1,2]):.3f} px  →  "
      f"최적 {straight_rms(cx_opt, cy_opt):.3f} px")
print(f"  가로선만: {straight_rms(K0[0,2], K0[1,2], 'row'):.3f} → "
      f"{straight_rms(cx_opt, cy_opt, 'row'):.3f} px")
print(f"  세로선만: {straight_rms(K0[0,2], K0[1,2], 'col'):.3f} → "
      f"{straight_rms(cx_opt, cy_opt, 'col'):.3f} px")
np.save("/home/xytron/cx_opt.npy", np.array([cx_opt, cy_opt]))
