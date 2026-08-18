#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yaw 직접 측정 — 세로 테이프(주행방향 평행선)들의 소실점으로.

주행방향과 평행한 직선들은 이미지에서 한 점(소실점)에 모인다.
카메라가 그 방향을 정면으로 보고 있으면 소실점 u = 주점 cx.
어긋난 만큼이 yaw 다:  yaw = atan((u_vp - cx) / fx)
소실점 v 는 지평선 높이 → pitch = atan((v_vp - cy) / fy)
"""
import sys
import numpy as np, cv2

PKG = "/home/xytron/xycar_ws/src/track_drive"
sys.path.insert(0, PKG)
from track_drive.lib import bev as B

PITCH, LDX = 0.453333, 0.10
cam = B.load_camera_config(PKG + "/config/camera.yaml")
K = np.array(cam["new_K"] if cam.get("new_K") is not None else cam["K"], np.float64)
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
print(f"주점 cx={cx:.2f} cy={cy:.2f}, 초점 fx={fx:.2f} fy={fy:.2f}")

raw = cv2.imread(PKG + "/dev/shots/tape2_raw.jpg")
Hh, Ww = raw.shape[:2]
und = B.Undistorter.from_config(cam, (Ww, Hh))
img = und.apply(raw)
g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.8)
bgm = cv2.GaussianBlur(g, (0, 0), 25)
Hi = np.linalg.inv(np.array(cam["H"], np.float64))
wref = np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2])


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


fits = {}
for j in (-2, -1, 1, 2):
    P = []
    for xx in np.linspace(0.40, 2.9, 130):
        p = gu(xx, j * PITCH)
        if p is None:
            continue
        v = int(round(p[1]))
        lo = max(0, int(p[0] - 16)); hi = min(Ww - 1, int(p[0] + 16))
        if hi - lo < 6 or v < 2 or v >= Hh - 2:
            continue
        c, ct = cen(g[v - 1:v + 2, lo:hi + 1].mean(0), bgm[v - 1:v + 2, lo:hi + 1].mean(0), lo)
        if c is not None and ct >= 20:
            P.append([c, float(v)])
    if len(P) < 20:
        print(f"  j={j}: 점 부족 {len(P)}"); continue
    P = np.array(P)
    # u = a*v + b (로버스트)
    m = np.ones(len(P), bool)
    for _ in range(3):
        A = np.polyfit(P[m, 1], P[m, 0], 1)
        r = P[:, 0] - np.polyval(A, P[:, 1])
        s = 1.4826 * np.median(np.abs(r[m] - np.median(r[m]))) + 1e-9
        m = np.abs(r - np.median(r[m])) < 2.5 * s
    fits[j] = A
    print(f"  j={j:+d}: {len(P)}점(inlier {int(m.sum())})  u = {A[0]:+.5f}*v + {A[1]:.2f}  "
          f"잔차 {np.std(r[m]):.2f}px")

if len(fits) < 2:
    print("선 부족"); sys.exit()

# 모든 쌍의 소실점
print("\n=== 소실점 (세로선 쌍의 교점) ===")
vps = []
ks = sorted(fits)
for i in range(len(ks)):
    for jj in range(i + 1, len(ks)):
        a1, b1 = fits[ks[i]]; a2, b2 = fits[ks[jj]]
        if abs(a1 - a2) < 1e-9:
            continue
        v = (b2 - b1) / (a1 - a2)
        u = a1 * v + b1
        vps.append((u, v))
        print(f"  j={ks[i]:+d} × j={ks[jj]:+d}  →  소실점 (u={u:7.2f}, v={v:7.2f})")

vps = np.array(vps)
u_vp, v_vp = np.median(vps[:, 0]), np.median(vps[:, 1])
print(f"\n소실점 중앙값: u={u_vp:.2f}, v={v_vp:.2f}")
yaw = np.degrees(np.arctan((u_vp - cx) / fx))
pit = np.degrees(np.arctan((v_vp - cy) / fy))
print(f"\n>>> yaw   = atan(({u_vp:.1f} - {cx:.1f}) / {fx:.1f}) = {yaw:+.2f}°")
print(f">>> pitch = atan(({v_vp:.1f} - {cy:.1f}) / {fy:.1f}) = {pit:+.2f}°")
print()
if abs(yaw) < 1.5:
    print("판정: 차가 타일과 거의 정렬됨 (yaw 무시 가능) →")
    print("      앞선 피팅의 yaw -9° 는 자유 8DOF 가 노이즈를 흡수한 허상이다.")
else:
    print(f"판정: 실제로 {yaw:+.1f}° 틀어져 있다. 차를 돌리거나 H 에 반영해야 한다.")
