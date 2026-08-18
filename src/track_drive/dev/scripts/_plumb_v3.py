#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plumb-line 3차 시도: 이상점 제거 + k3 추가 + 경계 확대.
이번에도 (a) 파라미터가 경계에 붙거나 (b) 개선이 미미하면 → 렌즈모델 문제가 아니라
데이터(테이프/바닥 자체의 비직선성) 문제로 결론짓고 중단한다.
"""
import sys
import numpy as np, cv2
from scipy.optimize import least_squares

PKG = "/home/xytron/xycar_ws/src/track_drive"
sys.path.insert(0, PKG)
from track_drive.lib import bev as B

CFG = PKG + "/config/camera.yaml"
PITCH, LDX = 0.453333, 0.10
cam = B.load_camera_config(CFG)
K0 = np.array(cam["K"], np.float64); D0 = np.array(np.ravel(cam["D"])[:5], np.float64)
Hi = np.linalg.inv(np.array(cam["H"], np.float64))
wref = np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2])
_nk = cam.get("new_K"); newK0 = np.array(_nk, np.float64) if _nk is not None else K0.copy()

raw = cv2.imread(PKG + "/dev/shots/tape_raw.jpg")
Hh, Ww = raw.shape[:2]
g = cv2.GaussianBlur(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.8)
bgm = cv2.GaussianBlur(g, (0, 0), 25)


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


def u2r(u, v):
    n = [(u - newK0[0, 2]) / newK0[0, 0], (v - newK0[1, 2]) / newK0[1, 1]]
    p, _ = cv2.projectPoints(np.array([[n[0], n[1], 1.0]]), np.zeros(3), np.zeros(3), K0, D0)
    return float(p[0, 0, 0]), float(p[0, 0, 1])


def gu(x, y):
    q = Hi @ np.array([x - LDX, y, 1.0])
    if abs(q[2]) < 1e-12 or np.sign(q[2]) != wref:
        return None
    return q[0] / q[2], q[1] / q[2]


lines = []
for i in (1, 2, 3):
    P = []
    for yy in np.linspace(-1.0, 1.0, 90):
        p = gu(i * PITCH, yy)
        if p is None: continue
        ur, vr = u2r(*p); u = int(round(ur))
        if not (3 < u < Ww - 3 and 20 < vr < Hh - 3): continue
        lo = max(0, int(vr - 13)); hi = min(Hh - 1, int(vr + 13))
        c, ct = cen(g[lo:hi+1, u-1:u+2].mean(1), bgm[lo:hi+1, u-1:u+2].mean(1), lo)
        if c is not None and ct >= 20: P.append([float(u), c])
    if len(P) >= 15: lines.append(("row%d" % i, np.array(P)))
for j in (-1, 1):
    P = []
    for xx in np.linspace(0.42, 2.75, 110):
        p = gu(xx, j * PITCH)
        if p is None: continue
        ur, vr = u2r(*p); v = int(round(vr))
        if not (3 < v < Hh - 3 and 3 < ur < Ww - 3): continue
        lo = max(0, int(ur - 15)); hi = min(Ww - 1, int(ur + 15))
        c, ct = cen(g[v-1:v+2, lo:hi+1].mean(0), bgm[v-1:v+2, lo:hi+1].mean(0), lo)
        if c is not None and ct >= 20: P.append([c, float(v)])
    if len(P) >= 15: lines.append(("col%+d" % j, np.array(P)))

# ── 이상점 제거: 각 선을 undistort(현재 모델) 후 직선피팅, 3σ 초과 제거 ──
print("=== 이상점 제거 ===")
clean = []
for name, P in lines:
    u = cv2.undistortPoints(P.reshape(-1, 1, 2), K0, D0, P=K0).reshape(-1, 2)
    c = u - u.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    r = c @ Vt[1]
    s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
    keep = np.abs(r - np.median(r)) < 3 * s
    print(f"  {name}: {len(P)} → {int(keep.sum())}점 (잔차 MAD {s:.2f}px, 최대 {np.abs(r).max():.1f})")
    if keep.sum() >= 12:
        clean.append((name, P[keep]))
lines = clean
npts = sum(len(P) for _, P in lines)
print(f"정제 후 {len(lines)}선 / {npts}점")


def resid(K, D):
    out = []
    for _, P in lines:
        u = cv2.undistortPoints(P.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
        c = u - u.mean(0)
        _, _, Vt = np.linalg.svd(c, full_matrices=False)
        out.append(c @ Vt[1])
    return np.concatenate(out)


def unpack(p):
    K = K0.copy(); K[0, 2], K[1, 2] = p[0], p[1]
    return K, np.array([p[2], p[3], p[4], p[5], p[6]])


p0 = np.array([K0[0, 2], K0[1, 2], D0[0], D0[1], D0[2], D0[3], 0.0])
lo = np.array([K0[0, 2] - 80, K0[1, 2] - 80, -2.0, -2.0, -0.02, -0.02, -2.0])
hi = np.array([K0[0, 2] + 80, K0[1, 2] + 80, 1.0, 2.0, 0.02, 0.02, 2.0])

r0 = resid(K0, D0)
print(f"\n보정 전 직선성 RMS {np.sqrt((r0**2).mean()):.3f} px / 최대 {np.abs(r0).max():.2f}")
sol = least_squares(lambda p: resid(*unpack(p)), p0, bounds=(lo, hi),
                    loss="soft_l1", f_scale=0.8, max_nfev=6000)
K1, D1 = unpack(sol.x)
r1 = resid(K1, D1)
print(f"보정 후 직선성 RMS {np.sqrt((r1**2).mean()):.3f} px / 최대 {np.abs(r1).max():.2f}")
print(f"\ncx {K0[0,2]:.2f} → {K1[0,2]:.2f}   cy {K0[1,2]:.2f} → {K1[1,2]:.2f}")
print("D  ", np.round(D0, 5), "\n → ", np.round(D1, 5))

# 경계 접촉 여부 (수렴 실패 신호)
at = [(n, v) for n, v, l_, h_ in
      zip(["cx", "cy", "k1", "k2", "p1", "p2", "k3"], sol.x, lo, hi)
      if abs(v - l_) < 1e-6 or abs(v - h_) < 1e-6]
print("\n경계에 붙은 파라미터:", at if at else "없음 (정상 수렴)")
imp = 1 - np.sqrt((r1**2).mean()) / np.sqrt((r0**2).mean())
print(f"직선성 개선율 {imp*100:.0f}%")
print("\n판정:", "성공 가능성 있음 — H 재계산 진행 검토" if (not at and imp > 0.4)
      else "실패 — 렌즈모델로 설명 불가. 데이터(테이프/바닥) 문제로 결론")
