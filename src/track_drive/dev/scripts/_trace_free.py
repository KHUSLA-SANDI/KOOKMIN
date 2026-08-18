#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델-비의존 테이프 추적 + 왜곡 재보정 (순환논리 완전 배제).

이전 실패 원인: 기존 H 로 예측한 경로를 따라가며 점을 찾아 '휜 경로'를 정답으로 삼았다.
여기서는 H 도 D 도 쓰지 않고, RAW 영상에서 어두운 띠 자체를 따라간다.
그 다음 그 선들이 undistort 후 직선이 되도록 cx,cy,k1,k2,k3 를 맞춘다.

사용: python3 _trace_free.py [--save]
"""
import argparse
import sys
import numpy as np, cv2
from scipy.optimize import least_squares

PKG = "/home/xytron/xycar_ws/src/track_drive"
sys.path.insert(0, PKG)
from track_drive.lib import bev as B

CFG = PKG + "/config/camera.yaml"
ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true")
A = ap.parse_args()

cam = B.load_camera_config(CFG)
K0 = np.array(cam["K"], np.float64)
D0 = np.array(np.ravel(cam["D"])[:5], np.float64)

raw = cv2.imread(PKG + "/dev/shots/tape2_raw.jpg")
Hh, Ww = raw.shape[:2]
g = cv2.GaussianBlur(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.9)
bgm = cv2.GaussianBlur(g, (0, 0), 21)
dk = bgm - g                                     # 어두울수록 큰 값


def sub_peak(prof, lo):
    k = int(np.argmax(prof))
    if k <= 0 or k >= len(prof) - 1:
        return None, 0.0
    a, b, c = prof[k - 1], prof[k], prof[k + 1]
    den = a - 2 * b + c
    d = 0.5 * (a - c) / den if abs(den) > 1e-6 else 0.0
    return lo + k + d, float(b)


def trace(seed_u, seed_v, mode, win=8, thr=12.0, step=1):
    """어두운 띠를 양방향으로 따라간다. mode='h': u 를 따라 v 추적 / 'v': v 를 따라 u 추적"""
    pts = []
    for direction in (1, -1):
        cur = float(seed_v if mode == "h" else seed_u)
        vel = 0.0
        pos = int(seed_u if mode == "h" else seed_v)
        miss = 0
        while True:
            pos += direction * step
            if mode == "h":
                if not (2 <= pos < Ww - 2):
                    break
                pred = cur + vel * direction * step
                lo = int(max(0, pred - win)); hi = int(min(Hh - 1, pred + win))
                if hi - lo < 4:
                    break
                prof = dk[lo:hi + 1, pos - 1:pos + 2].mean(axis=1)
            else:
                if not (2 <= pos < Hh - 2):
                    break
                pred = cur + vel * direction * step
                lo = int(max(0, pred - win)); hi = int(min(Ww - 1, pred + win))
                if hi - lo < 4:
                    break
                prof = dk[pos - 1:pos + 2, lo:hi + 1].mean(axis=0)
            c, peak = sub_peak(prof, lo)
            if c is None or peak < thr:
                miss += 1
                if miss > 12:
                    break
                continue
            miss = 0
            newvel = (c - cur) / (direction * step)
            vel = 0.7 * vel + 0.3 * newvel if pts else newvel
            cur = c
            pts.append((float(pos), c) if mode == "h" else (c, float(pos)))
    P = np.array(pts)
    return P[np.argsort(P[:, 0 if mode == "h" else 1])] if len(P) else np.zeros((0, 2))


# ── 시드 자동 탐색: 중앙 열에서 어두운 띠 행 ──
colprof = dk[:, 305:335].mean(axis=1)
seeds_h = []
for v in range(255, Hh - 6):
    if colprof[v] > 16 and colprof[v] >= colprof[max(0, v - 4):v + 5].max():
        if not seeds_h or v - seeds_h[-1] > 10:
            seeds_h.append(v)
rowprof = dk[300:340, :].mean(axis=0)
seeds_v = []
for u in range(20, Ww - 20):
    if rowprof[u] > 16 and rowprof[u] >= rowprof[max(0, u - 4):u + 5].max():
        if not seeds_v or u - seeds_v[-1] > 12:
            seeds_v.append(u)
print("가로선 시드 v:", seeds_h)
print("세로선 시드 u:", seeds_v)

lines, names = [], []
for sv in seeds_h:
    P = trace(320, sv, "h")
    if len(P) >= 60 and P[:, 0].ptp() > 200:
        lines.append(P); names.append(f"H@v{sv}")
for su in seeds_v:
    P = trace(su, 320, "v")
    if len(P) >= 40 and P[:, 1].ptp() > 60:
        lines.append(P); names.append(f"V@u{su}")

print(f"\n추적된 선 {len(lines)}개")
vis = raw.copy()
for nm, P in zip(names, lines):
    print(f"  {nm}: {len(P)}점  u {P[:,0].min():.0f}~{P[:,0].max():.0f}  "
          f"v {P[:,1].min():.0f}~{P[:,1].max():.0f}")
    for u_, v_ in P:
        cv2.circle(vis, (int(u_), int(v_)), 1, (0, 0, 255), -1)
cv2.imwrite("/home/xytron/trace_free.jpg", vis)

if len(lines) < 3:
    print("선 부족 — 중단"); sys.exit()


def straight(K, D):
    out = []
    for P in lines:
        u = cv2.undistortPoints(P.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
        c = u - u.mean(0)
        _, _, Vt = np.linalg.svd(c, full_matrices=False)
        out.append(c @ Vt[1])
    return np.concatenate(out)


def unpack(p):
    K = K0.copy(); K[0, 2], K[1, 2] = p[0], p[1]
    return K, np.array([p[2], p[3], 0.0, 0.0, p[4]])


r0 = straight(K0, D0)
print(f"\n[현재 K/D] 직선성 RMS {np.sqrt((r0**2).mean()):.3f} px / 최대 {np.abs(r0).max():.2f}")
per = []
i0 = 0
for nm, P in zip(names, lines):
    r = r0[i0:i0 + len(P)]; i0 += len(P)
    per.append((nm, np.sqrt((r ** 2).mean())))
    print(f"    {nm}: {np.sqrt((r**2).mean()):6.2f} px")

p0 = np.array([K0[0, 2], K0[1, 2], D0[0], D0[1], 0.0])
lo = np.array([K0[0, 2] - 60, K0[1, 2] - 60, -1.5, -1.5, -1.5])
hi = np.array([K0[0, 2] + 60, K0[1, 2] + 60, 0.5, 1.5, 1.5])
sol = least_squares(lambda p: straight(*unpack(p)), p0, bounds=(lo, hi),
                    loss="soft_l1", f_scale=1.0, max_nfev=8000)
K1, D1 = unpack(sol.x)
r1 = straight(K1, D1)
print(f"\n[재보정] 직선성 RMS {np.sqrt((r1**2).mean()):.3f} px / 최대 {np.abs(r1).max():.2f}")
i0 = 0
for nm, P in zip(names, lines):
    r = r1[i0:i0 + len(P)]; i0 += len(P)
    print(f"    {nm}: {np.sqrt((r**2).mean()):6.2f} px")
print(f"\ncx {K0[0,2]:.2f} → {K1[0,2]:.2f}   cy {K0[1,2]:.2f} → {K1[1,2]:.2f}")
print(f"k1 {D0[0]:+.5f} → {D1[0]:+.5f}   k2 {D0[1]:+.5f} → {D1[1]:+.5f}   k3 → {D1[4]:+.5f}")
at = [n for n, v, l_, h_ in zip(["cx", "cy", "k1", "k2", "k3"], sol.x, lo, hi)
      if abs(v - l_) < 1e-6 or abs(v - h_) < 1e-6]
print("경계 접촉:", at if at else "없음 ✅")
imp = 1 - np.sqrt((r1 ** 2).mean()) / np.sqrt((r0 ** 2).mean())
print(f"개선율 {imp*100:.0f}%")

np.savez("/home/xytron/trace_calib.npz", K=K1, D=D1)
nk, _ = cv2.getOptimalNewCameraMatrix(K1, D1, (Ww, Hh), 0.0)
mx, my = cv2.initUndistortRectifyMap(K1, D1, np.eye(3), nk, (Ww, Hh), cv2.CV_32FC1)
cv2.imwrite("/home/xytron/trace_und.jpg", cv2.remap(raw, mx, my, cv2.INTER_LINEAR))
print("saved: trace_free.jpg(추적), trace_und.jpg(재보정 결과), trace_calib.npz")

if A.save and imp > 0.3 and not at:
    import yaml
    with open(CFG, encoding="utf-8") as f:
        y = yaml.safe_load(f)
    y["K"] = [[float(t) for t in r] for r in K1]
    y["D"] = [float(t) for t in D1]
    y["new_K"] = [[float(t) for t in r] for r in nk]
    y["intrinsics_source"] = "dev/scripts/_trace_free.py (model-free tape plumb-line)"
    y["h_calibrated"] = False           # K/D 바뀌면 H 무효
    y["calibrated"] = False
    with open(CFG, "w", encoding="utf-8") as f:
        f.write("# 자동생성 — dev/scripts/_trace_free.py (모델비의존 추적 기반 K/D)\n")
        yaml.safe_dump(y, f, allow_unicode=True, sort_keys=True)
    print("camera.yaml 갱신 (⚠️ H 는 무효화됨 — 재계산 필요)")
