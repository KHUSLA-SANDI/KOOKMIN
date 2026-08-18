#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델-비의존 테이프 추적 v2 — 선 갈아타기(hop) 방지 + 종횡비 자유.

v1 실패: 교차점에서 window 내 최대값을 골라 다른 선으로 hop → 가짜 64px 편차.
v2: (1) 예측값에 가장 가까운 국소봉우리 선택  (2) |측정-예측|>2px 이면 거부
    (3) 최근 40점 2차 적합으로 예측  (4) 선끼리 중복 제거
종횡비 a(=fy/fx)를 자유롭게 둔다 — 센서를 눌러 담은 영상이면 방사왜곡이 타원이라
어떤 k1,k2,k3 로도 직선이 안 펴진다. a≈1 이 나오면 그 가설은 기각.

사용: python3 _trace2.py [--save]
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
dk = cv2.GaussianBlur(g, (0, 0), 21) - g          # 어두울수록 큰 값

WIN, THR, GATE, MAXMISS = 7, 10.0, 2.0, 8


def peaks(prof, lo):
    """국소 최대들을 서브픽셀로 반환"""
    out = []
    for k in range(1, len(prof) - 1):
        if prof[k] >= prof[k - 1] and prof[k] > prof[k + 1] and prof[k] >= THR:
            a, b, c = prof[k - 1], prof[k], prof[k + 1]
            den = a - 2 * b + c
            d = 0.5 * (a - c) / den if abs(den) > 1e-6 else 0.0
            out.append(lo + k + max(-1.0, min(1.0, d)))
    return out


def predict(hist, pos):
    n = len(hist)
    if n < 3:
        return hist[-1][1]
    h = np.array(hist[-40:])
    deg = 2 if len(h) >= 8 else 1
    return float(np.polyval(np.polyfit(h[:, 0], h[:, 1], deg), pos))


def trace(seed_pos, seed_cross, mode):
    """mode='h': u 를 따라 v 추적 / 'v': v 를 따라 u 추적.  hop 방지 게이트 적용."""
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
            pred = predict(hist, pos)
            lo = int(max(0, pred - WIN)); hi = int(min(LIM - 1, pred + WIN))
            if hi - lo < 4:
                break
            prof = (dk[lo:hi + 1, pos - 1:pos + 2].mean(axis=1) if mode == "h"
                    else dk[pos - 1:pos + 2, lo:hi + 1].mean(axis=0))
            cand = peaks(prof, lo)
            best = min(cand, key=lambda c: abs(c - pred)) if cand else None
            if best is None or abs(best - pred) > GATE:
                miss += 1
                if miss > MAXMISS:
                    break
                continue
            miss = 0
            hist.append((float(pos), best))
            pts.append((float(pos), best) if mode == "h" else (best, float(pos)))
    if not pts:
        return np.zeros((0, 2))
    P = np.array(pts)
    return P[np.argsort(P[:, 0 if mode == "h" else 1])]


def overlaps(P, mode, kept):
    """이미 추적한 선과 같은 선인지 (같은 pos 에서 cross 가 3px 이내면 동일)"""
    ax, ay = (0, 1) if mode == "h" else (1, 0)
    for Q, m2 in kept:
        if m2 != mode:
            continue
        common = np.intersect1d(P[:, ax].astype(int), Q[:, ax].astype(int))
        if len(common) < 20:
            continue
        pi = {int(r[ax]): r[ay] for r in P}
        qi = {int(r[ax]): r[ay] for r in Q}
        d = np.array([abs(pi[c] - qi[c]) for c in common])
        if np.median(d) < 3.0:
            return True
    return False


kept = []
for col in (200, 320, 440):                    # 가로선 시드
    prof = dk[:, col - 12:col + 13].mean(axis=1)
    for v in peaks(prof, 0):
        if v < 250:
            continue
        P = trace(col, v, "h")
        if len(P) >= 120 and P[:, 0].ptp() > 250 and not overlaps(P, "h", kept):
            kept.append((P, "h"))
for row in (300, 380, 450):                    # 세로선 시드
    prof = dk[row - 12:row + 13, :].mean(axis=0)
    for u in peaks(prof, 0):
        P = trace(u, row, "v")
        if len(P) >= 90 and P[:, 1].ptp() > 90 and not overlaps(P, "v", kept):
            kept.append((P, "v"))

lines = [P for P, _ in kept]
names = [f"{m.upper()}{i}" for i, (_, m) in enumerate(kept)]
print(f"추적된 선 {len(lines)}개 (hop 게이트 {GATE}px)")
COL = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
       (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255)]
vis = raw.copy()
for i, (nm, P) in enumerate(zip(names, lines)):
    print(f"  {nm}: {len(P):4d}점  u {P[:,0].min():5.0f}~{P[:,0].max():5.0f}  "
          f"v {P[:,1].min():5.0f}~{P[:,1].max():5.0f}")
    for u_, v_ in P:
        cv2.circle(vis, (int(u_), int(v_)), 1, COL[i % len(COL)], -1)
cv2.imwrite("/home/xytron/trace2_raw.jpg", vis)
if len(lines) < 4:
    print("선 부족 — 중단"); sys.exit()


def resid(K, D):
    out = []
    for P in lines:
        u = cv2.undistortPoints(P.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
        c = u - u.mean(0)
        _, _, Vt = np.linalg.svd(c, full_matrices=False)
        out.append(c @ Vt[1])
    return np.concatenate(out)


def report(tag, K, D):
    r = resid(K, D)
    print(f"\n[{tag}] 전체 RMS {np.sqrt((r**2).mean()):.3f} px  최대 {np.abs(r).max():.2f} px")
    i0 = 0
    for nm, P in zip(names, lines):
        s = r[i0:i0 + len(P)]; i0 += len(P)
        print(f"    {nm}: RMS {np.sqrt((s**2).mean()):6.2f}  최대 {np.abs(s).max():6.2f}")
    return np.sqrt((r ** 2).mean())


def unpack(p):
    K = K0.copy()
    K[0, 2], K[1, 2] = p[0], p[1]
    K[1, 1] = K0[0, 0] * p[5]                  # 종횡비 a = fy/fx
    return K, np.array([p[2], p[3], 0.0, 0.0, p[4]])


rms0 = report("현재 K/D", K0, D0)

NAMES = ["cx", "cy", "k1", "k2", "k3", "a"]
p0 = np.array([K0[0, 2], K0[1, 2], D0[0], D0[1], 0.0, K0[1, 1] / K0[0, 0]])
lo = np.array([K0[0, 2] - 70, K0[1, 2] - 70, -1.2, -1.2, -1.2, 0.70])
hi = np.array([K0[0, 2] + 70, K0[1, 2] + 70, 0.4, 1.2, 1.2, 1.40])
sol = least_squares(lambda p: resid(*unpack(p)), p0, bounds=(lo, hi),
                    loss="soft_l1", f_scale=0.7, max_nfev=20000)
K1, D1 = unpack(sol.x)
rms1 = report("재보정(cx,cy,k1,k2,k3,종횡비)", K1, D1)

print(f"\ncx {K0[0,2]:7.2f} → {K1[0,2]:7.2f}    cy {K0[1,2]:7.2f} → {K1[1,2]:7.2f}")
print(f"k1 {D0[0]:+8.5f} → {D1[0]:+8.5f}   k2 {D0[1]:+8.5f} → {D1[1]:+8.5f}   k3 → {D1[4]:+8.5f}")
print(f"종횡비 fy/fx {K0[1,1]/K0[0,0]:.4f} → {sol.x[5]:.4f}   (1.00 이면 눌러담기 가설 기각)")
at = [n for n, v, l_, h_ in zip(NAMES, sol.x, lo, hi)
      if abs(v - l_) < 1e-5 or abs(v - h_) < 1e-5]
print("경계 접촉:", at if at else "없음 ✅")
print(f"개선율 {(1-rms1/rms0)*100:.0f}%   (RMS {rms0:.2f} → {rms1:.2f} px)")

nk, _ = cv2.getOptimalNewCameraMatrix(K1, D1, (Ww, Hh), 0.0)
mx, my = cv2.initUndistortRectifyMap(K1, D1, np.eye(3), nk, (Ww, Hh), cv2.CV_32FC1)
cv2.imwrite("/home/xytron/trace2_und.jpg", cv2.remap(raw, mx, my, cv2.INTER_LINEAR))
nk0, _ = cv2.getOptimalNewCameraMatrix(K0, D0, (Ww, Hh), 0.0)
mx0, my0 = cv2.initUndistortRectifyMap(K0, D0, np.eye(3), nk0, (Ww, Hh), cv2.CV_32FC1)
cv2.imwrite("/home/xytron/trace2_old.jpg", cv2.remap(raw, mx0, my0, cv2.INTER_LINEAR))
np.savez("/home/xytron/trace2_calib.npz", K=K1, D=D1, new_K=nk)
print("saved: trace2_raw.jpg(추적,선별 색상) trace2_old.jpg(기존보정) trace2_und.jpg(신규보정)")

if A.save:
    if rms1 > 1.5 or at:
        print(f"\n⚠️ 저장 안 함 — RMS {rms1:.2f}px (기준 1.5) / 경계접촉 {at}")
    else:
        import yaml
        with open(CFG, encoding="utf-8") as f:
            y = yaml.safe_load(f)
        y["K"] = [[float(t) for t in r] for r in K1]
        y["D"] = [float(t) for t in D1]
        y["new_K"] = [[float(t) for t in r] for r in nk]
        y["intrinsics_source"] = "dev/scripts/_trace2.py (hop-safe tape plumb-line)"
        y["h_calibrated"] = False
        y["calibrated"] = False
        with open(CFG, "w", encoding="utf-8") as f:
            f.write("# 자동생성 — dev/scripts/_trace2.py (hop 방지 추적 기반 K/D)\n")
            yaml.safe_dump(y, f, allow_unicode=True, sort_keys=True)
        print("\ncamera.yaml 갱신 ✅  (⚠️ K/D 가 바뀌었으므로 H 재계산 필요)")
