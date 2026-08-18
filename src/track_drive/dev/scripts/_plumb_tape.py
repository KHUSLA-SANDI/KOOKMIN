#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
테이프 기반 plumb-line 왜곡 재보정 (robust).

지난 실패 원인 2가지를 고쳤다:
  1) 데이터 정밀도  — 줄눈(1~4px 흔들림) → 검정테이프 밝기가중중심(~0.5px)
  2) 파라미터 폭주 — 접선왜곡을 물리적 범위(|p|<0.005)로 제한 + soft_l1 강건손실

절차: RAW 영상에서 테이프 점 검출 → undistort 후 각 선이 직선이 되도록 cx,cy,k1,k2,p1,p2 최적화
      → 새 K/D 로 격자 교차점 재검출 → H 재계산
사용: python3 _plumb_tape.py [--save]
"""
import argparse
import sys

import numpy as np
import cv2
from scipy.optimize import least_squares

PKG = "/home/xytron/xycar_ws/src/track_drive"
sys.path.insert(0, PKG)
from track_drive.lib import bev as B  # noqa: E402

CFG = PKG + "/config/camera.yaml"
RAWP = PKG + "/dev/shots/tape_raw.jpg"
PITCH = 0.453333
LIDAR_DX = 0.10
ROWS = [1, 2, 3]
COLS = [-1, 1]


def centroid(prof_g, prof_bg, lo, thr=15.0, maxw=26):
    d = prof_bg - prof_g
    idx = np.where(d > thr)[0]
    if len(idx) < 2:
        return None, 0.0
    segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    seg = max(segs, key=len)
    if len(seg) < 2 or len(seg) > maxw:
        return None, 0.0
    w = d[seg]
    return lo + float((seg * w).sum() / w.sum()), float(w.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    cam = B.load_camera_config(CFG)
    K0 = np.array(cam["K"], np.float64)
    D0 = np.array(np.ravel(cam["D"])[:5], np.float64)
    Hm = np.array(cam["H"], np.float64)
    Hi = np.linalg.inv(Hm)
    wref = np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2])
    _nk = cam.get("new_K")
    newK0 = np.array(_nk, np.float64) if _nk is not None else K0.copy()

    raw = cv2.imread(RAWP)
    Hh, Ww = raw.shape[:2]
    g_raw = cv2.GaussianBlur(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.8)
    bg_raw = cv2.GaussianBlur(g_raw, (0, 0), 25)

    # ── 1) RAW 영상에서 테이프 점 검출 (현재 모델로 위치 예측) ──
    mx0, my0 = cv2.initUndistortRectifyMap(K0, D0, np.eye(3), newK0, (Ww, Hh), cv2.CV_32FC1)

    def undist_to_raw(u, v):
        """undist 픽셀 → raw 픽셀 (맵 역참조 대신 직접 투영)"""
        n = np.array([[[(u - newK0[0, 2]) / newK0[0, 0], (v - newK0[1, 2]) / newK0[1, 1]]]])
        X = np.array([[n[0, 0, 0], n[0, 0, 1], 1.0]])
        p, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), K0, D0)
        return float(p[0, 0, 0]), float(p[0, 0, 1])

    def g2undist(x, y):
        q = Hi @ np.array([x - LIDAR_DX, y, 1.0])
        if abs(q[2]) < 1e-12 or np.sign(q[2]) != wref:
            return None
        return q[0] / q[2], q[1] / q[2]

    lines_raw = []
    for i in ROWS:
        pts = []
        for yy in np.linspace(-1.0, 1.0, 80):
            p = g2undist(i * PITCH, yy)
            if p is None:
                continue
            ur, vr = undist_to_raw(*p)
            u = int(round(ur))
            if not (3 < u < Ww - 3 and 20 < vr < Hh - 3):
                continue
            lo = max(0, int(vr - 13)); hi = min(Hh - 1, int(vr + 13))
            c, ct = centroid(g_raw[lo:hi + 1, u - 1:u + 2].mean(axis=1),
                             bg_raw[lo:hi + 1, u - 1:u + 2].mean(axis=1), lo)
            if c is None or ct < 20:
                continue
            pts.append([float(u), c])
        if len(pts) >= 15:
            lines_raw.append(np.array(pts))
            print(f"  가로 i={i}: {len(pts)}점")
    for j in COLS:
        pts = []
        for xx in np.linspace(0.42, 2.75, 100):
            p = g2undist(xx, j * PITCH)
            if p is None:
                continue
            ur, vr = undist_to_raw(*p)
            v = int(round(vr))
            if not (3 < v < Hh - 3 and 3 < ur < Ww - 3):
                continue
            lo = max(0, int(ur - 15)); hi = min(Ww - 1, int(ur + 15))
            c, ct = centroid(g_raw[v - 1:v + 2, lo:hi + 1].mean(axis=0),
                             bg_raw[v - 1:v + 2, lo:hi + 1].mean(axis=0), lo)
            if c is None or ct < 20:
                continue
            pts.append([c, float(v)])
        if len(pts) >= 15:
            lines_raw.append(np.array(pts))
            print(f"  세로 j={j}: {len(pts)}점")

    npts = sum(len(p) for p in lines_raw)
    print(f"\n테이프 선 {len(lines_raw)}개 / 총 {npts}점")
    if len(lines_raw) < 4:
        print("선 부족 — 중단"); return

    # ── 2) 직선성 최적화 (robust + 물리적 범위 제한) ──
    def straight_resid(K, D):
        out = []
        for pts in lines_raw:
            u = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
            c = u - u.mean(0)
            _, _, Vt = np.linalg.svd(c, full_matrices=False)
            out.append(c @ Vt[1])
        return np.concatenate(out)

    def unpack(p):
        K = K0.copy(); K[0, 2], K[1, 2] = p[0], p[1]
        return K, np.array([p[2], p[3], p[4], p[5], 0.0])

    p0 = np.array([K0[0, 2], K0[1, 2], D0[0], D0[1], D0[2], D0[3]])
    lo = np.array([K0[0, 2] - 25, K0[1, 2] - 25, -1.0, -0.6, -0.005, -0.005])
    hi = np.array([K0[0, 2] + 25, K0[1, 2] + 25, 0.0, 0.6, 0.005, 0.005])

    r0 = straight_resid(K0, D0)
    print(f"보정 전 직선성: RMS {np.sqrt((r0**2).mean()):.3f} px / 최대 {np.abs(r0).max():.2f}")

    sol = least_squares(lambda p: straight_resid(*unpack(p)), p0,
                        bounds=(lo, hi), loss="soft_l1", f_scale=1.0, max_nfev=3000)
    K1, D1 = unpack(sol.x)
    r1 = straight_resid(K1, D1)
    print(f"보정 후 직선성: RMS {np.sqrt((r1**2).mean()):.3f} px / 최대 {np.abs(r1).max():.2f}")
    print(f"\ncx {K0[0,2]:.2f} → {K1[0,2]:.2f} ({K1[0,2]-K0[0,2]:+.2f})   "
          f"cy {K0[1,2]:.2f} → {K1[1,2]:.2f} ({K1[1,2]-K0[1,2]:+.2f})")
    print(f"k1 {D0[0]:+.5f} → {D1[0]:+.5f}   k2 {D0[1]:+.5f} → {D1[1]:+.5f}")
    print(f"p1 {D0[2]:+.5f} → {D1[2]:+.5f}   p2 {D0[3]:+.5f} → {D1[3]:+.5f}")

    imp = 1 - np.sqrt((r1 ** 2).mean()) / np.sqrt((r0 ** 2).mean())
    print(f"\n직선성 개선율 {imp*100:.0f}%")
    if imp < 0.15:
        print("개선 미미 — 저장하지 않음 (기존 K/D 유지)"); return

    # ── 3) 새 K/D 로 undistort → 격자 교차점 검출 → H 재계산 ──
    newK1, _ = cv2.getOptimalNewCameraMatrix(K1, D1, (Ww, Hh), 0.0)
    mx, my = cv2.initUndistortRectifyMap(K1, D1, np.eye(3), newK1, (Ww, Hh), cv2.CV_32FC1)
    und = cv2.remap(raw, mx, my, cv2.INTER_LINEAR)
    cv2.imwrite("/home/xytron/plumbtape_und.jpg", und)
    g2 = cv2.GaussianBlur(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.8)
    bg2 = cv2.GaussianBlur(g2, (0, 0), 25)

    # 새 좌표계에서 테이프 선을 다시 뽑아 교차점 계산 (선-선 교점 = 정밀)
    def fit_row(i):
        pts = []
        for yy in np.linspace(-1.0, 1.0, 90):
            p = g2undist(i * PITCH, yy)
            if p is None:
                continue
            ur, vr = undist_to_raw(*p)
            uu, vv = mx.shape[1], None
            # raw→새 undist 는 근사적으로 기존 undist 위치와 유사 → 그 근처 탐색
            u = int(round(p[0]))
            if not (3 < u < Ww - 3 and 20 < p[1] < Hh - 3):
                continue
            lo_ = max(0, int(p[1] - 14)); hi_ = min(Hh - 1, int(p[1] + 14))
            c, ct = centroid(g2[lo_:hi_ + 1, u - 1:u + 2].mean(axis=1),
                             bg2[lo_:hi_ + 1, u - 1:u + 2].mean(axis=1), lo_)
            if c is None or ct < 20:
                continue
            pts.append([float(u), c])
        if len(pts) < 15:
            return None
        P = np.array(pts)
        return np.polyfit(P[:, 0], P[:, 1], 1)      # v = a*u + b

    def fit_col(j):
        pts = []
        for xx in np.linspace(0.42, 2.75, 110):
            p = g2undist(xx, j * PITCH)
            if p is None:
                continue
            v = int(round(p[1]))
            if not (3 < v < Hh - 3 and 3 < p[0] < Ww - 3):
                continue
            lo_ = max(0, int(p[0] - 16)); hi_ = min(Ww - 1, int(p[0] + 16))
            c, ct = centroid(g2[v - 1:v + 2, lo_:hi_ + 1].mean(axis=0),
                             bg2[v - 1:v + 2, lo_:hi_ + 1].mean(axis=0), lo_)
            if c is None or ct < 20:
                continue
            pts.append([c, float(v)])
        if len(pts) < 15:
            return None
        P = np.array(pts)
        return np.polyfit(P[:, 1], P[:, 0], 1)      # u = a*v + b

    rowf = {i: fit_row(i) for i in ROWS}
    colf = {j: fit_col(j) for j in COLS}
    obs, gnd = [], []
    for i, ra in rowf.items():
        if ra is None:
            continue
        for j, ca in colf.items():
            if ca is None:
                continue
            # v = ra0*u + ra1 ,  u = ca0*v + ca1  → 교점
            den = 1 - ra[0] * ca[0]
            if abs(den) < 1e-9:
                continue
            u = (ca[0] * ra[1] + ca[1]) / den
            v = ra[0] * u + ra[1]
            if not (0 <= u < Ww and 0 <= v < Hh):
                continue
            obs.append([u, v]); gnd.append([i * PITCH - LIDAR_DX, j * PITCH])
    print(f"\n테이프 교차점 {len(obs)}개")
    if len(obs) < 4:
        print("교차점 부족 — K/D 만 저장 검토"); return
    obs = np.array(obs); gnd = np.array(gnd)
    Hn, _ = cv2.findHomography(obs, gnd, 0)
    ph = np.hstack([obs, np.ones((len(obs), 1))])
    q = (Hn @ ph.T).T; q = q[:, :2] / q[:, 2:3]
    e = np.linalg.norm(q - gnd, axis=1)
    print(f"새 H: RMS {np.sqrt((e**2).mean())*1000:.1f} mm / 최대 {e.max()*1000:.1f} mm")

    np.savez("/home/xytron/plumbtape.npz", K=K1, D=D1, new_K=newK1, H=Hn)
    print("저장: /home/xytron/plumbtape.npz, plumbtape_und.jpg")

    if a.save:
        import yaml
        with open(CFG, encoding="utf-8") as f:
            y = yaml.safe_load(f)
        y["K"] = [[float(x) for x in r] for r in K1]
        y["D"] = [float(x) for x in D1]
        y["new_K"] = [[float(x) for x in r] for r in newK1]
        y["H"] = [[float(x) for x in r] for r in (Hn / Hn[2, 2])]
        y["h_rms_m"] = float(round(np.sqrt((e ** 2).mean()), 5))
        y["h_frame"] = "lidar_frame"
        y["intrinsics_source"] = "dev/scripts/_plumb_tape.py (tape plumb-line)"
        for k in ("intrinsics_calibrated", "h_calibrated", "calibrated"):
            y[k] = True
        with open(CFG, "w", encoding="utf-8") as f:
            f.write("# 자동생성 — dev/scripts/_plumb_tape.py (검정테이프 직선성 K/D + 교차점 H)\n")
            yaml.safe_dump(y, f, allow_unicode=True, sort_keys=True)
        print("camera.yaml 갱신 완료")


if __name__ == "__main__":
    main()
