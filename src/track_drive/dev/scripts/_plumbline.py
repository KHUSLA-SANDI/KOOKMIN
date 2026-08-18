#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plumb-line 왜곡 캘리브 — "직선은 직선이어야 한다".

바닥 줄눈은 실제로 직선이다. 원본(왜곡) 영상에서 줄눈 위 점을 수백 개 뽑고,
왜곡보정 후 각 줄눈이 '한 직선 위'가 되도록 cx,cy,k1,k2,p1,p2 를 최소제곱으로 맞춘다.
격자 교차점(12개)만 쓰던 방식과 달리 줄눈당 수십 점을 쓰므로 훨씬 안정적이다.
초점거리는 직선성으로 구속되지 않으므로 고정한다.

사용: python3 _plumbline.py [--save]
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
RAW = PKG + "/dev/shots/calib_raw.jpg"
PITCH = 0.453333
LIDAR_DX = 0.10


def sample_line_points(gray, Hi, wref, mode, const, sweep, K, D, half=7, min_ct=6.0):
    """
    mode='row': x=const 고정, y 를 sweep → 가로 줄눈 위 점들
    mode='col': y=const 고정, x 를 sweep → 세로 줄눈 위 점들
    현재 K/D/H 로 예측 위치를 잡고 그 근처 어두운 골을 서브픽셀 검출 (원본 픽셀 반환)
    """
    Hh, Ww = gray.shape
    out = []
    for s in sweep:
        x, y = (const, s) if mode == "row" else (s, const)
        q = Hi @ np.array([x - LIDAR_DX, y, 1.0])
        if abs(q[2]) < 1e-12 or np.sign(q[2]) != wref:
            continue
        uu, vv = q[0] / q[2], q[1] / q[2]
        # undist 예상픽셀 → 원본 픽셀로 되돌리기
        n = np.array([[[(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1]]]], np.float64)
        p, _ = cv2.projectPoints(np.array([[n[0, 0, 0], n[0, 0, 1], 1.0]]),
                                 np.zeros(3), np.zeros(3), K, D)
        u0, v0 = float(p[0, 0, 0]), float(p[0, 0, 1])
        if not (half + 3 < u0 < Ww - half - 3 and half + 3 < v0 < Hh - half - 3):
            continue
        if mode == "row":            # 세로로 훑어 가로선 찾기
            lo = int(round(v0 - half)); hi = int(round(v0 + half))
            prof = gray[lo:hi + 1, int(round(u0)) - 2:int(round(u0)) + 3].mean(axis=1)
            k = int(np.argmin(prof))
            if k <= 0 or k >= len(prof) - 1:
                continue
            a, b, c = prof[k - 1], prof[k], prof[k + 1]
            den = a - 2 * b + c
            d = 0.5 * (a - c) / den if abs(den) > 1e-6 else 0.0
            if prof.max() - prof.min() < min_ct:
                continue
            out.append([u0, lo + k + d])
        else:                        # 가로로 훑어 세로선 찾기
            lo = int(round(u0 - half)); hi = int(round(u0 + half))
            prof = gray[int(round(v0)) - 2:int(round(v0)) + 3, lo:hi + 1].mean(axis=0)
            k = int(np.argmin(prof))
            if k <= 0 or k >= len(prof) - 1:
                continue
            a, b, c = prof[k - 1], prof[k], prof[k + 1]
            den = a - 2 * b + c
            d = 0.5 * (a - c) / den if abs(den) > 1e-6 else 0.0
            if prof.max() - prof.min() < min_ct:
                continue
            out.append([lo + k + d, v0])
    return np.array(out, np.float64) if out else np.zeros((0, 2))


def straightness_resid(lines, K, D):
    """각 선을 undistort 후 직선피팅 → 수직거리 잔차 (픽셀)"""
    res = []
    for pts in lines:
        if len(pts) < 5:
            continue
        u = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
        m = u.mean(0)
        c = u - m
        _, _, Vt = np.linalg.svd(c, full_matrices=False)
        n = Vt[1]                      # 최소분산 방향 = 법선
        res.append(c @ n)
    return np.concatenate(res) if res else np.zeros(0)


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
    newK = np.array(_nk, np.float64) if _nk is not None else K0

    raw = cv2.imread(RAW)
    gray = cv2.GaussianBlur(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32),
                            (0, 0), 1.1)

    def collect(K, D):
        lines = []
        for i in range(1, 8):                    # 가로 줄눈
            pts = sample_line_points(gray, Hi, wref, "row", i * PITCH,
                                     np.linspace(-1.3, 1.3, 45), newK, D)
            if len(pts) >= 6:
                lines.append(pts)
        for j in range(-3, 4):                   # 세로 줄눈
            pts = sample_line_points(gray, Hi, wref, "col", j * PITCH,
                                     np.linspace(0.42, 2.6, 45), newK, D)
            if len(pts) >= 6:
                lines.append(pts)
        return lines

    lines = collect(K0, D0)
    npts = sum(len(p) for p in lines)
    print(f"검출: 줄눈 {len(lines)}개, 총 {npts}점  (교차점 12개 방식 대비 {npts/12:.0f}배)")
    if len(lines) < 4 or npts < 60:
        print("데이터 부족 — 중단"); return

    r0 = straightness_resid(lines, K0, D0)
    print(f"보정 전 직선성 잔차: RMS {np.sqrt((r0**2).mean()):.3f} px, 최대 {np.abs(r0).max():.2f} px")

    def unpack(p):
        K = K0.copy(); K[0, 2], K[1, 2] = p[0], p[1]
        D = np.array([p[2], p[3], p[4], p[5], 0.0])
        return K, D

    def fun(p):
        K, D = unpack(p)
        return straightness_resid(lines, K, D)

    p0 = np.array([K0[0, 2], K0[1, 2], D0[0], D0[1], D0[2], D0[3]])
    sol = least_squares(fun, p0, method="lm", max_nfev=6000)
    K1, D1 = unpack(sol.x)
    r1 = sol.fun
    print(f"보정 후 직선성 잔차: RMS {np.sqrt((r1**2).mean()):.3f} px, 최대 {np.abs(r1).max():.2f} px")
    print(f"\ncx {K0[0,2]:.2f} → {K1[0,2]:.2f}  ({K1[0,2]-K0[0,2]:+.2f})")
    print(f"cy {K0[1,2]:.2f} → {K1[1,2]:.2f}  ({K1[1,2]-K0[1,2]:+.2f})")
    print(f"D  {np.round(D0,5)}\n → {np.round(D1,5)}")

    if np.sqrt((r1 ** 2).mean()) >= np.sqrt((r0 ** 2).mean()) * 0.9:
        print("\n개선이 미미하다 — 기존 K/D 가 이미 충분하다는 뜻. 저장하지 않음.")
        return

    # 새 K/D 로 H 재계산
    newK2, _ = cv2.getOptimalNewCameraMatrix(K1, D1, (640, 480), 0.0)
    mx, my = cv2.initUndistortRectifyMap(K1, D1, np.eye(3), newK2, (640, 480), cv2.CV_32FC1)
    und = cv2.remap(raw, mx, my, cv2.INTER_LINEAR)
    cv2.imwrite("/home/xytron/plumb_und.jpg", und)
    g2 = cv2.GaussianBlur(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.1)

    # 새 undist 좌표계에서 격자 교차점 검출 → H
    obs, gnd = [], []
    for i in range(1, 8):
        for j in (-2, -1, 0, 1, 2):
            q = Hi @ np.array([i * PITCH - LIDAR_DX, j * PITCH, 1.0])
            if abs(q[2]) < 1e-12 or np.sign(q[2]) != wref:
                continue
            u0, v0 = q[0] / q[2], q[1] / q[2]
            if not (10 < u0 < 630 and 10 < v0 < 470):
                continue
            lo = int(v0 - 5); hi = int(v0 + 5)
            pr = g2[lo:hi + 1, int(u0) - 2:int(u0) + 3].mean(axis=1)
            k = int(np.argmin(pr))
            if k <= 0 or k >= len(pr) - 1 or pr.max() - pr.min() < 7:
                continue
            aa, bb, cc = pr[k - 1], pr[k], pr[k + 1]
            dd = 0.5 * (aa - cc) / (aa - 2 * bb + cc) if abs(aa - 2 * bb + cc) > 1e-6 else 0
            vv = lo + k + dd
            lo2 = int(u0 - 10); hi2 = int(u0 + 10)
            if lo2 < 1 or hi2 > 638:
                continue
            pr2 = g2[int(vv) - 2:int(vv) + 3, lo2:hi2 + 1].mean(axis=0)
            k2 = int(np.argmin(pr2))
            if k2 <= 0 or k2 >= len(pr2) - 1 or pr2.max() - pr2.min() < 7:
                continue
            a2, b2, c2 = pr2[k2 - 1], pr2[k2], pr2[k2 + 1]
            d2 = 0.5 * (a2 - c2) / (a2 - 2 * b2 + c2) if abs(a2 - 2 * b2 + c2) > 1e-6 else 0
            obs.append([lo2 + k2 + d2, vv]); gnd.append([i * PITCH - LIDAR_DX, j * PITCH])
    obs = np.array(obs); gnd = np.array(gnd)
    print(f"\n새 좌표계 격자점 {len(obs)}개")
    if len(obs) < 8:
        print("H 재계산용 점 부족 — K/D 만 저장 검토 필요"); return
    Hn, mask = cv2.findHomography(obs, gnd, cv2.RANSAC, 0.015)
    ph = np.hstack([obs, np.ones((len(obs), 1))])
    qq = (Hn @ ph.T).T; qq = qq[:, :2] / qq[:, 2:3]
    e = np.linalg.norm(qq - gnd, axis=1)
    inl = mask.ravel().astype(bool)
    print(f"새 H: RMS {np.sqrt((e[inl]**2).mean())*1000:.1f} mm / 최대 {e[inl].max()*1000:.1f} mm "
          f"(inlier {inl.sum()}/{len(e)})")

    np.savez("/home/xytron/plumb_calib.npz", K=K1, D=D1, new_K=newK2, H=Hn)
    print("저장: /home/xytron/plumb_calib.npz, plumb_und.jpg")

    if a.save:
        import yaml
        with open(CFG, encoding="utf-8") as f:
            y = yaml.safe_load(f)
        y["K"] = [[float(x) for x in r] for r in K1]
        y["D"] = [float(x) for x in D1]
        y["new_K"] = [[float(x) for x in r] for r in newK2]
        y["H"] = [[float(x) for x in r] for r in (Hn / Hn[2, 2])]
        y["h_rms_m"] = float(round(np.sqrt((e[inl] ** 2).mean()), 5))
        y["h_frame"] = "lidar_frame"
        y["intrinsics_source"] = "dev/scripts/_plumbline.py (plumb-line + tile grid)"
        for k in ("intrinsics_calibrated", "h_calibrated", "calibrated"):
            y[k] = True
        with open(CFG, "w", encoding="utf-8") as f:
            f.write("# 자동생성 — dev/scripts/_plumbline.py (직선성 기반 K/D + 타일격자 H)\n")
            yaml.safe_dump(y, f, allow_unicode=True, sort_keys=True)
        print("camera.yaml 갱신 완료")


if __name__ == "__main__":
    main()
