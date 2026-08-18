#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K/D + 카메라자세 동시 최적화 (단일 뷰 번들조정).

바닥 타일 격자는 그 자체가 평면 캘리브 타깃이다. 원본(왜곡) 이미지에서 줄눈 교차점을
서브픽셀로 검출하고, 그 점들이 (i*p, j*p) 라는 사실을 이용해
  cx, cy, k1, k2, p1, p2, 카메라 자세(rvec,tvec)
를 동시에 최소제곱으로 맞춘다. BEV 에서 직선이 휘는 잔여 왜곡을 잡는 것이 목적.

사용: python3 _joint_calib.py [--free-f] [--save]
"""
import argparse
import os
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
LIDAR_DX = 0.10        # 라이다가 A' 보다 앞선 거리 (H 저장 시 반영)


def detect_grid_points(img_gray, K, D, rvec, tvec, ij_list, half=8, min_ct=7.0):
    """현재 파라미터로 예측한 위치 근처에서 줄눈 교차점을 서브픽셀 검출."""
    obs, objp, tags = [], [], []
    Hh, Ww = img_gray.shape
    for (i, j) in ij_list:
        X = np.array([[i * PITCH, j * PITCH, 0.0]], np.float64)
        p, _ = cv2.projectPoints(X, rvec, tvec, K, D)
        u0, v0 = float(p[0, 0, 0]), float(p[0, 0, 1])
        if not (half + 3 < u0 < Ww - half - 3 and half + 3 < v0 < Hh - half - 3):
            continue
        # 세로 방향(가로 줄눈) 서브픽셀
        lo, hi = int(round(v0 - half)), int(round(v0 + half))
        col = img_gray[lo:hi + 1, int(round(u0)) - 2:int(round(u0)) + 3].mean(axis=1)
        kv = int(np.argmin(col))
        if kv <= 0 or kv >= len(col) - 1:
            continue
        a, b, c = col[kv - 1], col[kv], col[kv + 1]
        den = a - 2 * b + c
        dv = 0.5 * (a - c) / den if abs(den) > 1e-6 else 0.0
        ctv = float(col.max() - col.min())
        vv = lo + kv + dv
        # 가로 방향(세로 줄눈) 서브픽셀
        lo2, hi2 = int(round(u0 - half * 2)), int(round(u0 + half * 2))
        if lo2 < 1 or hi2 >= Ww - 1:
            continue
        row = img_gray[int(round(vv)) - 2:int(round(vv)) + 3, lo2:hi2 + 1].mean(axis=0)
        ku = int(np.argmin(row))
        if ku <= 0 or ku >= len(row) - 1:
            continue
        a2, b2, c2 = row[ku - 1], row[ku], row[ku + 1]
        den2 = a2 - 2 * b2 + c2
        du = 0.5 * (a2 - c2) / den2 if abs(den2) > 1e-6 else 0.0
        ctu = float(row.max() - row.min())
        uu = lo2 + ku + du
        if ctv < min_ct or ctu < min_ct:
            continue
        obs.append([uu, vv]); objp.append([i * PITCH, j * PITCH, 0.0])
        tags.append(f"({i},{j})")
    return np.array(obs, np.float64), np.array(objp, np.float64), tags


def pack(K, D, rvec, tvec, free_f):
    p = [K[0, 2], K[1, 2], D[0], D[1], D[2], D[3]]
    if free_f:
        p = [K[0, 0], K[1, 1]] + p
    return np.array(p + list(rvec.ravel()) + list(tvec.ravel()))


def unpack(p, K0, free_f):
    K = K0.copy()
    k = 0
    if free_f:
        K[0, 0], K[1, 1] = p[0], p[1]; k = 2
    K[0, 2], K[1, 2] = p[k], p[k + 1]
    D = np.array([p[k + 2], p[k + 3], p[k + 4], p[k + 5], 0.0])
    rvec = p[k + 6:k + 9].reshape(3, 1)
    tvec = p[k + 9:k + 12].reshape(3, 1)
    return K, D, rvec, tvec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--free-f", action="store_true", help="초점거리도 최적화")
    ap.add_argument("--save", action="store_true", help="camera.yaml 에 반영")
    a = ap.parse_args()

    cam = B.load_camera_config(CFG)
    K0 = np.array(cam["K"], np.float64)
    D0 = np.array(np.ravel(cam["D"])[:5], np.float64)
    print("초기 K: fx=%.2f fy=%.2f cx=%.2f cy=%.2f" % (K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]))
    print("초기 D:", np.round(D0, 5))

    raw = cv2.imread(RAW)
    gray = cv2.GaussianBlur(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.1)

    # 초기 자세: 현재 H 로부터 (H: undistort픽셀→지면). 지면→undist픽셀 로 PnP
    Hm = np.array(cam["H"], np.float64)
    Hi = np.linalg.inv(Hm)
    _nk = cam.get("new_K")
    newK = np.array(_nk, np.float64) if _nk is not None else K0.copy()
    ij_all = [(i, j) for i in range(1, 8) for j in (-2, -1, 0, 1, 2)]
    # H 로 예측한 undist 픽셀 → PnP 로 초기 rvec/tvec (undist 는 무왜곡이므로 D=0)
    objs, imgs = [], []
    for (i, j) in ij_all:
        g = np.array([i * PITCH - LIDAR_DX, j * PITCH, 1.0])   # H 는 lidar_frame 기준
        q = Hi @ g
        if abs(q[2]) < 1e-9:
            continue
        u, v = q[0] / q[2], q[1] / q[2]
        if 0 <= u < 640 and 0 <= v < 480:
            objs.append([i * PITCH, j * PITCH, 0.0]); imgs.append([u, v])
    objs = np.array(objs, np.float64); imgs = np.array(imgs, np.float64)
    ok, rvec, tvec = cv2.solvePnP(objs, imgs, newK, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        print("초기 자세 추정 실패"); return
    print("초기 자세: 카메라높이 %.3f m" % abs(float(tvec[2])))

    # ── 반복: 검출 → 최적화 ──
    K, D = K0.copy(), D0.copy()
    for it in range(3):
        obs, objp, tags = detect_grid_points(gray, K, D, rvec, tvec, ij_all,
                                             half=[9, 6, 4][it])
        if len(obs) < 12:
            print(f"[{it}] 검출 부족 {len(obs)}"); break

        def resid(p):
            Kx, Dx, rx, tx = unpack(p, K0, a.free_f)
            pr, _ = cv2.projectPoints(objp, rx, tx, Kx, Dx)
            return (pr.reshape(-1, 2) - obs).ravel()

        p0 = pack(K, D, rvec, tvec, a.free_f)
        r0 = resid(p0)
        sol = least_squares(resid, p0, method="lm", max_nfev=4000)
        K, D, rvec, tvec = unpack(sol.x, K0, a.free_f)
        e0 = np.sqrt((r0.reshape(-1, 2) ** 2).sum(1))
        e1 = np.sqrt((sol.fun.reshape(-1, 2) ** 2).sum(1))
        print(f"[{it}차] 점 {len(obs)}개  재투영 RMS {e0.std():.3f} → {np.sqrt((e1**2).mean()):.3f} px "
              f"(최대 {e1.max():.2f})")

    print("\n최적화 K: fx=%.2f fy=%.2f cx=%.2f cy=%.2f" % (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    print("최적화 D:", np.round(D, 5))
    print("변화량: cx %+.2f, cy %+.2f px" % (K[0, 2] - K0[0, 2], K[1, 2] - K0[1, 2]))
    print("카메라 높이 %.3f m" % abs(float(tvec[2])))

    # ── 새 K/D 로 undistort 하고, 그 좌표계에서 H 재계산 ──
    newK2, roi = cv2.getOptimalNewCameraMatrix(K, D, (640, 480), 0.0)
    mx, my = cv2.initUndistortRectifyMap(K, D, np.eye(3), newK2, (640, 480), cv2.CV_32FC1)
    und = cv2.remap(raw, mx, my, cv2.INTER_LINEAR)
    cv2.imwrite("/home/xytron/joint_und.jpg", und)

    # 지면점 → 새 undist 픽셀 (D=0, K=newK2)
    obs2, objp2, _ = detect_grid_points(
        cv2.GaussianBlur(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.1),
        newK2, np.zeros(5), rvec, tvec, ij_all, half=4)
    print(f"\n새 좌표계 검출 {len(obs2)}점")
    gnd = objp2[:, :2].copy()
    gnd[:, 0] -= LIDAR_DX                      # A' → lidar_frame
    Hnew, mask = cv2.findHomography(obs2, gnd, cv2.RANSAC, 0.015)
    ph = np.hstack([obs2, np.ones((len(obs2), 1))])
    q = (Hnew @ ph.T).T; q = q[:, :2] / q[:, 2:3]
    e = np.linalg.norm(q - gnd, axis=1)
    inl = mask.ravel().astype(bool)
    print(f"새 H: RMS {np.sqrt((e[inl]**2).mean())*1000:.1f} mm / 최대 {e[inl].max()*1000:.1f} mm "
          f"(inlier {inl.sum()}/{len(e)})")

    np.savez("/home/xytron/joint_calib.npz", K=K, D=D, new_K=newK2, H=Hnew,
             rvec=rvec, tvec=tvec)
    print("저장: /home/xytron/joint_calib.npz, joint_und.jpg")

    if a.save:
        import yaml
        with open(CFG, encoding="utf-8") as f:
            y = yaml.safe_load(f)
        y["K"] = [[float(x) for x in r] for r in K]
        y["D"] = [float(x) for x in D]
        y["new_K"] = [[float(x) for x in r] for r in newK2]
        y["H"] = [[float(x) for x in r] for r in (Hnew / Hnew[2, 2])]
        y["h_rms_m"] = float(round(np.sqrt((e[inl] ** 2).mean()), 5))
        y["h_frame"] = "lidar_frame"
        y["intrinsics_calibrated"] = True
        y["h_calibrated"] = True
        y["calibrated"] = True
        y["intrinsics_source"] = "dev/scripts/_joint_calib.py (tile grid bundle adjust)"
        with open(CFG, "w", encoding="utf-8") as f:
            f.write("# 자동생성 — dev/scripts/_joint_calib.py (타일격자 K/D+H 동시최적화)\n")
            yaml.safe_dump(y, f, allow_unicode=True, sort_keys=True)
        print("camera.yaml 갱신 완료")


if __name__ == "__main__":
    main()
