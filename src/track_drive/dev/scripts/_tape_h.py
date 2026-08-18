#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
테이프 선 교점 기반 H 캘리브 (최종 방식).

개별 점을 클릭/검출해 쓰면 점 하나의 흔들림이 그대로 오차가 된다.
대신 각 테이프 선을 수십 점으로 로버스트 피팅하고, 선-선 교점을 계산해 대응점으로 쓴다.
→ 점 노이즈가 √N 으로 평균화되고, 교점은 서브픽셀 정밀도를 얻는다.

가로 테이프 i=1,2,3 (x = i*PITCH) · 세로 테이프 j=-2,-1,+1,+2 (y = j*PITCH)
사용: python3 _tape_h.py [--image ...] [--save]
"""
import argparse
import sys

import numpy as np
import cv2

PKG = "/home/xytron/xycar_ws/src/track_drive"
sys.path.insert(0, PKG)
from track_drive.lib import bev as B  # noqa: E402

CFG = PKG + "/config/camera.yaml"
PITCH = 0.453333
LIDAR_DX = 0.10
ROWS = [1, 2, 3]
COLS = [-2, -1, 1, 2]


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


def robust_fit(X, Y, deg=1, iters=3):
    """3σ 반복제거 다항피팅"""
    m = np.ones(len(X), bool)
    A = None
    for _ in range(iters):
        if m.sum() < deg + 3:
            break
        A = np.polyfit(X[m], Y[m], deg)
        r = Y - np.polyval(A, X)
        s = 1.4826 * np.median(np.abs(r[m] - np.median(r[m]))) + 1e-9
        m = np.abs(r - np.median(r[m])) < 3 * s
    return A, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=PKG + "/dev/shots/tape2_raw.jpg")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    cam = B.load_camera_config(CFG)
    raw = cv2.imread(a.image)
    if raw is None:
        print("이미지 없음:", a.image); return
    Hh, Ww = raw.shape[:2]
    und = B.Undistorter.from_config(cam, (Ww, Hh))
    img = und.apply(raw)
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.8)
    bgm = cv2.GaussianBlur(g, (0, 0), 25)

    Hi = np.linalg.inv(np.array(cam["H"], np.float64))
    wref = np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2])

    def gu(x, y):
        q = Hi @ np.array([x - LIDAR_DX, y, 1.0])
        if abs(q[2]) < 1e-12 or np.sign(q[2]) != wref:
            return None
        u, v = q[0] / q[2], q[1] / q[2]
        return (u, v) if (2 <= u < Ww - 2 and 2 <= v < Hh - 2) else None

    vis = img.copy()
    rowfit, colfit = {}, {}

    print("=== 가로 테이프 (v = a*u + b) ===")
    for i in ROWS:
        P = []
        for yy in np.linspace(-1.15, 1.15, 110):
            p = gu(i * PITCH, yy)
            if p is None:
                continue
            u = int(round(p[0]))
            lo = max(0, int(p[1] - 14)); hi = min(Hh - 1, int(p[1] + 14))
            if hi - lo < 6 or u < 2 or u >= Ww - 2:
                continue
            c, ct = cen(g[lo:hi + 1, u - 1:u + 2].mean(1), bgm[lo:hi + 1, u - 1:u + 2].mean(1), lo)
            if c is not None and ct >= 20:
                P.append([float(u), c])
        if len(P) < 15:
            print(f"  i={i}: 점 부족 {len(P)}"); continue
        P = np.array(P)
        A, m = robust_fit(P[:, 0], P[:, 1])
        r = P[m, 1] - np.polyval(A, P[m, 0])
        rowfit[i] = A
        print(f"  i={i} (x={i*PITCH:.3f}m): {len(P)}점 → inlier {int(m.sum())}, "
              f"잔차 RMS {r.std():.2f}px, u {P[:,0].min():.0f}~{P[:,0].max():.0f}")
        for u_, v_ in P[m]:
            cv2.circle(vis, (int(u_), int(round(v_))), 1, (0, 0, 255), -1)
        xs = np.linspace(P[:, 0].min(), P[:, 0].max(), 60)
        cv2.polylines(vis, [np.stack([xs, np.polyval(A, xs)], 1).astype(np.int32)],
                      False, (0, 255, 255), 1)

    print("\n=== 세로 테이프 (u = a*v + b) ===")
    for j in COLS:
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
        if len(P) < 15:
            print(f"  j={j}: 점 부족 {len(P)}"); continue
        P = np.array(P)
        A, m = robust_fit(P[:, 1], P[:, 0])
        r = P[m, 0] - np.polyval(A, P[m, 1])
        colfit[j] = A
        print(f"  j={j} (y={j*PITCH:+.3f}m): {len(P)}점 → inlier {int(m.sum())}, "
              f"잔차 RMS {r.std():.2f}px, v {P[:,1].min():.0f}~{P[:,1].max():.0f}")
        for u_, v_ in P[m]:
            cv2.circle(vis, (int(round(u_)), int(v_)), 1, (255, 0, 255), -1)
        ys = np.linspace(P[:, 1].min(), P[:, 1].max(), 60)
        cv2.polylines(vis, [np.stack([np.polyval(A, ys), ys], 1).astype(np.int32)],
                      False, (255, 255, 0), 1)

    # ── 교점 ──
    obs, gnd, tags = [], [], []
    for i, ra in rowfit.items():
        for j, ca in colfit.items():
            den = 1 - ra[0] * ca[0]
            if abs(den) < 1e-9:
                continue
            u = (ca[0] * ra[1] + ca[1]) / den
            v = ra[0] * u + ra[1]
            if not (-30 <= u < Ww + 30 and -30 <= v < Hh + 30):
                continue
            obs.append([u, v]); gnd.append([i * PITCH - LIDAR_DX, j * PITCH])
            tags.append(f"({i},{j:+d})")
            if 0 <= u < Ww and 0 <= v < Hh:
                cv2.drawMarker(vis, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 13, 2)
    print(f"\n교점 {len(obs)}개: {' '.join(tags)}")
    if len(obs) < 4:
        print("교점 부족 — 중단"); return

    obs = np.array(obs); gnd = np.array(gnd)
    Hn, _ = cv2.findHomography(obs, gnd, 0)
    ph = np.hstack([obs, np.ones((len(obs), 1))])
    q = (Hn @ ph.T).T; q = q[:, :2] / q[:, 2:3]
    e = np.linalg.norm(q - gnd, axis=1)
    print(f"\n새 H (테이프 교점 {len(obs)}개):")
    print(f"  RMS {np.sqrt((e**2).mean())*1000:.1f} mm / 최대 {e.max()*1000:.1f} mm")
    for t, ee in sorted(zip(tags, e), key=lambda z: -z[1])[:5]:
        print(f"    {t}: {ee*1000:6.1f} mm")

    old = float(cam.get("h_rms_m") or 9.9)
    print(f"\n기존 H: RMS {old*1000:.1f} mm  →  새 H: {np.sqrt((e**2).mean())*1000:.1f} mm")

    cv2.imwrite("/home/xytron/tape_h_fit.jpg", vis)
    np.save("/home/xytron/H_tape.npy", Hn)
    print("저장: tape_h_fit.jpg, H_tape.npy")

    if a.save:
        import yaml
        with open(CFG, encoding="utf-8") as f:
            y = yaml.safe_load(f)
        y["H"] = [[float(x) for x in r] for r in (Hn / Hn[2, 2])]
        y["h_rms_m"] = float(round(np.sqrt((e ** 2).mean()), 5))
        y["h_frame"] = "lidar_frame"
        y["h_source"] = "dev/scripts/_tape_h.py (tape line intersections)"
        for k in ("h_calibrated", "calibrated"):
            y[k] = True
        with open(CFG, "w", encoding="utf-8") as f:
            f.write("# 자동생성 — dev/scripts/_tape_h.py (검정테이프 선교점 H)\n")
            yaml.safe_dump(y, f, allow_unicode=True, sort_keys=True)
        print("camera.yaml 갱신 완료")


if __name__ == "__main__":
    main()
