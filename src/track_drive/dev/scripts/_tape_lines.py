#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델 기반 테이프 선 추출 + 직선성 측정.
현재 H 로 각 테이프 선(가로 i=1..3, 세로 j=±1)의 예상 경로를 그리고,
그 경로 주변에서 어두운 띠의 '밝기 가중 중심'을 서브픽셀로 뽑는다.
왜곡보정 영상에서 이 점들이 직선이 아니면 그 편차가 곧 잔여 왜곡이다.
"""
import sys
import numpy as np, cv2

PKG = "/home/xytron/xycar_ws/src/track_drive"
sys.path.insert(0, PKG)
from track_drive.lib import bev as B

PITCH = 0.453333
LIDAR_DX = 0.10
ROWS = [1, 2, 3]            # 가로 테이프 (전방)
COLS = [-1, 1]              # 세로 테이프 (좌우 1칸)

cam = B.load_camera_config(PKG + "/config/camera.yaml")
raw = cv2.imread(PKG + "/dev/shots/tape_raw.jpg")
und = B.Undistorter.from_config(cam, (raw.shape[1], raw.shape[0]))
img = und.apply(raw)
gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.8)
Hh, Ww = gray.shape
bg = cv2.GaussianBlur(gray, (0, 0), 25)

Hm = np.array(cam["H"], np.float64)
Hi = np.linalg.inv(Hm)
wref = np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2])


def g2p(x, y):
    q = Hi @ np.array([x - LIDAR_DX, y, 1.0])
    if abs(q[2]) < 1e-12 or np.sign(q[2]) != wref:
        return None
    u, v = q[0] / q[2], q[1] / q[2]
    return (u, v) if (2 <= u < Ww - 2 and 2 <= v < Hh - 2) else None


def centroid(prof_g, prof_bg, lo, thr=15.0):
    """어두운 띠의 밝기 가중 중심 (서브픽셀). 없으면 None"""
    d = prof_bg - prof_g
    idx = np.where(d > thr)[0]
    if len(idx) < 2:
        return None, 0.0, 0
    segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    seg = max(segs, key=len)
    if len(seg) < 2:
        return None, 0.0, 0
    w = d[seg]
    return lo + float((seg * w).sum() / w.sum()), float(w.max()), len(seg)


def collect_row(i, half=14):
    """가로 테이프: x=i*PITCH 고정, y 를 훑으며 각 u 에서 세로로 띠 중심"""
    pts = []
    for yy in np.linspace(-1.05, 1.05, 90):
        p = g2p(i * PITCH, yy)
        if p is None:
            continue
        u = int(round(p[0])); v0 = p[1]
        lo = max(0, int(v0 - half)); hi = min(Hh - 1, int(v0 + half))
        if hi - lo < 5 or u < 2 or u >= Ww - 2:
            continue
        c, ct, wd = centroid(gray[lo:hi + 1, u - 1:u + 2].mean(axis=1),
                             bg[lo:hi + 1, u - 1:u + 2].mean(axis=1), lo)
        if c is None or ct < 18 or wd > 30:
            continue
        pts.append((float(u), c, ct, wd))
    return pts


def collect_col(j, half=16):
    """세로 테이프: y=j*PITCH 고정, x 를 훑으며 각 v 에서 가로로 띠 중심"""
    pts = []
    for xx in np.linspace(0.42, 2.75, 110):
        p = g2p(xx, j * PITCH)
        if p is None:
            continue
        v = int(round(p[1])); u0 = p[0]
        lo = max(0, int(u0 - half)); hi = min(Ww - 1, int(u0 + half))
        if hi - lo < 5 or v < 2 or v >= Hh - 2:
            continue
        c, ct, wd = centroid(gray[v - 1:v + 2, lo:hi + 1].mean(axis=0),
                             bg[v - 1:v + 2, lo:hi + 1].mean(axis=0), lo)
        if c is None or ct < 18 or wd > 30:
            continue
        pts.append((c, float(v), ct, wd))
    return pts


vis = img.copy()
print("=== 가로 테이프 (직선이어야 함) ===")
all_lines = []
for i in ROWS:
    pts = collect_row(i)
    if len(pts) < 15:
        print(f"  i={i}: 점 부족 ({len(pts)})"); continue
    us = np.array([p[0] for p in pts]); vs = np.array([p[1] for p in pts])
    A1 = np.polyfit(us, vs, 1); r1 = vs - np.polyval(A1, us)
    A2 = np.polyfit(us, vs, 2); r2 = vs - np.polyval(A2, us)
    print(f"  i={i} (x={i*PITCH:.3f}m)  점 {len(us):3d}  u {us.min():.0f}~{us.max():.0f}  "
          f"평균폭 {np.mean([p[3] for p in pts]):.1f}px")
    print(f"      직선 잔차 RMS {r1.std():5.2f}px (최대 {np.abs(r1).max():5.2f})  |  "
          f"2차 잔차 {r2.std():5.2f}px  휨계수 {A2[0]*1e4:+6.2f}e-4  "
          f"→ 중앙-끝 처짐 {A2[0]*(us.ptp()/2)**2:+.1f}px")
    all_lines.append(np.stack([us, vs], 1))
    for u_, v_ in zip(us, vs):
        cv2.circle(vis, (int(u_), int(round(v_))), 1, (0, 0, 255), -1)
    xs = np.linspace(us.min(), us.max(), 60)
    cv2.polylines(vis, [np.stack([xs, np.polyval(A1, xs)], 1).astype(np.int32)],
                  False, (0, 255, 255), 1)

print("\n=== 세로 테이프 (직선이어야 함) ===")
for j in COLS:
    pts = collect_col(j)
    if len(pts) < 15:
        print(f"  j={j}: 점 부족 ({len(pts)})"); continue
    us = np.array([p[0] for p in pts]); vs = np.array([p[1] for p in pts])
    A1 = np.polyfit(vs, us, 1); r1 = us - np.polyval(A1, vs)
    A2 = np.polyfit(vs, us, 2); r2 = us - np.polyval(A2, vs)
    print(f"  j={j} (y={j*PITCH:+.3f}m)  점 {len(vs):3d}  v {vs.min():.0f}~{vs.max():.0f}  "
          f"평균폭 {np.mean([p[3] for p in pts]):.1f}px")
    print(f"      직선 잔차 RMS {r1.std():5.2f}px (최대 {np.abs(r1).max():5.2f})  |  "
          f"2차 잔차 {r2.std():5.2f}px  휨계수 {A2[0]*1e4:+6.2f}e-4  "
          f"→ 중앙-끝 처짐 {A2[0]*(vs.ptp()/2)**2:+.1f}px")
    all_lines.append(np.stack([us, vs], 1))
    for u_, v_ in zip(us, vs):
        cv2.circle(vis, (int(round(u_)), int(v_)), 1, (255, 0, 255), -1)
    ys = np.linspace(vs.min(), vs.max(), 60)
    cv2.polylines(vis, [np.stack([np.polyval(A1, ys), ys], 1).astype(np.int32)],
                  False, (255, 255, 0), 1)

cv2.imwrite("/home/xytron/tape_fit.jpg", vis)
np.save("/home/xytron/tape_lines.npy", np.array(all_lines, dtype=object), allow_pickle=True)
print("\nsaved: tape_fit.jpg, tape_lines.npy")
print("\n판정: '중앙-끝 처짐' 이 ±1px 이내면 왜곡보정 양호. 3px 이상이면 K/D 재보정 필요.")
