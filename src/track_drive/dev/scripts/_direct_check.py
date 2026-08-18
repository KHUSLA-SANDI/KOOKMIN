#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H 를 전혀 쓰지 않고 보정영상에서 테이프 직선성을 직접 확인 (순환논리 배제).
각 열(column)에서 어두운 띠 중심을 찾아 선을 추적하고, 양 끝점을 잇는 현(chord) 대비
최대 편차를 측정 + 확대 렌더.
"""
import sys
import numpy as np, cv2

PKG = "/home/xytron/xycar_ws/src/track_drive"
sys.path.insert(0, PKG)
from track_drive.lib import bev as B

cam = B.load_camera_config(PKG + "/config/camera.yaml")
raw = cv2.imread(PKG + "/dev/shots/tape2_raw.jpg")
und = B.Undistorter.from_config(cam, (raw.shape[1], raw.shape[0]))
img = und.apply(raw)
Hh, Ww = img.shape[:2]
g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.8)
bgm = cv2.GaussianBlur(g, (0, 0), 25)
dark = (bgm - g)          # 클수록 어두운 띠

vis = img.copy()


def trace_row(v_seed, u_from, u_to, win=10):
    """v_seed 근처에서 시작해 좌우로 어두운 띠를 따라가며 추적"""
    pts = []
    v_cur = float(v_seed)
    for u in range(u_from, u_to):
        lo = int(max(0, v_cur - win)); hi = int(min(Hh - 1, v_cur + win))
        if hi - lo < 4:
            break
        prof = dark[lo:hi + 1, max(0, u - 1):u + 2].mean(axis=1)
        k = int(np.argmax(prof))
        if prof[k] < 14:
            continue
        if k == 0 or k == len(prof) - 1:
            continue
        a, b, c = prof[k - 1], prof[k], prof[k + 1]
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den if abs(den) > 1e-6 else 0.0
        v_sub = lo + k + d
        pts.append((u, v_sub))
        v_cur = v_sub
    return np.array(pts) if pts else np.zeros((0, 2))


# 아래쪽부터 어두운 띠가 있는 행(seed) 자동 탐색 (중앙 열 기준)
col = dark[:, 300:340].mean(axis=1)
seeds = []
for v in range(250, Hh - 5):
    if col[v] > 18 and col[v] >= col[max(0, v - 3):v + 4].max():
        if not seeds or v - seeds[-1] > 8:
            seeds.append(v)
print("중앙열에서 찾은 어두운 띠 행:", seeds)

print(f"\n{'seed v':>7} {'점수':>5} {'u범위':>13} {'현 대비 최대편차':>16} {'RMS':>8}")
print("-" * 58)
for sv in seeds:
    # 중앙에서 좌우로 각각 추적
    left = trace_row(sv, 320, 5, win=9)[::-1]
    right = trace_row(sv, 320, Ww - 5, win=9)
    P = np.vstack([left, right]) if len(left) and len(right) else (left if len(left) else right)
    if len(P) < 40:
        continue
    P = P[np.argsort(P[:, 0])]
    # 양 끝점을 잇는 현
    x0, y0 = P[0]; x1, y1 = P[-1]
    chord = y0 + (P[:, 0] - x0) * (y1 - y0) / max(1e-6, (x1 - x0))
    dev = P[:, 1] - chord
    print(f"{sv:7d} {len(P):5d} {int(P[0,0]):5d}~{int(P[-1,0]):4d} "
          f"{np.abs(dev).max():16.2f} px {dev.std():7.2f}")
    for u_, v_ in P:
        cv2.circle(vis, (int(u_), int(round(v_))), 1, (0, 0, 255), -1)
    cv2.line(vis, (int(x0), int(round(y0))), (int(x1), int(round(y1))), (0, 255, 255), 1)

cv2.imwrite("/home/xytron/direct_check.jpg", vis)
z = cv2.resize(vis[250:480, :, :], None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
cv2.imwrite("/home/xytron/direct_zoom.jpg", z)
print("\nsaved: direct_check.jpg, direct_zoom.jpg")
print("빨강=추적점, 노랑=양끝 직선. 빨강이 노랑에서 벗어나면 그만큼 안 곧은 것.")
