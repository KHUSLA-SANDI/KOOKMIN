#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
세로 테이프만으로 물리제약 H 결정 (최종 방식).

바닥의 가로 줄눈은 실측상 안 곧다(잔차 6px≈3.7cm) — 캘리브 기준으로 쓰면 안 된다.
세로 줄눈은 0.4px 로 곧으므로 이것만 쓴다.

제약: 세로 테이프 j 위의 모든 점은 지면에서 y = j*PITCH 여야 한다.
미지수: height, pitch, yaw, roll, cam_y  (cam_x 는 원점 정의값 -0.10 고정)
→ 파라미터 5개 vs 관측 200+개. 과적합 불가, 물리적으로 항상 타당.
"""
import sys
import numpy as np, cv2
from scipy.optimize import least_squares

PKG = "/home/xytron/xycar_ws/src/track_drive"
sys.path.insert(0, PKG)
from track_drive.lib import bev as B

PITCH, LDX = 0.453333, 0.10
cam = B.load_camera_config(PKG + "/config/camera.yaml")
K0 = np.array(cam["K"], np.float64)
D0 = np.array(np.ravel(cam["D"])[:5], np.float64)
K = np.array(cam["new_K"] if cam.get("new_K") is not None else K0, np.float64)
Hi0 = np.linalg.inv(np.array(cam["H"], np.float64))
wref0 = np.sign((Hi0 @ np.array([1.0, 0.0, 1.0]))[2])

raw = cv2.imread(PKG + "/dev/shots/tape2_raw.jpg")
Hh, Ww = raw.shape[:2]
und = B.Undistorter.from_config(cam, (Ww, Hh))
img = und.apply(raw)
g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.8)
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


def gu0(x, y):
    q = Hi0 @ np.array([x - LDX, y, 1.0])
    if abs(q[2]) < 1e-12 or np.sign(q[2]) != wref0:
        return None
    u, v = q[0] / q[2], q[1] / q[2]
    return (u, v) if (2 <= u < Ww - 2 and 2 <= v < Hh - 2) else None


# ── 세로 테이프 점 수집 ──
obs, lab = [], []
for j in (-1, 1, 2):
    n = 0
    for xx in np.linspace(0.40, 2.9, 150):
        p = gu0(xx, j * PITCH)
        if p is None:
            continue
        v = int(round(p[1]))
        lo = max(0, int(p[0] - 16)); hi = min(Ww - 1, int(p[0] + 16))
        if hi - lo < 6 or v < 2 or v >= Hh - 2:
            continue
        c, ct = cen(g[v - 1:v + 2, lo:hi + 1].mean(0), bgm[v - 1:v + 2, lo:hi + 1].mean(0), lo)
        if c is not None and ct >= 20:
            obs.append([c, float(v)]); lab.append(j); n += 1
    print(f"  세로 j={j:+d}: {n}점")
obs = np.array(obs); lab = np.array(lab)
print(f"총 {len(obs)}점")


def build_H(h, pit, yaw, roll, cy_off):
    th, ps, rl = np.radians([pit, yaw, roll])
    R0 = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    Rp = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    Ry = np.array([[np.cos(ps), -np.sin(ps), 0], [np.sin(ps), np.cos(ps), 0], [0, 0, 1]])
    Rr = np.array([[np.cos(rl), -np.sin(rl), 0], [np.sin(rl), np.cos(rl), 0], [0, 0, 1]])
    R = Rr @ Rp @ R0 @ Ry
    t = -R @ np.array([-LDX, cy_off, h])
    return np.linalg.inv(K @ np.column_stack([R[:, 0], R[:, 1], t]))


ph = np.hstack([obs, np.ones((len(obs), 1))])
tgt_y = lab * PITCH


def resid(p):
    Hm = build_H(*p)
    q = (Hm @ ph.T).T
    w = q[:, 2]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    y = q[:, 1] / w
    return y - tgt_y


x0 = np.array([0.25, 2.0, 7.4, 0.0, 0.0])
sol = least_squares(resid, x0, method="lm", max_nfev=8000)
h, pit, yaw, roll, cyo = sol.x
r = resid(sol.x)
print(f"\n=== 세로선 전용 물리 피팅 ===")
print(f"  height = {h:.4f} m")
print(f"  pitch  = {pit:+.2f}°   (소실점 추정 +1.97°)")
print(f"  yaw    = {yaw:+.2f}°   (소실점 추정 +7.45°)")
print(f"  roll   = {roll:+.2f}°")
print(f"  cam_y  = {cyo:+.4f} m")
print(f"  횡방향 잔차 RMS {np.sqrt((r**2).mean())*1000:.1f} mm / 최대 {np.abs(r).max()*1000:.1f} mm")

Hn = build_H(*sol.x)

# 검증: 가로 테이프 위치가 어디로 나오나 (독립 확인)
print(f"\n=== 검증: 가로 테이프의 예측 x (실제는 {PITCH:.3f} 배수 - {LDX}) ===")
for i in (1, 2, 3):
    P = []
    for yy in np.linspace(-0.9, 0.9, 60):
        p = gu0(i * PITCH, yy)
        if p is None:
            continue
        u = int(round(p[0]))
        lo = max(0, int(p[1] - 13)); hi = min(Hh - 1, int(p[1] + 13))
        if hi - lo < 6 or u < 2 or u >= Ww - 2:
            continue
        c, ct = cen(g[lo:hi + 1, u - 1:u + 2].mean(1), bgm[lo:hi + 1, u - 1:u + 2].mean(1), lo)
        if c is not None and ct >= 20:
            P.append([float(u), c])
    if len(P) < 10:
        continue
    P = np.array(P)
    q = (Hn @ np.hstack([P, np.ones((len(P), 1))]).T).T
    x = q[:, 0] / q[:, 2]
    exp = i * PITCH - LDX
    print(f"  i={i}: 예측 x = {x.mean():.4f} ± {x.std():.4f} m   "
          f"(기대 {exp:.4f})   오차 {(x.mean()-exp)*1000:+.1f} mm")

np.save("/home/xytron/H_vert.npy", Hn)
np.save("/home/xytron/H_vert_params.npy", sol.x)
print("\n저장: H_vert.npy, H_vert_params.npy")
