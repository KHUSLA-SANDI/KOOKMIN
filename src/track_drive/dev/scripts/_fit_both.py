#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최종 H 피팅 — 물리 4DOF, 세로선=횡기준 / 가로선=종기준.

교훈 정리:
  · 자유 8DOF H → 노이즈 흡수해 물리적으로 불가능한 변환이 됨 (yaw -11.7°, 재현불가)
  · 세로선만 → yaw 와 주점오차가 축퇴(degenerate)돼 높이/거리가 45% 틀어짐
  · 가로선은 안 곧지만(±3.7cm 출렁) **평균 거리는 정확** → 종방향 스케일 기준으로는 유효

좌표계: 원점 A'(카메라 지면투영점), x 전방(타일 방향), y 좌.
        카메라는 원점 위에 있으므로 cam_x=cam_y=0 → 미지수는 height/pitch/yaw/roll 4개뿐.
저장 시 x 에서 LDX(0.10)를 빼 lidar_frame 으로 변환.
"""
import argparse
import sys
import numpy as np, cv2
from scipy.optimize import least_squares

PKG = "/home/xytron/xycar_ws/src/track_drive"
sys.path.insert(0, PKG)
from track_drive.lib import bev as B

PITCH, LDX = 0.453333, 0.10
CFG = PKG + "/config/camera.yaml"
ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true")
A = ap.parse_args()

cam = B.load_camera_config(CFG)
K = np.array(cam["new_K"] if cam.get("new_K") is not None else cam["K"], np.float64)
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


V_obs, V_y = [], []      # 세로선: y 구속
Hz_obs, Hz_x = [], []    # 가로선: x 구속
for j in (-1, 1, 2):
    for xx in np.linspace(0.40, 2.9, 150):
        p = gu0(xx, j * PITCH)
        if p is None: continue
        v = int(round(p[1])); lo = max(0, int(p[0] - 16)); hi = min(Ww - 1, int(p[0] + 16))
        if hi - lo < 6 or v < 2 or v >= Hh - 2: continue
        c, ct = cen(g[v-1:v+2, lo:hi+1].mean(0), bgm[v-1:v+2, lo:hi+1].mean(0), lo)
        if c is not None and ct >= 20:
            V_obs.append([c, float(v)]); V_y.append(j * PITCH)
for i in (1, 2, 3):
    for yy in np.linspace(-1.0, 1.0, 100):
        p = gu0(i * PITCH, yy)
        if p is None: continue
        u = int(round(p[0])); lo = max(0, int(p[1] - 13)); hi = min(Hh - 1, int(p[1] + 13))
        if hi - lo < 6 or u < 2 or u >= Ww - 2: continue
        c, ct = cen(g[lo:hi+1, u-1:u+2].mean(1), bgm[lo:hi+1, u-1:u+2].mean(1), lo)
        if c is not None and ct >= 20:
            Hz_obs.append([float(u), c]); Hz_x.append(i * PITCH)
V_obs = np.array(V_obs); V_y = np.array(V_y)
Hz_obs = np.array(Hz_obs); Hz_x = np.array(Hz_x)
print(f"세로선 {len(V_obs)}점(횡기준) / 가로선 {len(Hz_obs)}점(종기준)")


def build_H(h, pit, yaw, roll):
    th, ps, rl = np.radians([pit, yaw, roll])
    R0 = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    Rp = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    Ry = np.array([[np.cos(ps), -np.sin(ps), 0], [np.sin(ps), np.cos(ps), 0], [0, 0, 1]])
    Rr = np.array([[np.cos(rl), -np.sin(rl), 0], [np.sin(rl), np.cos(rl), 0], [0, 0, 1]])
    R = Rr @ Rp @ R0 @ Ry
    t = -R @ np.array([0.0, 0.0, h])
    return np.linalg.inv(K @ np.column_stack([R[:, 0], R[:, 1], t]))


phV = np.hstack([V_obs, np.ones((len(V_obs), 1))])
phH = np.hstack([Hz_obs, np.ones((len(Hz_obs), 1))])


def resid(p):
    Hm = build_H(*p)
    qv = (Hm @ phV.T).T; wv = np.where(np.abs(qv[:, 2]) < 1e-9, 1e-9, qv[:, 2])
    ry = qv[:, 1] / wv - V_y
    qh = (Hm @ phH.T).T; wh = np.where(np.abs(qh[:, 2]) < 1e-9, 1e-9, qh[:, 2])
    rx = qh[:, 0] / wh - Hz_x
    # 종방향은 원거리 가중이 과해지지 않게 상대오차로
    return np.concatenate([ry, rx / np.maximum(Hz_x, 0.3)])


best = None
for h0 in (0.14, 0.18, 0.22, 0.26, 0.30):
    for y0 in (-8.0, 0.0, 8.0):
        try:
            s = least_squares(resid, np.array([h0, 2.0, y0, 0.0]),
                              loss="soft_l1", f_scale=0.02, max_nfev=6000)
            if best is None or s.cost < best.cost:
                best = s
        except Exception:
            pass
h, pit, yaw, roll = best.x
Hn = build_H(*best.x)
print(f"\n=== 최종 물리 파라미터 (4DOF) ===")
print(f"  height = {h:.4f} m")
print(f"  pitch  = {pit:+.2f}°")
print(f"  yaw    = {yaw:+.2f}°")
print(f"  roll   = {roll:+.2f}°")

qv = (Hn @ phV.T).T; ry = qv[:, 1] / qv[:, 2] - V_y
qh = (Hn @ phH.T).T; rx = qh[:, 0] / qh[:, 2] - Hz_x
print(f"  횡오차(세로선) RMS {np.sqrt((ry**2).mean())*1000:6.1f} mm / 최대 {np.abs(ry).max()*1000:6.1f}")
print(f"  종오차(가로선) RMS {np.sqrt((rx**2).mean())*1000:6.1f} mm / 최대 {np.abs(rx).max()*1000:6.1f}")

print("\n=== 가로 테이프 거리 검증 ===")
for i in (1, 2, 3):
    m = np.isclose(Hz_x, i * PITCH)
    if m.sum() < 5: continue
    xs = (qh[m, 0] / qh[m, 2])
    print(f"  i={i}: 예측 {xs.mean():.4f} ± {xs.std():.4f} m  (기대 {i*PITCH:.4f})  "
          f"오차 {(xs.mean()-i*PITCH)*1000:+6.1f} mm")
print("=== 세로 테이프 횡위치 검증 ===")
for j in (-1, 1, 2):
    m = np.isclose(V_y, j * PITCH)
    if m.sum() < 5: continue
    ys = (qv[m, 1] / qv[m, 2])
    print(f"  j={j:+d}: 예측 {ys.mean():+.4f} ± {ys.std():.4f} m  (기대 {j*PITCH:+.4f})  "
          f"오차 {(ys.mean()-j*PITCH)*1000:+6.1f} mm")

# lidar_frame 변환 (x - LDX)
T = np.array([[1, 0, -LDX], [0, 1, 0], [0, 0, 1]], np.float64)
H_lidar = T @ Hn
H_lidar = H_lidar / H_lidar[2, 2]
np.save("/home/xytron/H_final.npy", H_lidar)
print("\n저장: H_final.npy (lidar_frame)")

if A.save:
    import yaml
    with open(CFG, encoding="utf-8") as f:
        y = yaml.safe_load(f)
    y["H"] = [[float(t) for t in r] for r in H_lidar]
    y["h_frame"] = "lidar_frame"
    y["h_calibrated"] = True
    y["calibrated"] = bool(y.get("intrinsics_calibrated", True))
    y["h_source"] = ("dev/scripts/_fit_both.py 물리4DOF "
                     "(h=%.4f pitch=%.2f yaw=%.2f roll=%.2f)" % (h, pit, yaw, roll))
    y["h_rms_m"] = float(round(np.sqrt((np.concatenate([ry, rx]) ** 2).mean()), 5))
    with open(CFG, "w", encoding="utf-8") as f:
        f.write("# 자동생성 — dev/scripts/_fit_both.py (물리제약 4DOF, 세로=횡/가로=종 기준)\n")
        yaml.safe_dump(y, f, allow_unicode=True, sort_keys=True)
    print("camera.yaml 갱신 완료")
