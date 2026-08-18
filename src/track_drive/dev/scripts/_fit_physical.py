#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
물리 제약 H 피팅 — 자유 8DOF 대신 카메라 자세 4~6개만 맞춘다.

자유 호모그래피는 노이즈를 흡수해 '물리적으로 불가능한' 변환이 될 수 있다
(실제로 현재 H 는 yaw -11.7deg / roll -3deg 로 재현조차 안 됨).
지면 H 는 반드시 K[r1 r2 t] 형태여야 하므로, 그 형태를 강제하고
height/pitch/yaw/roll(/cam_x/cam_y) 만 최소제곱으로 찾는다.
파라미터가 적어 노이즈에 강하고, 결과가 항상 물리적으로 타당하다.
"""
import sys
import numpy as np, cv2
from scipy.optimize import least_squares

PKG = "/home/xytron/xycar_ws/src/track_drive"
sys.path.insert(0, PKG)
from track_drive.lib import bev as B

CFG = PKG + "/config/camera.yaml"
PITCH = 0.453333
LDX = 0.10          # 라이다가 카메라(A')보다 앞선 거리


def build_H(K, h, pitch_d, yaw_d, roll_d, cx, cy):
    th, ps, rl = np.radians([pitch_d, yaw_d, roll_d])
    R0 = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    Rp = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    Ry = np.array([[np.cos(ps), -np.sin(ps), 0], [np.sin(ps), np.cos(ps), 0], [0, 0, 1]])
    Rr = np.array([[np.cos(rl), -np.sin(rl), 0], [np.sin(rl), np.cos(rl), 0], [0, 0, 1]])
    R = Rr @ Rp @ R0 @ Ry
    t = -R @ np.array([cx, cy, h])
    return np.linalg.inv(K @ np.column_stack([R[:, 0], R[:, 1], t]))


def load_pts(path):
    P = []
    for ln in open(path, encoding="utf-8"):
        s = ln.split("#")[0].strip()
        if not s:
            continue
        v = [float(t) for t in s.split(",")]
        if len(v) >= 4:
            P.append(v[:4])
    return np.array(P)


cam = B.load_camera_config(CFG)
K = np.array(cam["new_K"] if cam.get("new_K") is not None else cam["K"], np.float64)
pts = load_pts("dev/calib/h_pts_auto.csv")
pix, gnd = pts[:, :2], pts[:, 2:4]
print(f"대응점 {len(pts)}개 (자동검출 줄눈)")

# 지면 좌표는 lidar_frame 기준으로 저장돼 있음 → 그대로 사용
Hg = None


def resid_full(p):
    """이미지 평면에서의 재투영 오차 (통계적으로 올바른 방향)"""
    Hm = build_H(K, *p)
    Hi = np.linalg.inv(Hm)
    g = np.hstack([gnd, np.ones((len(gnd), 1))])
    q = (Hi @ g.T).T
    q = q[:, :2] / q[:, 2:3]
    return (q - pix).ravel()


def report(name, p, fixed=None):
    Hm = build_H(K, *p)
    ph = np.hstack([pix, np.ones((len(pix), 1))])
    q = (Hm @ ph.T).T
    q = q[:, :2] / q[:, 2:3]
    e = np.linalg.norm(q - gnd, axis=1)
    r = resid_full(p).reshape(-1, 2)
    ep = np.linalg.norm(r, axis=1)
    print(f"\n[{name}]")
    print(f"  height={p[0]:.4f}m  pitch={p[1]:+.2f}°  yaw={p[2]:+.2f}°  "
          f"roll={p[3]:+.2f}°  cam_x={p[4]:+.4f}  cam_y={p[5]:+.4f}")
    print(f"  지면오차 RMS {np.sqrt((e**2).mean())*1000:6.1f} mm / 최대 {e.max()*1000:6.1f} mm")
    print(f"  픽셀오차 RMS {np.sqrt((ep**2).mean()):6.2f} px / 최대 {ep.max():6.2f} px")
    return Hm, float(np.sqrt((e ** 2).mean()))


x0 = np.array([0.25, 4.0, 0.0, 0.0, -LDX, 0.0])

# ① 4DOF: cam_x, cam_y 고정 (물리적으로 아는 값)
def r4(q):
    return resid_full(np.array([q[0], q[1], q[2], q[3], -LDX, 0.0]))
s4 = least_squares(r4, x0[:4], method="lm", max_nfev=5000)
p4 = np.array([*s4.x, -LDX, 0.0])
H4, rms4 = report("4DOF (높이·피치·yaw·roll)", p4)

# ② 6DOF: 카메라 위치도 자유
s6 = least_squares(resid_full, x0, method="lm", max_nfev=8000)
H6, rms6 = report("6DOF (+ 카메라 전후/좌우 위치)", s6.x)

# ③ 비교: 기존 자유 8DOF H
Hf = np.array(cam["H"], np.float64)
ph = np.hstack([pix, np.ones((len(pix), 1))])
q = (Hf @ ph.T).T; q = q[:, :2] / q[:, 2:3]
ef = np.linalg.norm(q - gnd, axis=1)
print(f"\n[기존 자유 8DOF H]  지면오차 RMS {np.sqrt((ef**2).mean())*1000:.1f} mm "
      f"/ 최대 {ef.max()*1000:.1f} mm")

best = min([("4DOF", p4, H4, rms4), ("6DOF", s6.x, H6, rms6)], key=lambda z: z[3])
print(f"\n>>> 물리제약 최적: {best[0]}, 지면 RMS {best[3]*1000:.1f} mm")
np.save("/home/xytron/H_phys.npy", best[2])
np.save("/home/xytron/H_phys_params.npy", best[1])
print("저장: H_phys.npy, H_phys_params.npy")
print("\n※ 물리제약 H 는 RMS 가 조금 커도 '실제로 가능한 변환' 이라 BEV 가 안정적이다.")
