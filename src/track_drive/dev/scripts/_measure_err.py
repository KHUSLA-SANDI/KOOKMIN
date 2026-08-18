#!/usr/bin/env python3
"""정량 검증: 예측 줄눈 위치 vs 실제 줄눈(어두운 골) 위치 차이를 px·mm 로 측정"""
import sys
import numpy as np, cv2

sys.path.insert(0, "/home/xytron/xycar_ws/본선/track_drive")
from track_drive.lib import bev as B

CFG = "/home/xytron/xycar_ws/본선/track_drive/config/camera.yaml"
P = 0.453333
cam = B.load_camera_config(CFG)
raw = cv2.imread("/home/xytron/calib_raw.jpg")
und = B.Undistorter.from_config(cam, (raw.shape[1], raw.shape[0]))
img = und.apply(raw)
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
g = cv2.GaussianBlur(g, (0, 0), 1.2)

H = np.array(cam["H"], np.float64)
Hi = np.linalg.inv(H)
WREF = np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2])


def g2p(x, y):
    q = Hi @ np.array([x, y, 1.0])
    if abs(q[2]) < 1e-12 or np.sign(q[2]) != WREF:
        return None
    return q[0] / q[2], q[1] / q[2]


def find_dark_v(u, v0, half=9):
    """(u, v0) 주변 세로로 ±half 훑어 가장 어두운 행을 서브픽셀로 반환"""
    u = int(round(u))
    if not (2 <= u < g.shape[1] - 2):
        return None
    lo, hi = int(round(v0 - half)), int(round(v0 + half))
    if lo < 1 or hi >= g.shape[0] - 1:
        return None
    col = g[lo:hi + 1, u - 2:u + 3].mean(axis=1)   # 좌우 5px 평균으로 노이즈 억제
    k = int(np.argmin(col))
    if k == 0 or k == len(col) - 1:
        return None
    # 포물선 서브픽셀
    y0, y1, y2 = col[k - 1], col[k], col[k + 1]
    den = (y0 - 2 * y1 + y2)
    d = 0.5 * (y0 - y2) / den if abs(den) > 1e-6 else 0.0
    contrast = float(col.max() - col.min())
    return lo + k + d, contrast


print(f"{'i':>2} {'x[m]':>6} {'u':>5} {'예측v':>7} {'실제v':>7} {'Δv[px]':>7} "
      f"{'Δx[mm]':>8} {'대비':>5}")
print("-" * 62)
all_err = []
for i in range(1, 7):
    x = i * P
    for yy in (0.30, 0.0, -0.30):
        p = g2p(x, yy)
        if p is None:
            continue
        u, v = p
        if not (20 <= u <= 620):
            continue
        r = find_dark_v(u, v, half=max(4, int(28 / i)))
        if r is None:
            continue
        vreal, contrast = r
        if contrast < 6:                      # 줄눈이 안 보이는 구간
            print(f"{i:2d} {x:6.3f} {u:5.0f} {v:7.1f} {'--':>7} {'--':>7} "
                  f"{'--':>8} {contrast:5.1f}  (대비 부족)")
            continue
        # Δv → Δx 로 환산 (해당 지점의 지면 해상도)
        p2 = g2p(x + 0.02, yy)
        dvdx = (p2[1] - v) / 0.02 if p2 else None
        dx_mm = (vreal - v) / dvdx * 1000 if dvdx else float("nan")
        all_err.append(abs(dx_mm))
        print(f"{i:2d} {x:6.3f} {u:5.0f} {v:7.1f} {vreal:7.1f} {vreal-v:+7.2f} "
              f"{dx_mm:+8.1f} {contrast:5.1f}")

if all_err:
    a = np.array(all_err)
    print("-" * 62)
    print(f"측정 {len(a)}개 → 평균 |Δx| = {a.mean():.1f} mm, "
          f"중앙값 {np.median(a):.1f} mm, 최대 {a.max():.1f} mm")
