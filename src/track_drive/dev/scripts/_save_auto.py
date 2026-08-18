#!/usr/bin/env python3
"""자동정밀화 H 의 대응점을 CSV 로 덤프 → calib_ground_h.py 로 정식 저장"""
import sys
import numpy as np, cv2

sys.path.insert(0, "/home/xytron/xycar_ws/본선/track_drive")
from track_drive.lib import bev as B

CFG = "/home/xytron/xycar_ws/본선/track_drive/config/camera.yaml"
P = 0.453333
cam = B.load_camera_config(CFG)
raw = cv2.imread("/home/xytron/calib_raw.jpg")
und = B.Undistorter.from_config(cam, (raw.shape[1], raw.shape[0]))
gray = cv2.GaussianBlur(cv2.cvtColor(und.apply(raw), cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.1)
Himg, Wimg = gray.shape

H = np.load("/home/xytron/H_auto.npy")
Hi = np.linalg.inv(H)
wref = np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2])


def g2p(x, y):
    q = Hi @ np.array([x, y, 1.0])
    if abs(q[2]) < 1e-12 or np.sign(q[2]) != wref:
        return None
    u, v = q[0] / q[2], q[1] / q[2]
    return (u, v) if (3 <= u < Wimg - 3 and 3 <= v < Himg - 3) else None


def sub(prof, lo):
    k = int(np.argmin(prof))
    if k <= 0 or k >= len(prof) - 1:
        return None, 0.0
    a, b, c = prof[k - 1], prof[k], prof[k + 1]
    d = 0.5 * (a - c) / (a - 2 * b + c) if abs(a - 2 * b + c) > 1e-6 else 0.0
    return lo + k + d, float(prof.max() - prof.min())


rows = []
# 가로 줄눈 (탐색창 ±3px — 이미 정밀하므로 좁게 = 오검출 방지)
for i in range(1, 8):
    x = i * P
    for yy in (-0.60, -0.30, 0.0, 0.30, 0.60):
        p = g2p(x, yy)
        if not p:
            continue
        lo = int(round(p[1] - 3)); hi = int(round(p[1] + 3)); u = int(round(p[0]))
        if lo < 1 or hi >= Himg - 1 or u < 3 or u >= Wimg - 3:
            continue
        vv, ct = sub(gray[lo:hi + 1, u - 2:u + 3].mean(axis=1), lo)
        if vv is None or ct < 8:
            continue
        rows.append((p[0], vv, x, yy, f"auto row i={i} y={yy:+.2f}"))
# 세로 줄눈
for j in (-2, -1, 0, 1, 2):
    y = j * P
    for xx in (0.55, 0.75, 1.0, 1.3, 1.7):
        p = g2p(xx, y)
        if not p:
            continue
        lo = int(round(p[0] - 6)); hi = int(round(p[0] + 6)); v = int(round(p[1]))
        if lo < 1 or hi >= Wimg - 1 or v < 3 or v >= Himg - 3:
            continue
        uu, ct = sub(gray[v - 2:v + 3, lo:hi + 1].mean(axis=0), lo)
        if uu is None or ct < 8:
            continue
        rows.append((uu, p[1], xx, y, f"auto col j={j} x={xx:.2f}"))

print(f"자동 검출 대응점 {len(rows)}개")
with open("/home/xytron/h_pts_auto.csv", "w", encoding="utf-8") as f:
    f.write("# calib_ground_h.py 자동검출 대응점 — u,v,x,y\n")
    f.write("# 줄눈을 서브픽셀로 검출해 생성 (사람 클릭 오차 제거)\n")
    f.write(f"# calib_raw.jpg (auto-grout p={P}, frame=A_prime)\n")
    for u, v, x, y, tag in rows:
        f.write(f"{u:.2f},{v:.2f},{x:.4f},{y:.4f}  # {tag}\n")
print("저장: /home/xytron/h_pts_auto.csv")
