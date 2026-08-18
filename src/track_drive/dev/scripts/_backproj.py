#!/usr/bin/env python3
"""H 검증: 타일 격자를 원본(undistort) 사진에 역투영 → 실제 줄눈과 겹치는지 육안 판정"""
import sys
import numpy as np, cv2

sys.path.insert(0, "/home/xytron/xycar_ws/src/track_drive")
sys.path.insert(0, "/home/xytron/xycar_ws/본선/track_drive")
from track_drive.lib import bev as B

CFG = "/home/xytron/xycar_ws/본선/track_drive/config/camera.yaml"
P = 0.453333

cam = B.load_camera_config(CFG)
raw = cv2.imread("/home/xytron/calib_raw.jpg")
und = B.Undistorter.from_config(cam, (raw.shape[1], raw.shape[0]))
img = und.apply(raw)
print("h_calibrated =", cam.get("h_calibrated"), " model =", cam.get("model"))

H = np.array(cam["H"], np.float64)      # 픽셀 → 지면
Hi = np.linalg.inv(H)                   # 지면 → 픽셀


# 정면(전방 1m) 지점의 w 부호를 기준으로 삼는다 — H 스케일 규약이 구현마다 달라서
_W_REF = np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2])


def g2p(x, y):
    q = Hi @ np.array([x, y, 1.0])
    if abs(q[2]) < 1e-12 or np.sign(q[2]) != _W_REF:   # 지평선 반대편 제외
        return None
    return q[0] / q[2], q[1] / q[2]


# 대비 강화 (줄눈 보이게)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b_ = cv2.split(lab)
cl = cv2.createCLAHE(3.0, (8, 8)).apply(l)
vis = cv2.cvtColor(cv2.merge([cl, a, b_]), cv2.COLOR_LAB2BGR)
blur = cv2.GaussianBlur(vis, (0, 0), 3)
vis = cv2.addWeighted(vis, 1.6, blur, -0.6, 0)

# 가로 줄눈(x = i*P) : y 를 -1.4~1.4 로 훑어 선 그리기
for i in range(1, 8):
    x = i * P
    pts = []
    for yy in np.linspace(-1.4, 1.4, 60):
        p = g2p(x, yy)
        if p and -400 < p[0] < 1040 and -200 < p[1] < 680:
            pts.append((int(round(p[0])), int(round(p[1]))))
    if len(pts) > 1:
        cv2.polylines(vis, [np.array(pts, np.int32)], False, (0, 255, 255), 1, cv2.LINE_AA)
        px = g2p(x, 0.0)
        if px:
            cv2.putText(vis, f"i={i} {x:.2f}m", (8, int(px[1]) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

# 세로 줄눈(y = j*P)
for j in range(-3, 4):
    y = j * P
    pts = []
    for xx in np.linspace(0.35, 4.0, 80):
        p = g2p(xx, y)
        if p and -400 < p[0] < 1040 and -200 < p[1] < 680:
            pts.append((int(round(p[0])), int(round(p[1]))))
    if len(pts) > 1:
        col = (255, 80, 255) if j == 0 else (255, 200, 0)
        cv2.polylines(vis, [np.array(pts, np.int32)], False, col, 1, cv2.LINE_AA)

# 클릭했던 점 표시
for ln in open("/home/xytron/h_pts_clean.csv", encoding="utf-8"):
    if ln.strip().startswith("#") or not ln.strip():
        continue
    t = ln.split("#")[0].split(",")
    u, v = float(t[0]), float(t[1])
    cv2.drawMarker(vis, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 11, 2)

cv2.putText(vis, "YELLOW=tile rows(x)  ORANGE/MAGENTA=tile cols(y)  RED X=clicked",
            (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite("/home/xytron/backproj.jpg", vis)

# 확대본 (근~중거리)
z = cv2.resize(vis[250:480, :, :], None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
cv2.imwrite("/home/xytron/backproj_zoom.jpg", z)
print("saved: backproj.jpg, backproj_zoom.jpg")

print("\n예상 줄눈 위치 (y=0 기준):")
for i in range(1, 8):
    p = g2p(i * P, 0.0)
    if p:
        print(f"  i={i}  x={i*P:.3f}m  →  v={p[1]:7.1f}")
