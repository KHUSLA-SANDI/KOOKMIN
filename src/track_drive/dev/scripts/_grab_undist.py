#!/usr/bin/env python3
# 차에서 실행: 한 프레임 캡처 → undistort → 원본/보정본 저장 + 격자
import sys, time
import numpy as np, cv2
sys.path.insert(0, "/home/xytron/xycar_ws/src/track_drive")
from track_drive.lib import bev

CFG = "/home/xytron/xycar_ws/src/track_drive/config/camera.yaml"
OUT = "/home/xytron/calib"

cam = bev.load_camera_config(CFG)
print("model=%s intr=%s h=%s note=%s" % (
    cam.get("model"), cam.get("intrinsics_calibrated"), cam.get("h_calibrated"), cam.get("note", "")))
print("K[0,0]=%.2f D[0]=%.4f" % (cam["K"][0, 0], np.ravel(cam["D"])[0]))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("ERR: 카메라 열기 실패 (다른 프로세스가 점유중일 수 있음)"); sys.exit(1)
for _ in range(25):          # 노출 안정화
    cap.read(); time.sleep(0.04)
ok, raw = cap.read()
cap.release()
if not ok:
    print("ERR: 프레임 read 실패"); sys.exit(1)

und = bev.Undistorter.from_config(cam, (raw.shape[1], raw.shape[0]))
print("undistort enabled =", und.enabled, "model =", und.model)
fixed = und.apply(raw)

cv2.imwrite(OUT + "_raw.jpg", raw)
cv2.imwrite(OUT + "_und.jpg", fixed)

# 보정본에 행 눈금 (A4가 어느 행에 잡히는지 파악용)
g = fixed.copy()
for v in range(260, 480, 20):
    cv2.line(g, (0, v), (639, v), (0, 200, 0), 1)
    cv2.putText(g, str(v), (2, v - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
cv2.line(g, (320, 240), (320, 479), (255, 0, 0), 1)   # 이미지 중앙 세로선
cv2.imwrite(OUT + "_grid.jpg", g)
print("saved:", OUT + "_raw.jpg /", OUT + "_und.jpg /", OUT + "_grid.jpg")
