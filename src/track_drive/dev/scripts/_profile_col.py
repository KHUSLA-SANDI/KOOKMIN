#!/usr/bin/env python3
"""중앙 세로 스트립을 훑어 '실제 줄눈 v 위치 전부'를 뽑고 예측과 대조 (오검출 배제)"""
import sys
import numpy as np, cv2
from scipy.signal import find_peaks

sys.path.insert(0, "/home/xytron/xycar_ws/본선/track_drive")
from track_drive.lib import bev as B

CFG = "/home/xytron/xycar_ws/본선/track_drive/config/camera.yaml"
P = 0.453333
cam = B.load_camera_config(CFG)
raw = cv2.imread("/home/xytron/calib_raw.jpg")
und = B.Undistorter.from_config(cam, (raw.shape[1], raw.shape[0]))
img = und.apply(raw)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

for name, Hm in (("현재 저장 H", np.array(cam["H"], np.float64)),
                 ("자동정밀화 H", np.load("/home/xytron/H_auto.npy"))):
    Hi = np.linalg.inv(Hm)
    wref = np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2])

    def g2v(x):
        q = Hi @ np.array([x, 0.0, 1.0])
        if np.sign(q[2]) != wref:
            return None
        return q[1] / q[2]

    print(f"\n=== {name} — 예측 줄눈 v (y=0) ===")
    pred = []
    for i in range(1, 8):
        v = g2v(i * P)
        if v and 240 < v < 470:
            pred.append((i, v))
            print(f"  i={i}  x={i*P:5.3f}m  v={v:7.2f}")

    # 실제: u 340~356 평균 프로파일에서 어두운 골 찾기
    strip = gray[240:470, 338:356].mean(axis=1)
    strip_s = cv2.GaussianBlur(strip.reshape(-1, 1), (1, 5), 0).ravel()
    inv = strip_s.max() - strip_s
    pk, props = find_peaks(inv, prominence=3.0, distance=3)
    print(f"  실제 검출 줄눈 v: {[round(240+float(p),1) for p in pk]}")
    print(f"  (돌출도: {[round(float(x),1) for x in props['prominences']]})")

    # 매칭: 각 예측에 가장 가까운 실측
    if len(pk):
        real = 240 + pk.astype(float)
        print(f"  {'i':>2} {'예측v':>7} {'최근접실측':>10} {'Δv':>6}")
        for i, v in pred:
            k = int(np.argmin(np.abs(real - v)))
            print(f"  {i:2d} {v:7.2f} {real[k]:10.2f} {real[k]-v:+6.2f}")
