#!/usr/bin/env python3
# 줄눈 가시성 강화 (CLAHE + 언샤프) — 클릭용 이미지 생성
import numpy as np, cv2

img = cv2.imread("/home/xytron/calib_und.jpg")
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
enh = cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)
blur = cv2.GaussianBlur(enh, (0, 0), 3)
enh = cv2.addWeighted(enh, 1.6, blur, -0.6, 0)
cv2.imwrite("/home/xytron/calib_enh.jpg", enh)

# 원거리 구간(v 250~330) 4배 확대 — 줄눈 보이는지 판정
crop = enh[248:335, 60:580]
z = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
for vv in range(250, 335, 10):
    y = int((vv - 248) * 3)
    cv2.line(z, (0, y), (z.shape[1], y), (0, 200, 255), 1)
    cv2.putText(z, str(vv), (3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
cv2.imwrite("/home/xytron/far_zoom.jpg", z)
print("saved: calib_enh.jpg, far_zoom.jpg", enh.shape, z.shape)
