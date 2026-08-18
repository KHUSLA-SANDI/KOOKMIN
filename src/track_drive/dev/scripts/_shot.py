#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""새 프레임 캡처 — N장 모아 중앙값(노이즈/압축잡음 제거) 으로 저장.

사용: python3 _shot.py <출력이름>   (예: tape3 → dev/shots/tape3_raw.jpg)
"""
import os
import sys
import time

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

PKG = "/home/xytron/xycar_ws/src/track_drive"
sys.path.insert(0, PKG)
from track_drive.lib import bev as B          # noqa: E402
from track_drive.lib import preprocess as PP  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "tape3"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 15
out_dir = os.path.join(PKG, "dev", "shots")
os.makedirs(out_dir, exist_ok=True)

frames = []
rclpy.init()
node = Node("shot_grab")


def cb(msg):
    if len(frames) < N:
        bgr, _ = PP.image_msg_to_bgr(msg)
        if bgr is not None:
            frames.append(bgr)


node.create_subscription(Image, "/image_raw", cb, qos_profile_sensor_data)
t0 = time.time()
while len(frames) < N and time.time() - t0 < 20.0:
    rclpy.spin_once(node, timeout_sec=0.2)
node.destroy_node()
rclpy.shutdown()

if not frames:
    print("❌ 프레임을 못 받았다 — 카메라 노드 확인"); sys.exit(1)

med = np.median(np.stack(frames, 0), axis=0).astype(np.uint8)
raw_p = os.path.join(out_dir, name + "_raw.jpg")
cv2.imwrite(raw_p, med, [cv2.IMWRITE_JPEG_QUALITY, 97])

cam = B.load_camera_config(os.path.join(PKG, "config", "camera.yaml"))
und = B.Undistorter.from_config(cam, (med.shape[1], med.shape[0]))
und_p = os.path.join(out_dir, name + "_und.jpg")
cv2.imwrite(und_p, und.apply(med), [cv2.IMWRITE_JPEG_QUALITY, 97])

print("%d장 중앙값 → %s" % (len(frames), raw_p))
print("현재 K/D 로 보정한 것 → %s  (undistort=%s)" % (und_p, und.enabled))
