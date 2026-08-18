#!/usr/bin/env python3
"""/image_raw 에서 한 프레임 받아 원본/보정본 저장 (usb_cam 이 이미 떠 있을 때)"""
import sys, time
import numpy as np, cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

PKG = "/home/xytron/xycar_ws/src/track_drive"
sys.path.insert(0, PKG)
from track_drive.lib import bev as B
from track_drive.lib import preprocess as PP

OUT = sys.argv[1] if len(sys.argv) > 1 else PKG + "/dev/shots/tape"


class Grab(Node):
    def __init__(self):
        super().__init__("grab_once")
        self.cam = B.load_camera_config(PKG + "/config/camera.yaml")
        self.n = 0
        self.done = False
        self.create_subscription(Image, "/image_raw", self.cb, qos_profile_sensor_data)

    def cb(self, msg):
        if self.done:
            return
        self.n += 1
        if self.n < 12:            # 노출 안정화 위해 앞부분 버림
            return
        bgr, info = PP.image_msg_to_bgr(msg)
        if bgr is None:
            self.get_logger().error("변환 실패: %s" % info); self.done = True; return
        und = B.Undistorter.from_config(self.cam, (bgr.shape[1], bgr.shape[0]))
        fixed = und.apply(bgr)
        cv2.imwrite(OUT + "_raw.jpg", bgr)
        cv2.imwrite(OUT + "_und.jpg", fixed)
        # 대비강화본
        lab = cv2.cvtColor(fixed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(3.0, (8, 8)).apply(l)
        enh = cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)
        cv2.imwrite(OUT + "_enh.jpg", enh)
        print("saved:", OUT + "_raw.jpg /", OUT + "_und.jpg /", OUT + "_enh.jpg",
              bgr.shape, msg.encoding)
        self.done = True


def main():
    rclpy.init()
    n = Grab()
    t0 = time.time()
    while rclpy.ok() and not n.done and time.time() - t0 < 20:
        rclpy.spin_once(n, timeout_sec=0.2)
    if not n.done:
        print("ERR: /image_raw 프레임 수신 실패 (카메라 켜져 있나?)")
    n.destroy_node()
    rclpy.shutdown()


main()
