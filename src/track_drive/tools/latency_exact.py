#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""카메라 실제 지연 측정.

쓰는 법
  폰 스톱워치의 [시작] 버튼과 엔터를 '동시에' 누른다. 그게 전부다.
  카운트다운 없다. 네가 원할 때 누르면 된다.

원리
  엔터를 친 순간 = 스톱워치 0.00 초. 이걸로 폰 시계와 차 시계가 맞춰진다.
  그 다음 들어오는 프레임마다 사진에 스톱워치 숫자가 찍히므로,
  그 프레임이 '진짜 언제 찍혔는지' 를 알 수 있다.

      실제 지연 = (프레임 도착시각 - 엔터시각) - 사진 속 스톱워치 값

  앞선 측정처럼 "신호를 보고 반응하는" 동작이 아니라 네가 스스로 두 손을
  같이 움직이는 거라, 오차가 300~500ms 가 아니라 50ms 수준이다.
"""
import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy,
                       QoSDurabilityPolicy)
from sensor_msgs.msg import Image

OUTDIR = "/home/xytron/latency_shots2"


class Cap(Node):
    def __init__(self):
        super().__init__("sw_probe2")
        qos = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         durability=QoSDurabilityPolicy.VOLATILE)
        self.create_subscription(Image, "/image_raw", self.cb, qos)
        self.lock = threading.Lock()
        self.buf = []

    def cb(self, msg):
        now = time.time()
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            img = img[:, :, ::-1]
        with self.lock:
            self.buf.append((now, st, img.copy()))
            if len(self.buf) > 80:
                self.buf.pop(0)


def can_window():
    if not os.environ.get("DISPLAY"):
        return False
    try:
        cv2.namedWindow("__p__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__p__")
        cv2.waitKey(1)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=8, help="저장할 장수")
    a = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    rclpy.init()
    n = Cap()
    stop = threading.Event()
    shown = threading.Event()

    def spin():
        while not stop.is_set():
            rclpy.spin_once(n, timeout_sec=0.01)
    threading.Thread(target=spin, daemon=True).start()

    print("카메라 확인...", flush=True)
    t0 = time.time()
    while time.time() - t0 < 10:
        with n.lock:
            k = len(n.buf)
        if k > 5:
            break
        time.sleep(0.2)
    else:
        print("★ /image_raw 를 못 받았다.")
        stop.set()
        return 1

    use_win = can_window()
    W = "latency - 폰을 화면에 잡아라"
    if use_win:
        cv2.namedWindow(W, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(W, 900, 506)

        def preview():
            while not stop.is_set() and not shown.is_set():
                with n.lock:
                    if not n.buf:
                        continue
                    img = n.buf[-1][2]
                sh = cv2.resize(img, (900, 506), interpolation=cv2.INTER_AREA)
                cv2.putText(sh, "stopwatch START + ENTER together", (14, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.imshow(W, sh)
                cv2.waitKey(30)
        threading.Thread(target=preview, daemon=True).start()

    print()
    print("=" * 62)
    print("  폰 스톱워치를 카메라 앞에 들고, 숫자가 읽히게 위치를 잡아라.")
    print("  (스톱워치는 0 에서 멈춰 있어야 한다. 이미 돌고 있으면 초기화)")
    print()
    print("  준비되면 ★ 스톱워치 [시작] 버튼과 엔터를 동시에 ★ 눌러라.")
    print("=" * 62)
    print()

    input("> ")
    t_zero = time.time()
    shown.set()
    time.sleep(0.05)
    if use_win:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    # 엔터 직후 프레임들을 모은다 (0.05 ~ 1.2초 구간)
    time.sleep(1.3)
    with n.lock:
        snap = [(rx, st, im) for (rx, st, im) in n.buf if rx >= t_zero]
    stop.set()
    n.destroy_node()
    rclpy.shutdown()

    if not snap:
        print("★ 프레임을 못 모았다.")
        return 1

    # 너무 촘촘하면 스톱워치 숫자 차이가 안 보이니 몇 장 건너뛰며 고른다
    step = max(1, len(snap) // a.shots)
    snap = snap[::step][:a.shots]

    print()
    print("  엔터 = 스톱워치 0.00 초")
    print()
    print("  %-4s %-14s %-14s" % ("장", "도착-엔터", "stamp-엔터"))
    for i, (rx, st, _) in enumerate(snap, 1):
        print("  %-4d %+13.3f %+13.3f" % (i, rx - t_zero, st - t_zero))

    print("\n저장:")
    for i, (rx, st, img) in enumerate(snap, 1):
        out = img.copy()
        h = out.shape[0]
        band = int(h * 0.17)
        out[:band] = (out[:band] * 0.20).astype(np.uint8)
        fs = h / 640.0
        lines = ["arrive = %+.3f s" % (rx - t_zero),
                 "stamp  = %+.3f s" % (st - t_zero)]
        for k, s in enumerate(lines):
            cv2.putText(out, s, (18, int(band * 0.40) + k * int(band * 0.40)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255),
                        max(2, int(fs * 2)), cv2.LINE_AA)
        p = os.path.join(OUTDIR, "m%d.jpg" % i)
        cv2.imwrite(p, out, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print("  %s" % p)

    print()
    print("=" * 62)
    print("  실제 지연 = arrive - (사진 속 스톱워치 값)")
    print()
    print("  예) arrive=+0.35, 사진 스톱워치가 0.30 이면  지연 = 0.05초")
    print("      arrive=+0.35, 사진 스톱워치가 0.00 이면  지연 = 0.35초")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
