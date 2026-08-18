#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""폰 스톱워치로 카메라 지연이 진짜인지 판별한다.

배경
  /image_raw 의 header.stamp 와 실제 도착 시각이 653ms 차이난다. 두 가지 가능성:
    A. 진짜로 653ms 늦게 도착한다        → 심각 (1.9m/s 에서 125cm)
    B. 도착은 제때인데 stamp 만 과거로 찍힌다 → 무해 (planner 는 수신시각을 쓴다)
  사진에 스톱워치가 찍히면 "진짜 촬영 시각"을 알 수 있어서 둘을 가를 수 있다.

쓰는 법
  1. 폰 스톱워치를 카메라 앞에 든다
  2. 프리뷰 창에서 숫자가 읽히게 위치를 잡는다 (창이 안 뜨면 아래 참고)
  3. 스페이스를 누르면 카운트다운 → "시작!" 에 맞춰 스톱워치를 누른다
  4. 3/5/7초 지점에서 자동으로 3장 찍힌다

창이 안 뜰 때
  그냥 ssh 로는 창을 못 띄운다. 둘 중 하나로 해라.
    · 원격 데스크톱(차 화면)에서 터미널을 열고 실행
    · 또는 ssh 에서 export DISPLAY=:0 을 먼저 실행
  그래도 안 되면 --no-preview 를 붙여라. 프리뷰 대신 /home/xytron/latency_shots/
  아래에 preview.jpg 를 계속 갱신하니 그걸 열어 보고 위치를 잡으면 된다.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy,
                       QoSDurabilityPolicy)
from sensor_msgs.msg import Image

OUTDIR = "/home/xytron/latency_shots"
SHOT_AT = [3.0, 5.0, 7.0]


class Cap(Node):
    def __init__(self):
        super().__init__("sw_probe")
        qos = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         durability=QoSDurabilityPolicy.VOLATILE)
        self.create_subscription(Image, "/image_raw", self.cb, qos)
        self.last = None
        self.count = 0

    def cb(self, msg):
        now = time.time()
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            img = img[:, :, ::-1]
        self.last = (now, st, img.copy())
        self.count += 1


def can_window():
    """창을 띄울 수 있는 환경인지 확인. 못 띄우면 조용히 False."""
    if not os.environ.get("DISPLAY"):
        return False
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        cv2.waitKey(1)
        return True
    except Exception:
        return False


def label(img, lines, color=(0, 255, 255)):
    """이미지 위쪽에 큰 글씨를 얹는다. 원본을 건드리지 않는다."""
    out = img.copy()
    h = out.shape[0]
    band = int(h * 0.17)
    out[:band] = (out[:band] * 0.22).astype(np.uint8)
    fs = h / 620.0
    for k, s in enumerate(lines):
        y = int(band * 0.40) + k * int(band * 0.40)
        cv2.putText(out, s, (20, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    color, max(2, int(fs * 2)), cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-preview", action="store_true",
                    help="창 없이 진행 (preview.jpg 를 대신 갱신)")
    ap.add_argument("--wait", type=float, default=0.0,
                    help="창 없이 쓸 때 위치잡을 시간(초). 예: --wait 20")
    a = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    rclpy.init()
    n = Cap()

    print("카메라 연결 확인...", flush=True)
    t0 = time.time()
    while n.last is None and time.time() - t0 < 10:
        rclpy.spin_once(n, timeout_sec=0.1)
    if n.last is None:
        print("★ /image_raw 를 못 받았다. 카메라가 떠 있는지 확인해라:")
        print("    ros2 topic hz /image_raw")
        return 1

    H, W = n.last[2].shape[:2]
    print("OK — %dx%d" % (W, H))

    use_win = (not a.no_preview) and can_window()
    if not use_win and not a.no_preview:
        print()
        print("▲ 창을 띄울 수 없다 (DISPLAY 없음). 프리뷰 없이 진행한다.")
        print("  창을 보고 싶으면 둘 중 하나:")
        print("    · 원격 데스크톱(차 화면)에서 터미널 열고 실행")
        print("    · 또는  export DISPLAY=:0  먼저 실행")
        print()

    # ── 위치 잡기 ──
    if use_win:
        print()
        print("=" * 62)
        print("  프리뷰 창이 떴다. 폰 스톱워치를 카메라 앞에 들어라.")
        print("  숫자가 읽히게 위치를 잡고 [스페이스] 를 눌러라. (q = 취소)")
        print("=" * 62)
        cv2.namedWindow("latency - 폰 스톱워치를 화면에 잡아라", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("latency - 폰 스톱워치를 화면에 잡아라", 960, 540)
        while True:
            rclpy.spin_once(n, timeout_sec=0.02)
            _, _, img = n.last
            sh = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
            cv2.putText(sh, "SPACE = start,  q = quit", (14, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("latency - 폰 스톱워치를 화면에 잡아라", sh)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(" "):
                break
            if k in (ord("q"), 27):
                cv2.destroyAllWindows()
                print("취소")
                return 0
    else:
        wait = a.wait if a.wait > 0 else 20.0
        pv = os.path.join(OUTDIR, "preview.jpg")
        print("=" * 62)
        print("  %.0f초 동안 %s 를 0.5초마다 갱신한다." % (wait, pv))
        print("  그 파일을 열어 놓고 폰 스톱워치가 화면에 잡히게 위치를 맞춰라.")
        print("=" * 62)
        t = time.time()
        while time.time() - t < wait:
            rclpy.spin_once(n, timeout_sec=0.02)
            if int((time.time() - t) * 2) != int((time.time() - t - 0.02) * 2):
                _, _, img = n.last
                cv2.imwrite(pv, cv2.resize(img, (960, 540)),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
                left = wait - (time.time() - t)
                print("  남은 시간 %4.0f초" % left, end="\r", flush=True)
        print("\n위치잡기 끝.")

    # ── 카운트다운 ──
    for c in (3, 2, 1):
        print("  %d..." % c, flush=True)
        t = time.time()
        while time.time() - t < 1.0:
            rclpy.spin_once(n, timeout_sec=0.02)
            if use_win:
                _, _, img = n.last
                sh = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
                cv2.putText(sh, str(c), (430, 320), cv2.FONT_HERSHEY_SIMPLEX,
                            5.0, (0, 200, 255), 10, cv2.LINE_AA)
                cv2.imshow("latency - 폰 스톱워치를 화면에 잡아라", sh)
                cv2.waitKey(1)

    print("\n  ★ 시작!  (지금 스톱워치를 눌러라)\n", flush=True)
    T0 = time.time()

    shots = []
    for target in SHOT_AT:
        while time.time() - T0 < target:
            rclpy.spin_once(n, timeout_sec=0.01)
            if use_win:
                _, _, img = n.last
                sh = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
                cv2.putText(sh, "%.1f s" % (time.time() - T0), (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.imshow("latency - 폰 스톱워치를 화면에 잡아라", sh)
                cv2.waitKey(1)
        rx, st, img = n.last
        shots.append((target, rx - T0, st - T0, img))
        print("    %.1fs 촬영   도착 %+.3fs   stamp %+.3fs"
              % (target, rx - T0, st - T0), flush=True)

    if use_win:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    n.destroy_node()
    rclpy.shutdown()

    print("\n저장:")
    for i, (target, d_rx, d_st, img) in enumerate(shots, 1):
        out = label(img, ["arrive = %+.3f s" % d_rx, "stamp  = %+.3f s" % d_st])
        p = os.path.join(OUTDIR, "shot%d.jpg" % i)
        cv2.imwrite(p, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("  %s" % p)

    print()
    print("=" * 62)
    print("  사진 속 스톱워치 숫자를 위의 두 값과 비교해라:")
    print()
    print("    스톱워치 ≈ stamp   →  진짜 지연이다 (문제)")
    print("    스톱워치 ≈ arrive  →  stamp 만 틀렸다 (무해)")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
