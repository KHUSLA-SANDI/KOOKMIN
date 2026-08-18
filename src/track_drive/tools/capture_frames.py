#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""캘리브용 프레임 캡처 도구 (독립 CLI — --topic 사용 시에만 rclpy import).

역할: USB 카메라(--device) 또는 ROS2 이미지 토픽(--topic)을 미리보기로 띄우고
      스페이스를 누를 때마다 jpg 연번으로 저장 (체커보드/바닥 기준점 촬영용).
입력: --device N 또는 --topic /image_raw 중 하나, --out 저장 폴더.
출력: <out>/frame_0000.jpg, frame_0001.jpg ... (폴더에 이미 있으면 연번 이어감)
예선 대비 변경점: 신규 도구 (시뮬 카메라는 무왜곡이라 캘리브 자체가 없었음).

사용법:
  # 노트북에 USB 캠 직결
  python3 capture_frames.py --device 0 --out calib_imgs
  # 차 위(ROS2 구동 중)에서 토픽 구독 — /image_raw 는 rgb8 (다른 인코딩이면 경고)
  python3 capture_frames.py --topic /image_raw --out calib_imgs
  키: [스페이스] 현재 프레임 저장   [q]/[ESC] 종료

170도 어안 체커보드 촬영 팁:
  - 20~40장, 화면 중앙~가장자리 골고루 + 다양한 기울기/거리.
  - 극가장자리는 왜곡이 심해 코너 검출이 자주 실패 — calib_fisheye.py 가
    실패 이미지를 자동 제외하므로 넉넉히 찍어 두는 편이 낫다.
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np

WIN = "capture_frames"
JPEG_QUALITY = 95        # jpg 저장 품질
CAM_W, CAM_H = 640, 480  # 실차 카메라 해상도


def _next_index(out_dir, prefix):
    """기존 저장분 연번 이어가기: frame_0007.jpg 까지 있으면 8 반환."""
    idx = -1
    for name in os.listdir(out_dir):
        m = re.match(re.escape(prefix) + r"(\d+)\.jpe?g$", name, re.IGNORECASE)
        if m:
            idx = max(idx, int(m.group(1)))
    return idx + 1


def _overlay(frame, saved, next_name):
    """상단에 저장 상태 표시 (cv2.putText 는 한글 불가 — 영문 고정)."""
    disp = frame.copy()
    txt = "saved %d | next %s | [SPACE] save  [q] quit" % (saved, next_name)
    cv2.rectangle(disp, (0, 0), (disp.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(disp, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1, cv2.LINE_AA)
    return disp


def _save(frame, out_dir, prefix, idx):
    path = os.path.join(out_dir, "%s%04d.jpg" % (prefix, idx))
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    print("저장: %s" % path)


def run_device(args, out_dir):
    """USB 캠 직결 모드 (ROS 불필요)."""
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        sys.exit("카메라를 열 수 없음: --device %d" % args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    idx = _next_index(out_dir, args.prefix)
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[경고] 프레임 읽기 실패 — 재시도")
            if cv2.waitKey(100) & 0xFF in (ord("q"), 27):
                break
            continue
        next_name = "%s%04d.jpg" % (args.prefix, idx)
        cv2.imshow(WIN, _overlay(frame, saved, next_name))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            _save(frame, out_dir, args.prefix, idx)
            idx += 1
            saved += 1
    cap.release()
    cv2.destroyAllWindows()
    print("종료 — 이번 세션 저장 %d장" % saved)


def _img_msg_to_bgr(msg, warned_encodings):
    """sensor_msgs/Image → BGR. step(행 stride) 고려.

    예선 버그 8) 재발 방지: 모르는 인코딩을 무음 드랍하지 않고 반드시 경고.
    """
    enc = msg.encoding.lower()
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    try:
        if enc in ("rgb8", "bgr8"):
            img = buf.reshape(h, step)[:, :w * 3].reshape(h, w, 3)
            return img[..., ::-1].copy() if enc == "rgb8" else img.copy()
        if enc == "mono8":
            return cv2.cvtColor(buf.reshape(h, step)[:, :w], cv2.COLOR_GRAY2BGR)
        if enc in ("yuyv", "yuv422_yuy2", "yuv422"):
            img = buf.reshape(h, step)[:, :w * 2].reshape(h, w, 2)
            return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUY2)
    except ValueError:
        pass  # step/크기 불일치 — 아래 경고로 합류
    if enc not in warned_encodings:
        warned_encodings.add(enc)
        print("[경고] 지원하지 않는 인코딩 '%s' (%dx%d step=%d) — 프레임 무시됨, rgb8 기대"
              % (enc, w, h, step))
    return None


def run_topic(args, out_dir):
    """ROS2 토픽 구독 모드 — 이 함수 안에서만 rclpy import."""
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    rclpy.init()
    node = rclpy.create_node("capture_frames")
    state = {"frame": None}
    warned = set()

    def cb(msg):
        bgr = _img_msg_to_bgr(msg, warned)
        if bgr is not None:
            state["frame"] = bgr

    # 카메라 QoS가 RELIABLE 이어도 BEST_EFFORT 구독은 호환됨 → sensor_data 로 통일
    node.create_subscription(Image, args.topic, cb, qos_profile_sensor_data)

    idx = _next_index(out_dir, args.prefix)
    saved = 0
    waiting = np.zeros((CAM_H, CAM_W, 3), np.uint8)
    cv2.putText(waiting, "waiting for %s ..." % args.topic, (40, CAM_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            frame = state["frame"]
            next_name = "%s%04d.jpg" % (args.prefix, idx)
            disp = _overlay(frame, saved, next_name) if frame is not None else waiting
            cv2.imshow(WIN, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if frame is None:
                    print("[경고] 아직 프레임 수신 없음 — 저장 생략")
                    continue
                _save(frame, out_dir, args.prefix, idx)
                idx += 1
                saved += 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
    print("종료 — 이번 세션 저장 %d장" % saved)


def main():
    ap = argparse.ArgumentParser(
        description="캘리브용 프레임 캡처 — 스페이스 저장 / q 종료 (자세한 사용법은 파일 헤더)")
    ap.add_argument("--device", type=int, default=None,
                    help="USB 캠 번호 (예: 0). --topic 과 둘 중 하나만")
    ap.add_argument("--topic", type=str, default=None,
                    help="ROS2 이미지 토픽 (예: /image_raw). 이때만 rclpy 사용")
    ap.add_argument("--out", type=str, default="calib_imgs", help="저장 폴더")
    ap.add_argument("--prefix", type=str, default="frame_", help="파일명 접두사")
    args = ap.parse_args()

    if (args.device is None) == (args.topic is None):
        ap.error("--device 또는 --topic 중 정확히 하나를 지정")

    os.makedirs(args.out, exist_ok=True)
    if args.topic is not None:
        run_topic(args, args.out)
    else:
        run_device(args, args.out)


if __name__ == "__main__":
    main()
