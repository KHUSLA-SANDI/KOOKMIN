#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""원본 vs 왜곡보정 나란히 보기 — 보정이 맞는지 눈으로 판정한다.

역할: /image_raw 를 왼쪽엔 원본 그대로, 오른쪽엔 config/camera.yaml 의 K/D 로
      보정해서 나란히 띄운다. 어느 쪽 화면이든 두 점을 클릭하면 **완벽한 직선**을
      긋는다 — 바닥의 테이프/줄눈이 그 직선을 따라가는지 보면 끝이다.
      원본에서는 휘고(배럴), 보정영상에서는 곧아야 정상이다.

판정
  · 보정영상에서 직선 자와 테이프가 붙으면 → 보정이 맞다
  · 보정영상에서도 테이프가 자에서 벗어나면 → 보정이 부족하거나 과하다
  · [t] 로 보정을 껐다 켰다 하면서 같은 자에 대고 비교하면 제일 확실하다

alpha 모드 ([n] 키)
  yaml   : camera.yaml 의 new_K 그대로 (주행 노드가 실제로 쓰는 값)
  a=0    : 검은 여백 없이 크롭
  a=1    : 원본 화소 전부 보존 — 화면 테두리가 어떻게 펴지는지 가장 잘 보인다

실행 (차 PC):
  cd ~/xycar_ws/src/track_drive
  source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=7
  python3 tools/undist_check.py --topic /image_raw
"""
import argparse
import os
import sys
import threading
import time

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.realpath(__file__))
_PKG = os.path.dirname(_HERE)
sys.path.insert(0, _PKG)
from track_drive.lib import bev as B                      # noqa: E402


def _load_text():
    import importlib.util
    p = os.path.join(_HERE, "tuner.py")
    if not os.path.isfile(p):
        return None
    try:
        spec = importlib.util.spec_from_file_location("_tuner_text", p)
        m = importlib.util.module_from_spec(spec)
        sys.modules["_tuner_text"] = m
        old, sys.argv = sys.argv, ["tuner.py"]
        try:
            spec.loader.exec_module(m)
        finally:
            sys.argv = old
        return getattr(m, "Text", None)
    except Exception:
        return None


_TextCls = _load_text()
TXT = _TextCls() if _TextCls is not None else None


def draw_text(img, items):
    if TXT is not None:
        return TXT.draw(img, items)
    for t, x, y, s, c in items:
        cv2.putText(img, t, (int(x), int(y + s)), cv2.FONT_HERSHEY_SIMPLEX,
                    s / 30.0, c, 1, cv2.LINE_AA)
    return img


class Source:
    def __init__(self, args):
        self._lock = threading.Lock()
        self._frame = None
        if args.image:
            f = cv2.imread(args.image)
            if f is None:
                print("사진을 못 읽었다:", args.image); sys.exit(1)
            self._frame = f
            self.kind = "static"
        elif args.device is not None:
            self.kind = "cam"
            self.cap = cv2.VideoCapture(int(args.device))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not self.cap.isOpened():
                print("USB 캠을 못 열었다"); sys.exit(1)
        else:
            self.kind = "ros"
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image
            from track_drive.lib import preprocess as PP
            rclpy.init()
            node = Node("undist_check")

            def cb(msg):
                bgr, _i = PP.image_msg_to_bgr(msg)
                if bgr is not None:
                    with self._lock:
                        self._frame = bgr
            node.create_subscription(Image, args.topic, cb, qos_profile_sensor_data)
            self._node, self._rclpy = node, rclpy
            threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()
            print("구독 %s — 첫 프레임 대기…" % args.topic)

    def read(self):
        if self.kind == "cam":
            ok, f = self.cap.read()
            return f if ok else None
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self):
        if self.kind == "cam":
            self.cap.release()
        elif self.kind == "ros":
            try:
                self._node.destroy_node()
                if self._rclpy.ok():
                    self._rclpy.shutdown()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="원본 vs 왜곡보정 비교")
    ap.add_argument("--camera", default=os.path.join(_PKG, "config", "camera.yaml"))
    ap.add_argument("--topic", default="/image_raw")
    ap.add_argument("--device", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default=os.path.expanduser("~/undist_shots"))
    args = ap.parse_args()

    cam = B.load_camera_config(args.camera)
    K = np.asarray(cam["K"], np.float64)
    D = np.ravel(cam["D"])[:5].astype(np.float64)
    print("K  fx=%.2f fy=%.2f cx=%.2f cy=%.2f" % (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    print("D  k1=%.5f k2=%.5f p1=%.5f p2=%.5f k3=%.5f" % tuple(D))
    print("intrinsics_calibrated=%s" % cam.get("intrinsics_calibrated"))
    os.makedirs(args.out, exist_ok=True)

    src = Source(args)
    MODES = ["yaml", "a=0", "a=1"]
    mode_i = 0
    maps = {}
    size = None

    def get_map(mode, size):
        key = (mode, size)
        if key in maps:
            return maps[key]
        if mode == "yaml":
            nk = cam.get("new_K")
            nk = np.asarray(nk, np.float64).reshape(3, 3) if nk is not None else \
                np.asarray(cv2.getOptimalNewCameraMatrix(K, D, size, 0.0)[0], np.float64)
        else:
            a = 0.0 if mode == "a=0" else 1.0
            nk = np.asarray(cv2.getOptimalNewCameraMatrix(K, D, size, a)[0], np.float64)
        m1, m2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), nk, size, cv2.CV_16SC2)
        maps[key] = (m1, m2, nk)
        return maps[key]

    frozen = None
    rulers = {0: [], 1: []}      # 화면별(0=원본, 1=보정) 직선 자
    pending = {0: None, 1: None}
    undist_on = True
    show_grid = False
    msg, msg_t = "", 0.0
    st = {"w": 640, "s": 1.0}
    win = "raw vs undistorted"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(ev, x, y, flags, _):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        pane = 0 if x < st["w"] else 1
        px = (x - (0 if pane == 0 else st["w"])) / st["s"]
        py = y / st["s"]
        if pending[pane] is None:
            pending[pane] = (px, py)
        else:
            rulers[pane].append((pending[pane], (px, py)))
            pending[pane] = None
    cv2.setMouseCallback(win, on_mouse)

    print(__doc__.split("실행")[0])
    while True:
        live = src.read()
        if live is None:
            blank = np.full((180, 720, 3), 25, np.uint8)
            draw_text(blank, [("이미지 대기중… 카메라 노드 확인", 20, 80, 18, (200, 200, 200))])
            cv2.imshow(win, blank)
            if (cv2.waitKeyEx(100) & 0xFF) in (27, ord('q')):
                break
            continue
        frame = frozen if frozen is not None else live
        if size is None:
            size = (frame.shape[1], frame.shape[0])
        m1, m2, nk = get_map(MODES[mode_i], size)
        und = cv2.remap(frame, m1, m2, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT) if undist_on else frame.copy()

        panes = [frame.copy(), und]
        for pi, im in enumerate(panes):
            if show_grid:
                for gx in range(0, im.shape[1], 40):
                    cv2.line(im, (gx, 0), (gx, im.shape[0] - 1), (90, 90, 90), 1)
                for gy in range(0, im.shape[0], 40):
                    cv2.line(im, (0, gy), (im.shape[1] - 1, gy), (90, 90, 90), 1)
            for a, b in rulers[pi]:
                cv2.line(im, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                         (0, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(im, (int(a[0]), int(a[1])), 3, (0, 165, 255), -1)
                cv2.circle(im, (int(b[0]), int(b[1])), 3, (0, 165, 255), -1)
            if pending[pi] is not None:
                cv2.drawMarker(im, (int(pending[pi][0]), int(pending[pi][1])),
                               (0, 165, 255), cv2.MARKER_CROSS, 14, 2)

        th = 480
        s = th / panes[0].shape[0]
        L = cv2.resize(panes[0], (int(round(panes[0].shape[1] * s)), th))
        R = cv2.resize(panes[1], (int(round(panes[1].shape[1] * s)), th))
        st["w"], st["s"] = L.shape[1], s
        for im, lab, col in ((L, "원본 (보정 안 함)", (200, 200, 255)),
                             (R, "보정영상 [%s]%s" % (MODES[mode_i],
                                                   "" if undist_on else "  ← 보정 꺼짐"),
                              (200, 255, 200))):
            cv2.rectangle(im, (0, 0), (im.shape[1] - 1, 20), (0, 0, 0), -1)
            draw_text(im, [(lab, 5, 3, 14, col)])
        top = cv2.hconcat([L, R])

        pad = np.full((96, top.shape[1], 3), 24, np.uint8)
        it = [("아무 화면에서나 두 점을 클릭하면 완벽한 직선(노랑)이 그어진다 — "
               "바닥 테이프가 그 선을 따라가는지 보면 된다",
               8, 6, 14, (200, 220, 255)),
              ("[스페이스] 정지/실시간   [t] 보정 껐다켜기   [n] alpha 모드   "
               "[c] 자 지우기   [g] 격자   [f] 저장   [q] 종료",
               8, 28, 13, (150, 150, 150)),
              ("K fx=%.1f fy=%.1f cx=%.1f cy=%.1f | D k1=%+.4f k2=%+.4f p1=%+.4f "
               "p2=%+.4f k3=%+.4f | new_K fx=%.1f cx=%.1f"
               % (K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0], D[1], D[2], D[3], D[4],
                  nk[0, 0], nk[0, 2]), 8, 50, 12, (140, 200, 240))]
        if msg and time.time() - msg_t < 4.0:
            it.append((msg, 8, 70, 14, (120, 255, 255)))
        else:
            it.append(("원본에서는 휘고 보정영상에서는 곧아야 정상. "
                       "보정영상에서도 벗어나면 보정이 부족/과한 것.",
                       8, 70, 13, (170, 170, 170)))
        draw_text(pad, it)
        cv2.imshow(win, cv2.vconcat([top, pad]))

        k = cv2.waitKeyEx(30)
        if k == -1:
            continue
        ch = k & 0xFF
        if ch in (27, ord('q')):
            break
        elif ch == ord(' '):
            if frozen is None:
                frozen = live.copy()
                msg, msg_t = "정지 — 다시 스페이스로 실시간", time.time()
            else:
                frozen = None
                msg, msg_t = "실시간", time.time()
        elif ch == ord('t'):
            undist_on = not undist_on
            msg, msg_t = ("보정 %s" % ("켬" if undist_on else "끔")), time.time()
        elif ch == ord('n'):
            mode_i = (mode_i + 1) % len(MODES)
            msg, msg_t = "alpha 모드: %s" % MODES[mode_i], time.time()
        elif ch == ord('c'):
            rulers[0].clear(); rulers[1].clear()
            pending[0] = pending[1] = None
        elif ch == ord('g'):
            show_grid = not show_grid
        elif ch == ord('f'):
            t = time.strftime("%m%d_%H%M%S")
            cv2.imwrite(os.path.join(args.out, "raw_%s.jpg" % t), frame)
            cv2.imwrite(os.path.join(args.out, "und_%s.jpg" % t), und)
            cv2.imwrite(os.path.join(args.out, "cmp_%s.jpg" % t), top)
            msg, msg_t = "저장: %s/{raw,und,cmp}_%s.jpg" % (args.out, t), time.time()
            print("  " + msg)

    cv2.destroyAllWindows()
    src.close()


if __name__ == "__main__":
    main()
