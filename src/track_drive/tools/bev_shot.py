#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BEV 확인기 — 스페이스로 한 장 잡아 버드아이뷰로 보고, 클릭해서 실제 거리를 잰다.

역할: /image_raw 를 실시간으로 보다가 [스페이스] 를 누르면 그 프레임을 고정하고
      현재 camera.yaml 의 K/D/H 로 BEV 를 만들어 나란히 보여준다.
      BEV 위에서 두 점을 클릭하면 그 사이 **실제 거리(m)** 를 띄운다.
      → 자·A4·타일 같은 아는 물체를 놓고 재보면 캘리브 상태를 직접 판정할 수 있다.
입력: /image_raw (또는 --image 로 저장된 사진), config/camera.yaml
출력: 화면. [s] 로 raw/보정/BEV 3장 저장.

판정법
  · 노란 점선 = 타일 줄눈 예상 위치(피치 0.4533 m). 실제 줄눈과 겹치면 H 가 맞다.
  · 초록 실선 = 0.5 m 눈금.
  · 클릭 2점 = 그 사이 실제 거리. 아는 길이와 비교하면 축척이 맞는지 바로 나온다.
  · BEV 에서 직선이 곧게, 평행선이 평행하게 보여야 한다.

실행 (차 PC):
  cd ~/xycar_ws/src/track_drive
  source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=7
  python3 tools/bev_shot.py --topic /image_raw
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
    """한글 렌더러를 tuner.py 에서 경로로 직접 로드 (tools/ 에 __init__.py 가 없음)."""
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
        self.kind = "static"
        if args.image:
            img = cv2.imread(args.image)
            if img is None:
                print("사진을 못 읽었다:", args.image)
                sys.exit(1)
            self._frame = img
        elif args.device is not None:
            self.kind = "cam"
            self.cap = cv2.VideoCapture(int(args.device))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not self.cap.isOpened():
                print("USB 캠을 못 열었다:", args.device)
                sys.exit(1)
        else:
            self.kind = "ros"
            self._start_ros(args.topic)

    def _start_ros(self, topic):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from track_drive.lib import preprocess as PP
        rclpy.init()
        node = Node("bev_shot")

        def cb(msg):
            bgr, _info = PP.image_msg_to_bgr(msg)
            if bgr is not None:
                with self._lock:
                    self._frame = bgr
        node.create_subscription(Image, topic, cb, qos_profile_sensor_data)
        self._node, self._rclpy = node, rclpy
        threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()
        print("구독 %s — 첫 프레임 대기…" % topic)

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


class BevView:
    """지면 범위 ↔ BEV 픽셀 변환과 역매핑 테이블을 함께 들고 있는다."""

    def __init__(self, H, xr, yr, ppm):
        self.H = np.asarray(H, np.float64)
        self.set_range(xr, yr, ppm)

    def set_range(self, xr, yr, ppm):
        self.x0, self.x1 = float(xr[0]), float(xr[1])
        self.y0, self.y1 = float(yr[0]), float(yr[1])
        self.ppm = float(ppm)
        self.W = max(8, int(round((self.y1 - self.y0) * self.ppm)))
        self.Hh = max(8, int(round((self.x1 - self.x0) * self.ppm)))
        ys = self.y1 - np.arange(self.W, dtype=np.float64) / self.ppm
        xs = self.x1 - np.arange(self.Hh, dtype=np.float64) / self.ppm
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        G = np.stack([XX, YY, np.ones_like(XX)], -1).reshape(-1, 3).T
        try:
            Hi = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            self.map = None
            return
        P = Hi @ G
        wref = np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2]) or 1.0
        w = P[2]
        bad = (np.abs(w) < 1e-12) | (np.sign(w) != wref)
        w = np.where(bad, 1.0, w)
        mu = (P[0] / w).reshape(self.Hh, self.W).astype(np.float32)
        mv = (P[1] / w).reshape(self.Hh, self.W).astype(np.float32)
        mu[bad.reshape(self.Hh, self.W)] = -1
        mv[bad.reshape(self.Hh, self.W)] = -1
        self.map = (mu, mv)

    def px_to_ground(self, c, r):
        return (self.x1 - r / self.ppm, self.y1 - c / self.ppm)

    def ground_to_px(self, x, y):
        return ((self.y1 - y) * self.ppm, (self.x1 - x) * self.ppm)

    def render(self, und):
        if self.map is None:
            return np.zeros((self.Hh, self.W, 3), np.uint8)
        return cv2.remap(und, self.map[0], self.map[1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def draw_grid(self, img, pitch):
        # 타일 줄눈 예상 위치 (노랑 점선) — 실제 줄눈과 겹치는지가 판정 기준
        k = int(np.ceil(self.x0 / pitch))
        while k * pitch <= self.x1:
            r = int(round((self.x1 - k * pitch) * self.ppm))
            for c in range(0, img.shape[1], 14):
                cv2.line(img, (c, r), (min(c + 7, img.shape[1] - 1), r),
                         (0, 255, 255), 1)
            k += 1
        k = int(np.ceil(self.y0 / pitch))
        while k * pitch <= self.y1:
            c = int(round((self.y1 - k * pitch) * self.ppm))
            for r in range(0, img.shape[0], 14):
                cv2.line(img, (c, r), (c, min(r + 7, img.shape[0] - 1)),
                         (0, 255, 255), 1)
            k += 1
        # 0.5 m 눈금 (초록)
        it = []
        d = np.ceil(self.x0 / 0.5) * 0.5
        while d <= self.x1:
            r = int(round((self.x1 - d) * self.ppm))
            cv2.line(img, (0, r), (img.shape[1] - 1, r), (0, 190, 0), 1)
            it.append(("x=%.1f" % d, 3, max(0, r - 15), 12, (0, 255, 0)))
            d += 0.5
        d = np.ceil(self.y0 / 0.5) * 0.5
        while d <= self.y1:
            c = int(round((self.y1 - d) * self.ppm))
            cv2.line(img, (c, 0), (c, img.shape[0] - 1), (0, 190, 0), 1)
            it.append(("y=%.1f" % d, min(c + 3, img.shape[1] - 40),
                       img.shape[0] - 18, 12, (0, 255, 0)))
            d += 0.5
        draw_text(img, it)


def main():
    ap = argparse.ArgumentParser(description="BEV 확인기 (스페이스 캡처 + 클릭 거리측정)")
    ap.add_argument("--camera", default=os.path.join(_PKG, "config", "camera.yaml"))
    ap.add_argument("--topic", default="/image_raw")
    ap.add_argument("--device", default=None, help="USB 캠 번호")
    ap.add_argument("--image", default=None, help="저장된 사진으로 확인")
    ap.add_argument("--range-x", type=float, nargs=2, default=[0.0, 2.0])
    ap.add_argument("--range-y", type=float, nargs=2, default=[-1.0, 1.0])
    ap.add_argument("--ppm", type=float, default=260.0, help="BEV 픽셀/미터")
    ap.add_argument("--pitch", type=float, default=0.453333, help="타일 피치 m")
    ap.add_argument("--out", default=os.path.expanduser("~/bev_shots"))
    args = ap.parse_args()

    cam = B.load_camera_config(args.camera)
    print("camera.yaml: intrinsics=%s h=%s model=%s"
          % (cam.get("intrinsics_calibrated"), cam.get("h_calibrated"), cam.get("model")))
    if cam.get("note"):
        print("  note:", cam["note"])
    H = np.asarray(cam["H"], np.float64)
    os.makedirs(args.out, exist_ok=True)

    src = Source(args)
    undist = None
    view = BevView(H, args.range_x, args.range_y, args.ppm)

    frozen = None           # 고정된 raw 프레임
    picks = []              # BEV 위 클릭 (지면 좌표)
    show_grid = True
    msg, msg_t = "", 0.0
    win = "BEV shot"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    state = {"bev_off": (0, 0), "bev_scale": 1.0}

    def on_mouse(ev, x, y, flags, _):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        ox, oy = state["bev_off"]
        s = state["bev_scale"]
        c, r = (x - ox) / s, (y - oy) / s
        if not (0 <= c < view.W and 0 <= r < view.Hh):
            return
        picks.append(view.px_to_ground(c, r))
        if len(picks) > 2:
            del picks[0]
    cv2.setMouseCallback(win, on_mouse)

    print(__doc__.split("실행")[0])
    while True:
        live = src.read()
        if live is None:
            blank = np.full((200, 700, 3), 25, np.uint8)
            draw_text(blank, [("이미지 대기중… 카메라 노드가 켜져 있는지 확인",
                               20, 90, 18, (200, 200, 200))])
            cv2.imshow(win, blank)
            if (cv2.waitKeyEx(100) & 0xFF) in (27, ord('q')):
                break
            continue

        if undist is None:
            undist = B.Undistorter.from_config(cam, (live.shape[1], live.shape[0]))
            print("undistort=%s  BEV %dx%d px (%.2f~%.2f m 전방, %.2f~%.2f m 좌우, %.0f px/m)"
                  % (undist.enabled, view.W, view.Hh, view.x0, view.x1,
                     view.y0, view.y1, view.ppm))

        frame = frozen if frozen is not None else live
        und = undist.apply(frame)
        bev = view.render(und)
        if show_grid:
            view.draw_grid(bev, args.pitch)

        # 클릭한 점 / 잰 거리
        for i, (gx, gy) in enumerate(picks):
            c, r = view.ground_to_px(gx, gy)
            cv2.drawMarker(bev, (int(c), int(r)), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        dist_txt = ""
        if len(picks) == 2:
            (ax, ay), (bx, by) = picks
            c0, r0 = view.ground_to_px(ax, ay)
            c1, r1 = view.ground_to_px(bx, by)
            cv2.line(bev, (int(c0), int(r0)), (int(c1), int(r1)), (0, 0, 255), 2)
            d = float(np.hypot(bx - ax, by - ay))
            dist_txt = ("잰 거리 %.1f mm  (전방차 %.1f mm, 좌우차 %.1f mm)"
                        % (1000 * d, 1000 * (bx - ax), 1000 * (by - ay)))
            draw_text(bev, [("%.1f mm" % (1000 * d),
                             int((c0 + c1) / 2) + 6, int((r0 + r1) / 2) - 18,
                             16, (0, 0, 255))])

        # 좌: 보정영상 / 우: BEV
        th = 480
        sL = th / und.shape[0]
        L = cv2.resize(und, (int(round(und.shape[1] * sL)), th))
        sB = th / bev.shape[0]
        Bp = cv2.resize(bev, (max(1, int(round(bev.shape[1] * sB))), th))
        state["bev_off"] = (L.shape[1], 0)
        state["bev_scale"] = sB
        for im, lab in ((L, "보정영상 %s" % ("[정지]" if frozen is not None else "[실시간]")),
                        (Bp, "BEV — 노랑=타일줄눈 예상, 초록=0.5m")):
            cv2.rectangle(im, (0, 0), (im.shape[1] - 1, 20), (0, 0, 0), -1)
            draw_text(im, [(lab, 5, 3, 14, (255, 255, 255))])
        top = cv2.hconcat([L, Bp])

        pad = np.full((92, top.shape[1], 3), 24, np.uint8)
        it = [("[스페이스] 잡기/풀기   [c] 잰 점 지우기   [g] 격자   "
               "[+/-] 확대   [w/s] 전방범위   [a/d] 좌우범위   [f] 저장   [q] 종료",
               8, 6, 13, (150, 150, 150)),
              ("범위: 전방 %.2f~%.2f m / 좌우 %.2f~%.2f m / %.0f px/m"
               % (view.x0, view.x1, view.y0, view.y1, view.ppm),
               8, 26, 14, (120, 220, 255))]
        if dist_txt:
            it.append((dist_txt, 8, 48, 16, (120, 255, 160)))
        else:
            it.append(("BEV 위에서 두 점을 클릭하면 그 사이 실제 거리가 나온다 "
                       "— 아는 물체(자·A4·타일)로 재보면 축척이 맞는지 바로 안다",
                       8, 48, 13, (180, 180, 180)))
        if msg and time.time() - msg_t < 4.0:
            it.append((msg, 8, 70, 14, (120, 255, 255)))
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
                msg, msg_t = "프레임 고정 — 다시 스페이스 누르면 실시간", time.time()
            else:
                frozen = None
                picks.clear()
                msg, msg_t = "실시간으로 복귀", time.time()
        elif ch == ord('c'):
            picks.clear()
        elif ch == ord('g'):
            show_grid = not show_grid
        elif ch in (ord('+'), ord('=')):
            view.set_range((view.x0, view.x1), (view.y0, view.y1), view.ppm * 1.25)
        elif ch in (ord('-'), ord('_')):
            view.set_range((view.x0, view.x1), (view.y0, view.y1),
                           max(40.0, view.ppm / 1.25))
        elif ch == ord('w'):
            view.set_range((view.x0, view.x1 + 0.25), (view.y0, view.y1), view.ppm)
        elif ch == ord('s'):
            view.set_range((view.x0, max(view.x0 + 0.3, view.x1 - 0.25)),
                           (view.y0, view.y1), view.ppm)
        elif ch == ord('d'):
            view.set_range((view.x0, view.x1), (view.y0 - 0.25, view.y1 + 0.25), view.ppm)
        elif ch == ord('a'):
            ny = max(0.15, (view.y1 - view.y0) / 2 - 0.25)
            view.set_range((view.x0, view.x1), (-ny, ny), view.ppm)
        elif ch == ord('f'):
            t = time.strftime("%m%d_%H%M%S")
            cv2.imwrite(os.path.join(args.out, "raw_%s.jpg" % t), frame)
            cv2.imwrite(os.path.join(args.out, "und_%s.jpg" % t), und)
            cv2.imwrite(os.path.join(args.out, "bev_%s.jpg" % t), bev)
            msg, msg_t = "저장: %s/{raw,und,bev}_%s.jpg" % (args.out, t), time.time()
            print("  " + msg)

    cv2.destroyAllWindows()
    src.close()


if __name__ == "__main__":
    main()
