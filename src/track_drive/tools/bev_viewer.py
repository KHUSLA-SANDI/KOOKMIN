#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실시간 BEV 뷰어 — 캘리브 검증용.

역할 : /image_raw 를 받아 (undistort → 지면 H → 버드아이뷰) 를 실시간 표시한다.
입력 : /image_raw (sensor_msgs/Image), config/camera.yaml (K/D/H/new_K)
출력 : 화면 창 3분할 [RAW | UNDIST | BEV]  (--save 주면 캡처 저장)

검증 포인트
  · BEV 에서 바닥 타일이 **정사각형**으로 보이고, 노란 격자(피치 단위)와 줄눈이 겹치면 H 가 맞다.
  · 초록 눈금은 0.5 m 간격. BEV 세로축 위쪽이 전방(+x), 가로축 왼쪽이 +y.
  · 좌표 원점은 camera.yaml 의 h_frame (lidar_frame 이면 라이다 지면투영점).

실행 (노트북 WSL 권장 — WSLg 로 창이 뜬다):
    source /opt/ros/humble/setup.bash
    export ROS_DOMAIN_ID=7
    python3 bev_viewer.py --camera ../config/camera.yaml
키 : [s] 현재 화면 저장   [g] 격자 on/off   [e] 대비강화 on/off   [q] 종료
"""
import argparse
import os
import sys
import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # track_drive 패키지 루트
from track_drive.lib import bev as B                # noqa: E402
from track_drive.lib import preprocess as PP        # noqa: E402


def enhance(img):
    """테라조 줄눈처럼 대비가 약한 바닥을 보이게 (표시 전용, 기하 불변)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 3)
    return cv2.addWeighted(out, 1.6, blur, -0.6, 0)


class BevViewer(Node):
    def __init__(self, args):
        super().__init__("bev_viewer")
        self.args = args
        self.cam = B.load_camera_config(args.camera)
        self.get_logger().info(
            "[camera] model=%s intrinsics=%s h=%s frame=%s"
            % (self.cam.get("model"), self.cam.get("intrinsics_calibrated"),
               self.cam.get("h_calibrated"), self.cam.get("h_frame", "?")))
        if not self.cam.get("h_calibrated"):
            self.get_logger().warn("H 미보정 — BEV 는 시뮬 폴백값이라 의미 없음")

        self.H = np.array(self.cam["H"], np.float64)
        self._undist = None
        self._map = None
        self.show_grid = True
        self.do_enh = args.enhance
        self.n = 0
        self.t0 = time.time()

        self.create_subscription(Image, args.topic, self._on_image,
                                 qos_profile_sensor_data)
        self.get_logger().info("구독: %s — 창이 뜰 때까지 잠시 기다리세요" % args.topic)

    # ---------- BEV 역매핑 테이블 (한 번만 만들고 재사용) ----------
    def _build_map(self, size):
        x0, x1 = self.args.range_x
        y0, y1 = self.args.range_y
        ppm = self.args.ppm
        W = int(round((y1 - y0) * ppm))
        Hh = int(round((x1 - x0) * ppm))
        # BEV 픽셀 (col, row) → 지면 (x, y)
        cols = np.arange(W, dtype=np.float64)
        rows = np.arange(Hh, dtype=np.float64)
        ys = y1 - cols / ppm                 # 왼쪽이 +y
        xs = x1 - rows / ppm                 # 위쪽이 먼 쪽(+x)
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        ones = np.ones_like(XX)
        G = np.stack([XX, YY, ones], axis=-1).reshape(-1, 3).T
        Hi = np.linalg.inv(self.H)
        Pq = Hi @ G
        wref = np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2])
        w = Pq[2]
        bad = (np.abs(w) < 1e-12) | (np.sign(w) != wref)
        w = np.where(bad, 1.0, w)
        mu = (Pq[0] / w).reshape(Hh, W).astype(np.float32)
        mv = (Pq[1] / w).reshape(Hh, W).astype(np.float32)
        mu[bad.reshape(Hh, W)] = -1
        mv[bad.reshape(Hh, W)] = -1
        self._map = (mu, mv, W, Hh)

    def _draw_grid(self, bevimg):
        x0, x1 = self.args.range_x
        y0, y1 = self.args.range_y
        ppm = self.args.ppm
        p = self.args.pitch
        # 타일 격자 (노랑 점선)
        k = int(np.ceil(x0 / p))
        while k * p <= x1:
            r = int(round((x1 - k * p) * ppm))
            for cx in range(0, bevimg.shape[1], 14):
                cv2.line(bevimg, (cx, r), (min(cx + 7, bevimg.shape[1] - 1), r),
                         (0, 255, 255), 1)
            k += 1
        k = int(np.ceil(y0 / p))
        while k * p <= y1:
            c = int(round((y1 - k * p) * ppm))
            for cy in range(0, bevimg.shape[0], 14):
                cv2.line(bevimg, (c, cy), (c, min(cy + 7, bevimg.shape[0] - 1)),
                         (0, 255, 255), 1)
            k += 1
        # 0.5 m 눈금 (초록 실선 + 라벨)
        d = x0 - (x0 % 0.5) + 0.5
        while d <= x1:
            r = int(round((x1 - d) * ppm))
            cv2.line(bevimg, (0, r), (bevimg.shape[1] - 1, r), (0, 200, 0), 1)
            cv2.putText(bevimg, "x=%.1f" % d, (3, r - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            d += 0.5
        d = y0 - (y0 % 0.5) + 0.5
        while d <= y1:
            c = int(round((y1 - d) * ppm))
            cv2.line(bevimg, (c, 0), (c, bevimg.shape[0] - 1), (0, 200, 0), 1)
            cv2.putText(bevimg, "y=%.1f" % d, (c + 3, bevimg.shape[0] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            d += 0.5

    def _on_image(self, msg):
        bgr, info = PP.image_msg_to_bgr(msg)
        if bgr is None:
            self.get_logger().warn("이미지 변환 실패: %s" % info, throttle_duration_sec=5.0)
            return
        h, w = bgr.shape[:2]
        if self._undist is None:
            self._undist = B.Undistorter.from_config(self.cam, (w, h))
            self._build_map((w, h))
            self.get_logger().info(
                "undistort=%s  BEV %dx%d (%.1f~%.1f m 전방, %.1f~%.1f m 좌우, %d px/m)"
                % (self._undist.enabled, self._map[2], self._map[3],
                   self.args.range_x[0], self.args.range_x[1],
                   self.args.range_y[0], self.args.range_y[1], self.args.ppm))

        und = self._undist.apply(bgr)
        src = enhance(und) if self.do_enh else und
        mu, mv, W, Hh = self._map
        bevimg = cv2.remap(src, mu, mv, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        if self.show_grid:
            self._draw_grid(bevimg)

        # 3분할 표시 (높이 맞춤)
        th = 380
        def fit(im, label):
            s = th / im.shape[0]
            r = cv2.resize(im, (int(im.shape[1] * s), th))
            cv2.rectangle(r, (0, 0), (r.shape[1] - 1, 18), (0, 0, 0), -1)
            cv2.putText(r, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
            return r
        panel = cv2.hconcat([fit(bgr, "RAW"), fit(und, "UNDIST"),
                             fit(bevimg, "BEV (tiles must be SQUARE)")])

        self.n += 1
        fps = self.n / max(1e-6, time.time() - self.t0)
        cv2.rectangle(panel, (0, panel.shape[0] - 20),
                      (panel.shape[1], panel.shape[0]), (0, 0, 0), -1)
        cv2.putText(panel, "%.1f fps | h_frame=%s rms=%s | [s]save [g]grid [e]enh [q]quit"
                    % (fps, self.cam.get("h_frame", "?"), self.cam.get("h_rms_m", "?")),
                    (4, panel.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 255, 200), 1)

        if self.args.save and self.n == 1:
            cv2.imwrite(self.args.save, panel)
            self.get_logger().info("저장: %s" % self.args.save)

        cv2.imshow("BEV viewer", panel)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            raise SystemExit
        elif k == ord('g'):
            self.show_grid = not self.show_grid
        elif k == ord('e'):
            self.do_enh = not self.do_enh
        elif k == ord('s'):
            fn = "/tmp/bev_%d.jpg" % int(time.time())
            cv2.imwrite(fn, panel)
            self.get_logger().info("저장: %s" % fn)


def main():
    ap = argparse.ArgumentParser(description="실시간 BEV 뷰어 (캘리브 검증)")
    ap.add_argument("--camera", default=os.path.join(_HERE, "..", "config", "camera.yaml"))
    ap.add_argument("--topic", default="/image_raw")
    ap.add_argument("--pitch", type=float, default=0.453333, help="타일 피치 m")
    ap.add_argument("--range-x", type=float, nargs=2, default=[0.2, 2.5])
    ap.add_argument("--range-y", type=float, nargs=2, default=[-1.2, 1.2])
    ap.add_argument("--ppm", type=int, default=200, help="BEV 픽셀/미터")
    ap.add_argument("--enhance", action="store_true", help="줄눈 대비 강화로 시작")
    ap.add_argument("--save", default=None, help="첫 프레임을 이 경로에 저장")
    args = ap.parse_args()

    if not os.path.exists(args.camera):
        print("camera.yaml 없음:", args.camera); sys.exit(1)

    rclpy.init()
    node = BevViewer(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
