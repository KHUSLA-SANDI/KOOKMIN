#!/usr/bin/env python3
"""실차 주행 뷰어 (2패널) — 원본 사진 + BEV.

역할: 지금 차가 "무엇을 보고 어디로 가려는지"만 한 화면에 보여준다.
      기존 yolo_lane_viewer 는 흰선·오프셋 후보까지 다 그려 판단이 흐려져서,
      주행 검증에 필요한 것만 남긴 얇은 버전이다.

입력: /image_raw(sensor QoS), /detect/lane(PoseArray, z=8 이 노란 중앙선),
      /center_path(PoseArray, 갈 경로), /target(PointStamped, 목표점),
      /scan(LaserScan, 장애물), /xycar_motor(실제 발행 조향/속도),
      /detect/lane_quality(Float32)
출력: 창 하나 (imshow). 발행 없음 — 주행에 영향 주지 않는다.

BEV 패널이 보여주는 것
  · 노란 점   = YOLO 가 잡은 중앙선 (경로 피팅의 재료)
  · 초록 선   = /center_path — 차가 가려는 경로
  · 빨간 원   = /target — 조향이 겨냥하는 점
  · 회색 점   = 라이다 원시 스캔
  · 주황 원   = 장애물로 묶인 라이다 클러스터
  · 파란 부채 = lookahead 가중이 실제로 보고 있는 구간

사용: ros2 run track_drive live_viewer
      ros2 run track_drive live_viewer --ros-args -p bev_x_max:=4.0
"""

import math
import os

import cv2
import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy,
                       QoSDurabilityPolicy, qos_profile_sensor_data)
from geometry_msgs.msg import PointStamped, PoseArray
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32, Float32MultiArray

from .lib import cluster as CL

WIN = "xycar - 원본 | BEV"

# /detect/lane 의 z 값 규약 (yolo_node.Z_MID)
Z_MID = 8

C_BG = (24, 24, 24)
C_GRID = (55, 55, 55)
C_AXIS = (90, 90, 90)
C_MID = (60, 235, 255)       # 노란 중앙선 점
C_PATH = (90, 255, 90)       # 갈 경로
C_TARGET = (60, 60, 255)     # 목표점
C_SCAN = (95, 95, 95)        # 라이다 원시점
C_OBS = (60, 150, 255)       # 장애물 클러스터
C_LOOK = (200, 140, 60)      # lookahead 구간


def _font(size):
    """한글 렌더용 폰트. 없으면 None → ASCII 로 떨어진다."""
    for p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"):
        if os.path.exists(p):
            try:
                from PIL import ImageFont
                return ImageFont.truetype(p, size)
            except Exception:                       # noqa: BLE001
                return None
    return None


class _Text:
    """cv2.putText 는 한글을 못 그린다. 폰트가 있으면 PIL 로, 없으면 영문으로."""

    def __init__(self):
        self.f = _font(17)
        self.fs = _font(14)

    def __call__(self, img, s, org, color=(235, 235, 235), small=False):
        f = self.fs if small else self.f
        if f is None:
            cv2.putText(img, s.encode("ascii", "replace").decode(), org,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 if small else 0.5,
                        color, 1, cv2.LINE_AA)
            return img
        from PIL import Image as PImage, ImageDraw
        pil = PImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text(org, s, font=f, fill=(color[2], color[1], color[0]))
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return img


class LiveViewer(Node):
    def __init__(self):
        super().__init__("live_viewer_node")

        # BEV 표시 범위 — 우리 인식이 1.6m 안쪽이라 3.5m 면 충분히 넉넉하다
        self.declare_parameter("bev_x_min", -0.5)
        self.declare_parameter("bev_x_max", 3.5)
        self.declare_parameter("bev_y_half", 2.0)
        self.declare_parameter("panel_h", 520)
        self.declare_parameter("rate_hz", 15.0)
        # lookahead 표시용 (motion_node 와 같은 값을 넣어야 실제와 맞는다)
        self.declare_parameter("steer_focus_base", 1.0)
        self.declare_parameter("steer_focus_k", 0.020)
        self.declare_parameter("steer_focus_min", 0.8)
        self.declare_parameter("steer_focus_max", 2.0)
        self.declare_parameter("steer_focus_sigma", 0.70)
        # 장애물 클러스터링 (lib/cluster.py 와 같은 기본값)
        self.declare_parameter("cluster_gap_m", 0.35)
        self.declare_parameter("cluster_min_pts", 2)
        self.declare_parameter("obs_r_max", 0.6)
        # 조향 표시용 — car.yaml 과 같은 값이어야 화면이 실제와 맞는다
        self.declare_parameter("steer_trim", 0.0)
        self.declare_parameter("steer_limit_left", -62.593314622)
        self.declare_parameter("steer_limit_right", 58.592366078)
        # 발행 조향을 화면 화살표 각도로 바꿀 때 쓰는 최대 시각각(도).
        # 기계값→실제 바퀴각 환산은 실측이 없어서, "한계 대비 몇 %" 를 각도로 보여준다.
        self.declare_parameter("steer_view_max_deg", 32.0)

        g = lambda k: self.get_parameter(k).value          # noqa: E731
        self.x0 = float(g("bev_x_min"))
        self.x1 = float(g("bev_x_max"))
        self.yh = float(g("bev_y_half"))
        self.ph = int(g("panel_h"))
        # BEV 픽셀 크기 — 세로를 패널 높이에 맞추고 가로는 비율 유지
        self.bh = self.ph
        self.res = (self.x1 - self.x0) / float(self.bh)     # m/px
        self.bw = max(120, int(round((2 * self.yh) / self.res)))

        self.txt = _Text()
        self._img = None
        self._lane = []
        self._path = []
        self._target = None
        self._scan = []
        self._obs = []
        self._motor = None
        self._quality = None
        self._t_img = 0.0
        self._t_path = 0.0

        img_qos = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT,
                             durability=QoSDurabilityPolicy.VOLATILE)
        self.create_subscription(Image, "/image_raw", self._on_img, img_qos)
        self.create_subscription(PoseArray, "/detect/lane", self._on_lane, 10)
        self.create_subscription(PoseArray, "/center_path", self._on_path, 10)
        self.create_subscription(PointStamped, "/target", self._on_target, 10)
        self.create_subscription(LaserScan, "/scan", self._on_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, "/xycar_motor",
                                 self._on_motor, 10)
        self.create_subscription(Float32, "/detect/lane_quality",
                                 self._on_quality, 10)

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 1400, self.ph + 104)
        self.create_timer(1.0 / max(1.0, float(g("rate_hz"))), self._draw)
        self.get_logger().info(
            "live_viewer 시작 — BEV x %.1f~%.1fm, y ±%.1fm (창에서 q 로 종료)"
            % (self.x0, self.x1, self.yh))

    # ---------------- 수신 ----------------

    def _on_img(self, m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        self._img = (cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
                     if m.encoding == "rgb8" else a.copy())
        self._t_img = self._now()

    def _on_lane(self, m):
        self._lane = [(p.position.x, p.position.y, int(round(p.position.z)))
                      for p in m.poses]

    def _on_path(self, m):
        self._path = [(p.position.x, p.position.y) for p in m.poses]
        self._t_path = self._now()

    def _on_target(self, m):
        self._target = (m.point.x, m.point.y)

    def _on_scan(self, m):
        pts, ang = [], m.angle_min
        for r in m.ranges:
            if math.isfinite(r) and 0.05 < r < 8.0:
                pts.append((r * math.cos(ang), r * math.sin(ang)))
            ang += m.angle_increment
        self._scan = pts
        # 클러스터는 planner 와 같은 lib 로 뽑는다 (화면과 실제 판단을 일치시킨다)
        try:
            self._obs = CL.cluster_scan(
                m.ranges, m.angle_min, m.angle_increment,
                gap_m=float(self.get_parameter("cluster_gap_m").value),
                min_pts=int(self.get_parameter("cluster_min_pts").value))
        except Exception:                            # noqa: BLE001
            self._obs = []

    def _on_motor(self, m):
        if len(m.data) >= 2:
            self._motor = (float(m.data[0]), float(m.data[1]))

    def _on_quality(self, m):
        self._quality = float(m.data)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------- BEV 좌표 ----------------

    def _px(self, x, y):
        """라이다 좌표(x 전방, y 좌) → BEV 픽셀. 위가 멀고 왼쪽이 +y."""
        col = int(round((self.yh - y) / self.res))
        row = int(round((self.x1 - x) / self.res))
        return col, row

    def _bev(self):
        img = np.full((self.bh, self.bw, 3), C_BG, np.uint8)
        # 1m 격자 + 0.5m 보조
        gx = math.floor(self.x0)
        while gx <= self.x1:
            _, r = self._px(gx, 0)
            if 0 <= r < self.bh:
                cv2.line(img, (0, r), (self.bw - 1, r), C_GRID, 1)
                self.txt(img, "%.0fm" % gx, (4, max(0, r - 16)),
                         (120, 120, 120), small=True)
            gx += 1.0
        gy = -math.floor(self.yh)
        while gy <= self.yh:
            c, _ = self._px(0, gy)
            if 0 <= c < self.bw:
                cv2.line(img, (c, 0), (c, self.bh - 1),
                         C_AXIS if abs(gy) < 1e-6 else C_GRID, 1)
            gy += 1.0
        # 차 위치 (원점)
        c, r = self._px(0.0, 0.0)
        cv2.circle(img, (c, r), 5, (200, 200, 200), -1, cv2.LINE_AA)
        return img

    def _poly(self, img, pts, color, th=2):
        prev = None
        for x, y in pts:
            p = self._px(x, y)
            if prev is not None:
                cv2.line(img, prev, p, color, th, cv2.LINE_AA)
            prev = p

    # ---------------- 그리기 ----------------

    def _focus_x(self):
        spd = abs(self._motor[1]) if self._motor else 0.0
        return float(np.clip(
            self.get_parameter("steer_focus_base").value
            + self.get_parameter("steer_focus_k").value * spd,
            self.get_parameter("steer_focus_min").value,
            self.get_parameter("steer_focus_max").value))

    def _steer_state(self):
        """발행 조향 → (트림기준 rel, 한계대비 비율 -1~+1, 포화여부)."""
        if self._motor is None:
            return None
        trim = float(self.get_parameter("steer_trim").value)
        lo = float(self.get_parameter("steer_limit_left").value)
        hi = float(self.get_parameter("steer_limit_right").value)
        rel = self._motor[0] - trim
        span = abs(lo) if rel < 0 else hi
        ratio = 0.0 if span <= 0 else float(np.clip(rel / span, -1.0, 1.0))
        return rel, ratio, abs(ratio) > 0.97, lo, hi

    def _draw_steer(self, bev):
        """차 원점에서 조향 방향으로 화살표. 길이는 고정, 각도는 한계 대비 비율."""
        st = self._steer_state()
        if st is None:
            return
        rel, ratio, sat, lo, hi = st
        vmax = math.radians(float(self.get_parameter("steer_view_max_deg").value))
        th = ratio * vmax                       # +면 좌(+y), -면 우(-y)
        L = min(1.2, (self.x1 - self.x0) * 0.35)
        c0, r0 = self._px(0.0, 0.0)
        c1, r1 = self._px(L * math.cos(th), L * math.sin(th))
        col = (60, 60, 255) if sat else (255, 200, 90)
        cv2.arrowedLine(bev, (c0, r0), (c1, r1), col, 3, cv2.LINE_AA, tipLength=0.22)
        # 직진 기준선 (트림 위치) — 화살표가 여기서 얼마나 벗어났는지 보라고
        cs, rs = self._px(L, 0.0)
        cv2.line(bev, (c0, r0), (cs, rs), (110, 110, 110), 1, cv2.LINE_AA)
        lab = "좌" if rel < -0.5 else ("우" if rel > 0.5 else "직진")
        self.txt(bev, "%s %+.0f %s" % (lab, rel, "★한계" if sat else ""),
                 (c1 + 8, r1 - 10), col, small=True)

    def _steer_bar(self, bar, x, y, w=300, h=14):
        """좌 한계 ~ 0 ~ 우 한계 를 한 줄로. 지금 값이 어디쯤인지 눈으로."""
        st = self._steer_state()
        cv2.rectangle(bar, (x, y), (x + w, y + h), (60, 60, 60), 1)
        if st is None:
            return
        rel, _, sat, lo, hi = st
        zero = int(x + w * (0.0 - lo) / (hi - lo))
        cv2.line(bar, (zero, y), (zero, y + h), (140, 140, 140), 1)
        px = int(x + w * (float(np.clip(rel, lo, hi)) - lo) / (hi - lo))
        col = (60, 60, 255) if sat else (255, 200, 90)
        cv2.rectangle(bar, (min(zero, px), y + 3), (max(zero, px), y + h - 3), col, -1)
        cv2.line(bar, (px, y - 2), (px, y + h + 2), (255, 255, 255), 2)
        self.txt(bar, "좌%.0f" % lo, (x - 44, y - 3), (140, 140, 140), small=True)
        self.txt(bar, "우%+.0f" % hi, (x + w + 6, y - 3), (140, 140, 140), small=True)

    def _draw(self):
        if self._img is None:
            return
        bev = self._bev()

        # ── lookahead 구간 (조향 가중이 실제로 보는 곳) ──
        fx = self._focus_x()
        sig = float(self.get_parameter("steer_focus_sigma").value)
        for d, alpha in ((sig, 1), (sig * 0.5, 2)):
            for xx in (fx - d, fx + d):
                _, r = self._px(xx, 0)
                if 0 <= r < self.bh:
                    cv2.line(bev, (0, r), (self.bw - 1, r), C_LOOK, alpha)
        _, rf = self._px(fx, 0)
        if 0 <= rf < self.bh:
            cv2.line(bev, (0, rf), (self.bw - 1, rf), C_LOOK, 2)
            self.txt(bev, "lookahead %.2fm" % fx, (self.bw - 150, max(0, rf - 20)),
                     C_LOOK, small=True)

        # ── 라이다 원시점 + 장애물 클러스터 ──
        for x, y in self._scan:
            if self.x0 <= x <= self.x1 and abs(y) <= self.yh:
                c, r = self._px(x, y)
                cv2.circle(bev, (c, r), 1, C_SCAN, -1)
        n_obs = 0
        rmax = float(self.get_parameter("obs_r_max").value)
        for ob in self._obs:
            ox, oy, orr = float(ob["x"]), float(ob["y"]), float(ob["r"])
            # 벽처럼 큰 덩어리는 장애물이 아니라 배경이라 뺀다 (planner 와 같은 기준)
            if orr > rmax or not (self.x0 <= ox <= self.x1) or abs(oy) > self.yh:
                continue
            n_obs += 1
            c, r = self._px(ox, oy)
            cv2.circle(bev, (c, r), max(4, int(orr / self.res)),
                       C_OBS, 2, cv2.LINE_AA)
            self.txt(bev, "%.1fm" % ox, (c + 8, r - 8), C_OBS, small=True)

        # ── 노란 중앙선 점 ──
        mid = [(x, y) for x, y, z in self._lane if z == Z_MID]
        for x, y in mid:
            c, r = self._px(x, y)
            cv2.circle(bev, (c, r), 3, C_MID, -1, cv2.LINE_AA)

        # ── 갈 경로 + 목표점 ──
        fresh = (self._now() - self._t_path) < 0.5
        if self._path and fresh:
            self._poly(bev, self._path, C_PATH, 2)
        if self._target and fresh:
            c, r = self._px(*self._target)
            cv2.circle(bev, (c, r), 7, C_TARGET, -1, cv2.LINE_AA)
            cv2.circle(bev, (c, r), 12, C_TARGET, 1, cv2.LINE_AA)

        # ── 지금 어디로 조향하고 있나 ──
        self._draw_steer(bev)

        self.txt(bev, "BEV", (8, 6))

        # ── 원본 ──
        raw = self._img
        s = self.ph / float(raw.shape[0])
        p1 = cv2.resize(raw, (int(raw.shape[1] * s), self.ph))
        self.txt(p1, "원본 %dx%d" % (raw.shape[1], raw.shape[0]), (8, 6))

        sep = np.full((self.ph, 2, 3), 70, np.uint8)
        top = np.hstack([p1, sep, bev])

        # ── 상태줄 ──
        bar = np.zeros((104, top.shape[1], 3), np.uint8)
        q = "-" if self._quality is None else "%.2f" % self._quality
        self.txt(bar, "노란점 %d개   품질 %s   장애물 %d개"
                 % (len(mid), q, n_obs), (10, 6),
                 C_MID if len(mid) >= 6 else (120, 120, 255))
        if self._path and fresh:
            tx, ty = self._target if self._target else (0.0, 0.0)
            self.txt(bar, "경로 있음 (%d점)   목표 x %.2f  y %+.3f m"
                     % (len(self._path), tx, ty), (10, 30), C_PATH)
        else:
            self.txt(bar, "경로 없음 — 노란선을 못 보거나 점이 부족하다",
                     (10, 30), (110, 110, 255))
        if self._motor:
            a, v = self._motor
            st = self._steer_state()
            rel, ratio, sat = (st[0], st[1], st[2]) if st else (0.0, 0.0, False)
            side = "좌" if rel < -0.5 else ("우" if rel > 0.5 else "직진")
            self.txt(bar, "조향 %s %+.1f (한계의 %.0f%%)%s   발행 %+.1f   속도 %.1f (%.2f m/s)"
                     % (side, rel, abs(ratio) * 100, "  ★한계 포화" if sat else "",
                        a, v, v * 0.08), (10, 54),
                     (60, 60, 255) if sat else (235, 235, 235))
        self._steer_bar(bar, 60, 78)
        self.txt(bar, "lookahead %.2fm ±%.2f   |  q 종료" % (fx, sig),
                 (self.bw + 120 if top.shape[1] > self.bw + 400 else 400, 76),
                 (170, 170, 170), small=True)

        cv2.imshow(WIN, np.vstack([top, bar]))
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            raise KeyboardInterrupt


def main(args=None):
    rclpy.init(args=args)
    node = LiveViewer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
