#!/usr/bin/env python3
"""Read-only live dashboard for the real perception, CNN, and motion pipeline.

The node never publishes a control command.  It shows the latest camera frame,
the exact mode-specific CNN input BEV, path output, trigger diagnostics, YOLO
signal confidences, and the remapped dry-run motion output.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float32MultiArray, String

from .lidar_geometry import GRID_H, GRID_W
from .path_contract import (
    GRID_RESOLUTION_M,
    GRID_X_BOUNDS_M,
    GRID_Y_BOUNDS_M,
)
from .replay_preview_node import _decode_bev, _decode_camera, _stamp_ns


LATEST_SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


class LivePipelineViewerNode(Node):
    def __init__(self) -> None:
        super().__init__("live_pipeline_viewer_node")
        p = self._parameter
        self._image_topic = str(p("image_topic", "/image_raw"))
        self._bev_topic = str(p("bev_topic", "/perception/bev"))
        self._cnn_input_bev_topic = str(
            p("cnn_input_bev_topic", "/debug/cnn_input_bev")
        )
        self._signal_topic = str(p("signal_topic", "/perception/signals"))
        self._cone_trigger_topic = str(
            p("cone_trigger_topic", "/perception/cone_trigger")
        )
        self._cone_mode_topic = str(p("cone_mode_topic", "/cone_mode"))
        self._cnn_mode_topic = str(p("cnn_mode_topic", "/cnn_mode"))
        self._traffic_stop_topic = str(p("traffic_stop_topic", "/traffic_stop"))
        self._decision_center_y_max = float(
            p("signal_decision_center_y_max", 0.22)
        )
        self._decision_min_width = float(p("signal_decision_min_width", 0.08))
        self._motion_topic = str(
            p("motion_topic", "/debug/xycar_motor_dryrun")
        )
        self._compressed_topic = str(
            p("compressed_topic", "/debug/pipeline_view/compressed")
        )
        self._render_hz = float(p("render_hz", 8.0))
        self._show_window = bool(p("show_window", True))
        self._publish_compressed = bool(p("publish_compressed", True))
        self._jpeg_quality = int(p("jpeg_quality", 75))
        self._camera_width = int(p("camera_panel_width", 960))
        self._canvas_height = int(p("canvas_height", 720))
        self._bev_scale = int(p("bev_scale", 4))
        self._speed_visual_max = float(p("speed_visual_max", 10.0))
        self._window_name = str(p("window_name", "Xycar CNN live dry-run"))
        self._screenshot_dir = Path(
            str(p("screenshot_dir", "/home/xytron/pipeline_view_shots_gpt"))
        ).expanduser()

        if self._render_hz <= 0.0 or self._camera_width <= 0:
            raise ValueError("render_hz and camera_panel_width must be positive")
        if self._canvas_height < 540 or self._bev_scale < 1:
            raise ValueError("canvas_height/bev_scale is too small")
        if not 1 <= self._jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1,100]")
        if self._speed_visual_max <= 0.0:
            raise ValueError("speed_visual_max must be positive")
        if not 0.0 <= self._decision_center_y_max <= 1.0:
            raise ValueError("signal_decision_center_y_max must be within [0,1]")
        if not 0.0 <= self._decision_min_width <= 1.0:
            raise ValueError("signal_decision_min_width must be within [0,1]")

        if self._show_window and not os.environ.get("DISPLAY"):
            self.get_logger().warning(
                "DISPLAY is not set; OpenCV window disabled. "
                f"Use {self._compressed_topic} from another viewer."
            )
            self._show_window = False
        if self._show_window:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

        self._lock = threading.Lock()
        self._camera_message: Image | None = None
        self._bev = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
        self._bev_stamp = -1
        self._lidar = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
        self._main = np.empty((0, 2), dtype=np.float32)
        self._shortcut = np.empty((0, 2), dtype=np.float32)
        self._selected = np.empty((0, 2), dtype=np.float32)
        self._signals: dict[str, float] = {}
        self._signal_detections: dict[str, dict[str, float]] = {}
        self._signal_sequence = -1
        self._route_intent = "main"
        self._cone_mode = False
        self._cone_trigger_raw = False
        self._cnn_mode = "GENERAL"
        self._mode_remaining_sec = 0.0
        self._lidar_preprocess = "waiting"
        self._raw_lidar_points = 0
        self._used_lidar_points = 0
        self._road_trigger = False
        self._road_trigger_points = 0
        self._overtake_confirm_count = 0
        self._overtake_confirm_frames = 2
        self._left_streak = 0
        self._traffic_stop = False
        self._traffic_stop_seen = False
        self._cnn_status = "waiting for CNN"
        self._motion_angle = 0.0
        self._motion_speed = 0.0
        self._motion_seen = False
        self._arbitration = "waiting for motion"
        self._last_canvas: np.ndarray | None = None
        self._frames = 0
        self._started = time.monotonic()

        self.create_subscription(
            Image, self._image_topic, self._on_camera, LATEST_SENSOR_QOS
        )
        self.create_subscription(
            Image, self._bev_topic, self._on_bev, LATEST_SENSOR_QOS
        )
        self.create_subscription(
            Image,
            self._cnn_input_bev_topic,
            self._on_cnn_input_bev,
            LATEST_SENSOR_QOS,
        )
        self.create_subscription(PoseArray, "/cnn/path_main", self._on_main, 10)
        self.create_subscription(
            PoseArray, "/cnn/path_shortcut", self._on_shortcut, 10
        )
        self.create_subscription(PoseArray, "/center_path", self._on_selected, 10)
        self.create_subscription(String, "/route_intent", self._on_route, 10)
        self.create_subscription(String, "/debug/cnn_path", self._on_cnn_diag, 10)
        self.create_subscription(String, self._signal_topic, self._on_signals, 10)
        self.create_subscription(Bool, self._cone_mode_topic, self._on_cone_mode, 10)
        self.create_subscription(
            Bool, self._cone_trigger_topic, self._on_cone_trigger, 10
        )
        self.create_subscription(String, self._cnn_mode_topic, self._on_cnn_mode, 10)
        self.create_subscription(Bool, self._traffic_stop_topic, self._on_traffic_stop, 10)
        self.create_subscription(
            Float32MultiArray, self._motion_topic, self._on_motion, 10
        )
        self.create_subscription(
            String, "/debug/arbitration", self._on_arbitration, 10
        )
        self._compressed_pub = self.create_publisher(
            CompressedImage, self._compressed_topic, 1
        )
        self.create_timer(1.0 / self._render_hz, self._render)
        self.get_logger().info(
            "read-only live dashboard ready: no control publishers; "
            f"render_hz={self._render_hz:.1f} compressed={self._compressed_topic}"
        )

    def _parameter(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _on_camera(self, message: Image) -> None:
        # Hold only the latest ROS message. Decode/downscale on the 8 Hz render
        # timer instead of doing 1920x1080 color conversion at camera rate.
        with self._lock:
            self._camera_message = message

    def _on_bev(self, message: Image) -> None:
        try:
            bev = _decode_bev(message)
        except ValueError as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=1.0)
            return
        stamp = _stamp_ns(message.header)
        with self._lock:
            self._bev = bev
            if stamp != self._bev_stamp:
                self._main = np.empty((0, 2), np.float32)
                self._shortcut = np.empty((0, 2), np.float32)
                self._selected = np.empty((0, 2), np.float32)
            self._bev_stamp = stamp

    def _on_cnn_input_bev(self, message: Image) -> None:
        try:
            bev = _decode_bev(message)
            if bev.shape != (GRID_H, GRID_W, 3):
                raise ValueError(f"CNN input BEV shape is invalid: {bev.shape}")
        except ValueError as exc:
            self.get_logger().warning(
                f"viewer dropped CNN input BEV: {exc}", throttle_duration_sec=1.0
            )
            return
        stamp = _stamp_ns(message.header)
        with self._lock:
            if stamp != self._bev_stamp:
                self._main = np.empty((0, 2), np.float32)
                self._shortcut = np.empty((0, 2), np.float32)
                self._selected = np.empty((0, 2), np.float32)
            self._bev = bev
            self._lidar = bev[:, :, 2].copy()
            self._bev_stamp = stamp

    @staticmethod
    def _poses(message: PoseArray) -> np.ndarray:
        return np.asarray(
            [[pose.position.x, pose.position.y] for pose in message.poses],
            dtype=np.float32,
        ).reshape(-1, 2)

    def _store_path(self, message: PoseArray, name: str) -> None:
        with self._lock:
            if _stamp_ns(message.header) != self._bev_stamp:
                return
            setattr(self, name, self._poses(message))

    def _on_main(self, message: PoseArray) -> None:
        self._store_path(message, "_main")

    def _on_shortcut(self, message: PoseArray) -> None:
        self._store_path(message, "_shortcut")

    def _on_selected(self, message: PoseArray) -> None:
        self._store_path(message, "_selected")

    def _on_route(self, message: String) -> None:
        with self._lock:
            self._route_intent = str(message.data).strip().casefold() or "unknown"

    def _on_signals(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            signals = payload.get("signals", {})
            if not isinstance(signals, dict):
                raise ValueError("signals is not an object")
            parsed = {
                str(name).upper(): float(confidence)
                for name, confidence in signals.items()
                if math.isfinite(float(confidence))
            }
            raw_detections = payload.get("detections", {})
            if not isinstance(raw_detections, dict):
                raise ValueError("detections is not an object")
            parsed_detections: dict[str, dict[str, float]] = {}
            for name, item in raw_detections.items():
                if not isinstance(item, dict):
                    continue
                center_y = float(item["center_y_norm"])
                width = float(item["width_norm"])
                if math.isfinite(center_y) and math.isfinite(width):
                    parsed_detections[str(name).upper()] = {
                        "center_y_norm": center_y,
                        "width_norm": width,
                    }
            sequence = int(payload.get("sequence", -1))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warning(
                f"viewer signal parse failed: {exc}", throttle_duration_sec=1.0
            )
            return
        with self._lock:
            self._signals = parsed
            self._signal_detections = parsed_detections
            self._signal_sequence = sequence

    def _on_cone_mode(self, message: Bool) -> None:
        with self._lock:
            self._cone_mode = bool(message.data)

    def _on_cone_trigger(self, message: Bool) -> None:
        with self._lock:
            self._cone_trigger_raw = bool(message.data)

    def _on_cnn_mode(self, message: String) -> None:
        with self._lock:
            self._cnn_mode = str(message.data).strip().upper() or "UNKNOWN"

    def _on_traffic_stop(self, message: Bool) -> None:
        with self._lock:
            self._traffic_stop = bool(message.data)
            self._traffic_stop_seen = True

    def _on_cnn_diag(self, message: String) -> None:
        payload: dict = {}
        road: dict = {}
        try:
            payload = json.loads(message.data)
            route = payload.get("route_intent")
            road = payload.get("road_lidar", {})
            if not isinstance(road, dict):
                road = {}
            if payload.get("ok"):
                status = (
                    f"CNN {payload.get('cnn_total_ms', '?')}ms "
                    f"cam={payload.get('camera_cells', '?')} "
                    f"lidar={payload.get('lidar_cells', '?')} "
                    f"passes={payload.get('cnn_passes', '?')}"
                )
            else:
                status = f"CNN BLOCKED: {payload.get('reason', 'unknown')}"
        except (TypeError, ValueError, json.JSONDecodeError):
            status = "CNN diagnostic parse error"
            route = None
        with self._lock:
            self._cnn_status = status
            mode = payload.get("cnn_mode") if isinstance(payload, dict) else None
            if isinstance(mode, str) and mode.strip():
                self._cnn_mode = mode.strip().upper()
            self._mode_remaining_sec = float(payload.get("mode_remaining_sec", 0.0))
            self._lidar_preprocess = str(
                payload.get("lidar_preprocess_mode", self._lidar_preprocess)
            )
            self._raw_lidar_points = int(payload.get("raw_lidar_points", 0))
            self._used_lidar_points = int(
                payload.get(
                    "used_lidar_points",
                    payload.get("filtered_lidar_points", 0),
                )
            )
            self._road_trigger = bool(road.get("trigger_detected", False))
            self._road_trigger_points = int(road.get("trigger_inside_points", 0))
            self._overtake_confirm_count = int(
                payload.get("overtake_confirm_count", 0)
            )
            self._overtake_confirm_frames = int(
                payload.get("overtake_confirm_frames", 2)
            )
            self._left_streak = int(payload.get("left_streak", 0))
            if isinstance(route, str) and route.strip():
                self._route_intent = route.strip().casefold()

    def _on_motion(self, message: Float32MultiArray) -> None:
        if len(message.data) < 2:
            return
        angle, speed = float(message.data[0]), float(message.data[1])
        if not (math.isfinite(angle) and math.isfinite(speed)):
            return
        with self._lock:
            self._motion_angle = angle
            self._motion_speed = speed
            self._motion_seen = True

    def _on_arbitration(self, message: String) -> None:
        text = str(message.data).strip() or "motion debug empty"
        try:
            payload = json.loads(text)
            text = (
                f"MOTION profile={payload.get('profile', '?')} "
                f"lookahead={float(payload.get('lookahead_m', 0.0)):.2f}m "
                f"gain={float(payload.get('steer_gain', 0.0)):.2f} "
                f"alpha={float(payload.get('steer_smooth_alpha', 0.0)):.2f} "
                f"speed={float(payload.get('requested_speed', 0.0)):.1f} "
                f"drive={payload.get('drive', False)} reason={payload.get('reason', '?')}"
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        with self._lock:
            self._arbitration = text

    @staticmethod
    def _path_pixels(points: np.ndarray, height: int, width: int) -> np.ndarray:
        pixels = []
        x_max = GRID_X_BOUNDS_M[1]
        y_max = GRID_Y_BOUNDS_M[1]
        for x, y in points:
            row = int(np.floor((x_max - float(x)) / GRID_RESOLUTION_M))
            col = int(np.floor((y_max - float(y)) / GRID_RESOLUTION_M))
            if 0 <= row < height and 0 <= col < width:
                pixels.append((col, row))
        return np.asarray(pixels, dtype=np.int32).reshape(-1, 2)

    def _bev_panel(
        self,
        bev: np.ndarray,
        lidar: np.ndarray,
        main: np.ndarray,
        shortcut: np.ndarray,
        selected: np.ndarray,
    ) -> np.ndarray:
        height, width = bev.shape[:2]
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[bev[:, :, 0] > 0] = (0, 220, 255)       # yellow mid
        image[bev[:, :, 1] > 0] = (245, 245, 245)    # white lane
        image[lidar > 0] = (30, 30, 255)              # raw LiDAR

        for x in np.arange(0.0, GRID_X_BOUNDS_M[1] + 0.01, 0.5):
            row = int(np.floor((GRID_X_BOUNDS_M[1] - x) / GRID_RESOLUTION_M))
            if 0 <= row < height:
                cv2.line(image, (0, row), (width - 1, row), (55, 55, 55), 1)
        for y in np.arange(GRID_Y_BOUNDS_M[0], GRID_Y_BOUNDS_M[1] + 0.01, 0.5):
            col = int(np.floor((GRID_Y_BOUNDS_M[1] - y) / GRID_RESOLUTION_M))
            if 0 <= col < width:
                cv2.line(image, (col, 0), (col, height - 1), (55, 55, 55), 1)

        for points, color, thickness in (
            (main, (40, 255, 40), 2),
            (shortcut, (255, 80, 255), 2),
            (selected, (255, 255, 0), 3),
        ):
            pixels = self._path_pixels(points, height, width)
            if len(pixels) >= 2:
                cv2.polylines(image, [pixels], False, color, thickness, cv2.LINE_AA)
            elif len(pixels) == 1:
                cv2.circle(image, tuple(pixels[0]), 2, color, -1)

        origin_row = int(
            np.floor((GRID_X_BOUNDS_M[1] - 0.0) / GRID_RESOLUTION_M)
        )
        origin_col = int(
            np.floor((GRID_Y_BOUNDS_M[1] - 0.0) / GRID_RESOLUTION_M)
        )
        cv2.circle(image, (origin_col, origin_row), 3, (255, 180, 0), -1)
        return cv2.resize(
            image,
            (width * self._bev_scale, height * self._bev_scale),
            interpolation=cv2.INTER_NEAREST,
        )

    def _draw_motion_arrow(
        self, image: np.ndarray, angle: float, speed: float, seen: bool
    ) -> None:
        height, width = image.shape[:2]
        base = (width // 2, height - 48)
        if not seen:
            cv2.putText(
                image, "MOTION: waiting", (base[0] - 110, base[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2, cv2.LINE_AA,
            )
            return
        limit = 62.593314622 if angle < 0.0 else 58.592366078
        normalized = float(np.clip(angle / max(limit, 1e-6), -1.0, 1.0))
        display_angle = math.radians(normalized * 50.0)
        speed_ratio = float(np.clip(abs(speed) / self._speed_visual_max, 0.0, 1.0))
        length = int(round(45.0 + 125.0 * speed_ratio))
        tip = (
            int(round(base[0] + math.sin(display_angle) * length)),
            int(round(base[1] - math.cos(display_angle) * length)),
        )
        color = (40, 255, 40) if abs(speed) > 0.01 else (30, 30, 255)
        cv2.arrowedLine(image, base, tip, color, 8, cv2.LINE_AA, tipLength=0.20)
        direction = "LEFT" if angle < -0.5 else "RIGHT" if angle > 0.5 else "STRAIGHT"
        cv2.putText(
            image,
            f"MOTION {direction} steer={angle:+.2f} speed={speed:.2f}",
            (max(10, base[0] - 245), base[1] + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _fit_camera(image: np.ndarray, width: int, height: int) -> np.ndarray:
        scale = min(width / image.shape[1], height / image.shape[0])
        new_size = (
            max(1, int(round(image.shape[1] * scale))),
            max(1, int(round(image.shape[0] * scale))),
        )
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        panel = np.full((height, width, 3), 18, dtype=np.uint8)
        top = (height - resized.shape[0]) // 2
        left = (width - resized.shape[1]) // 2
        panel[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
        return panel

    def _render(self) -> None:
        with self._lock:
            camera_message = self._camera_message
            bev = self._bev.copy()
            lidar = self._lidar.copy()
            main = self._main.copy()
            shortcut = self._shortcut.copy()
            selected = self._selected.copy()
            signals = dict(self._signals)
            signal_detections = {
                name: dict(item) for name, item in self._signal_detections.items()
            }
            signal_sequence = self._signal_sequence
            route = self._route_intent
            cone_mode = self._cone_mode
            cone_trigger_raw = self._cone_trigger_raw
            cnn_mode = self._cnn_mode
            mode_remaining_sec = self._mode_remaining_sec
            lidar_preprocess = self._lidar_preprocess
            raw_lidar_points = self._raw_lidar_points
            used_lidar_points = self._used_lidar_points
            road_trigger = self._road_trigger
            road_trigger_points = self._road_trigger_points
            overtake_confirm_count = self._overtake_confirm_count
            overtake_confirm_frames = self._overtake_confirm_frames
            left_streak = self._left_streak
            traffic_stop = self._traffic_stop
            traffic_stop_seen = self._traffic_stop_seen
            cnn_status = self._cnn_status
            angle = self._motion_angle
            speed = self._motion_speed
            motion_seen = self._motion_seen
            arbitration = self._arbitration
        if camera_message is None:
            return
        try:
            camera = _decode_camera(camera_message)
        except ValueError as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=1.0)
            return

        camera_height = min(540, self._canvas_height - 120)
        camera_panel = self._fit_camera(camera, self._camera_width, camera_height)
        self._draw_motion_arrow(camera_panel, angle, speed, motion_seen)
        bev_panel = self._bev_panel(bev, lidar, main, shortcut, selected)
        right_width = max(bev_panel.shape[1], 480)
        canvas = np.full(
            (self._canvas_height, self._camera_width + right_width, 3),
            22,
            dtype=np.uint8,
        )
        canvas[96 : 96 + camera_panel.shape[0], : self._camera_width] = camera_panel
        bev_top = max(96, (self._canvas_height - bev_panel.shape[0]) // 2)
        bev_left = self._camera_width + (right_width - bev_panel.shape[1]) // 2
        canvas[
            bev_top : bev_top + bev_panel.shape[0],
            bev_left : bev_left + bev_panel.shape[1],
        ] = bev_panel

        dominant = max(
            ((float(confidence), str(name)) for name, confidence in signals.items()),
            default=None,
        )
        dominant_name = dominant[1] if dominant is not None else "NONE"
        dominant_conf = dominant[0] if dominant is not None else 0.0
        detection = signal_detections.get(dominant_name, {})
        center_y = detection.get("center_y_norm")
        box_width = detection.get("width_norm")
        decision_zone = bool(
            center_y is not None
            and box_width is not None
            and center_y <= self._decision_center_y_max
            and box_width >= self._decision_min_width
        )
        if traffic_stop_seen:
            traffic_command = "STOP" if traffic_stop else "GO"
            traffic_color = (30, 30, 245) if traffic_stop else (30, 225, 30)
        else:
            traffic_command = "WAIT"
            traffic_color = (0, 210, 255)
        geometry_text = (
            f"cy={center_y:.3f}<={self._decision_center_y_max:.2f} "
            f"w={box_width:.3f}>={self._decision_min_width:.2f}"
            if center_y is not None and box_width is not None
            else "box=NONE"
        )
        cv2.putText(
            canvas,
            (
                f"TRAFFIC={traffic_command} | RED/YELLOW=STOP, GREEN/LEFT=GO | "
                f"YOLO={dominant_name} {dominant_conf:.2f} "
                f"zone={'IN' if decision_zone else 'OUT'} | {geometry_text}"
            )[:150],
            (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            traffic_color, 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                f"MODE={cnn_mode} remain={mode_remaining_sec:.1f}s "
                f"input={lidar_preprocess} lidar={raw_lidar_points}->{used_lidar_points} "
                f"route={route.upper()} selected={len(selected)}pts | {cnn_status}"
            )[:165],
            (14, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (255, 255, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                f"TRIGGER LEFT={left_streak}/2 START_R={int(cone_trigger_raw)} "
                f"IN_ROAD={int(road_trigger)} pts={road_trigger_points} "
                f"OVERTAKE={overtake_confirm_count}/{overtake_confirm_frames} "
                f"CONE_MODE={int(cone_mode)} | BEV red=actual CNN LiDAR input"
            )[:180],
            (14, 77), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (225, 225, 225), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            arbitration[:150],
            (14, self._canvas_height - 18), cv2.FONT_HERSHEY_SIMPLEX,
            0.43, (180, 220, 180), 1, cv2.LINE_AA,
        )

        if self._publish_compressed:
            ok, encoded = cv2.imencode(
                ".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            )
            if ok:
                message = CompressedImage()
                message.header.stamp = self.get_clock().now().to_msg()
                message.format = "jpeg"
                message.data = encoded.tobytes()
                self._compressed_pub.publish(message)

        if self._show_window:
            cv2.imshow(self._window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyWindow(self._window_name)
                self._show_window = False
            elif key == ord("s"):
                self._screenshot_dir.mkdir(parents=True, exist_ok=True)
                destination = self._screenshot_dir / time.strftime(
                    "pipeline_%Y%m%d_%H%M%S.jpg"
                )
                cv2.imwrite(str(destination), canvas)
                self.get_logger().info(f"saved viewer screenshot: {destination}")

        with self._lock:
            self._last_canvas = canvas
        self._frames += 1

    def destroy_node(self) -> bool:
        if self._show_window:
            cv2.destroyWindow(self._window_name)
        elapsed = max(1e-9, time.monotonic() - self._started)
        self.get_logger().info(
            f"live dashboard stopped frames={self._frames} "
            f"effective_fps={self._frames / elapsed:.2f}"
        )
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = LivePipelineViewerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
