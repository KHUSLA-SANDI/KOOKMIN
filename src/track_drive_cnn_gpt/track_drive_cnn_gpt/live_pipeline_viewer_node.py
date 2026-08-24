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


# OpenCV uses BGR.  Keep the path, mode label, timer, and trigger diagnostics
# on the same stable four-mode palette so the active controller is obvious at
# a glance.
MODE_COLORS_BGR = {
    "GENERAL": (40, 255, 40),
    "CONE": (255, 255, 0),
    "OVERTAKE": (30, 30, 255),
    "SHORTCUT": (255, 80, 255),
}
SIGNAL_COLORS_BGR = {
    "RED": (30, 30, 255),
    "YELLOW": (0, 230, 255),
    "GREEN": (40, 255, 40),
    "LEFT": (40, 255, 40),
}
INACTIVE_TEXT_BGR = (145, 145, 145)
PRIMARY_TEXT_BGR = (232, 232, 232)


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
        self._yolo_state_topic = str(
            p("yolo_state_topic", "/debug/yolo_state")
        )
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
        self._signal_display_floor = float(p("signal_display_floor", 0.25))
        self._cone_enter_bottom_y_min = float(
            p("cone_enter_bottom_y_min", 0.55)
        )
        self._cone_approach_bottom_y_min = float(
            p("cone_approach_bottom_y_min", 0.70)
        )
        self._overtake_min_raw_points = int(
            p("overtake_min_raw_points", 3)
        )
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
        if not 0.0 <= self._signal_display_floor <= 1.0:
            raise ValueError("signal_display_floor must be within [0,1]")
        if not 0.0 <= self._cone_enter_bottom_y_min <= 1.0:
            raise ValueError("cone_enter_bottom_y_min must be within [0,1]")
        if not 0.0 <= self._cone_approach_bottom_y_min <= 1.0:
            raise ValueError("cone_approach_bottom_y_min must be within [0,1]")
        if self._cone_approach_bottom_y_min >= self._cone_enter_bottom_y_min:
            raise ValueError(
                "cone_approach_bottom_y_min must be lower than cone entry"
            )
        if self._overtake_min_raw_points < 1:
            raise ValueError("overtake_min_raw_points must be positive")

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
        self._path_stamps = {
            "_main": -1,
            "_shortcut": -1,
            "_selected": -1,
        }
        self._signals: dict[str, float] = {}
        self._signal_detections: dict[str, dict[str, float]] = {}
        self._signal_sequence = -1
        self._traffic_detection: dict[str, float] = {}
        self._traffic_classifier_ran = False
        self._cone_detection: dict[str, float] = {}
        self._cone_detection_seen_mono = -math.inf
        self._cone_approach_ready = False
        self._cone_approach_confirm_count = 0
        self._cone_approach_confirm_frames = 1
        self._cone_approach_latch_active = False
        self._cone_enter_ready = False
        self._cone_confirm_count = 0
        self._cone_confirm_frames = 2
        self._cone_state_latch_active = False
        self._route_intent = "main"
        self._cone_mode = False
        self._cone_trigger_raw = False
        self._cnn_mode = "GENERAL"
        self._mode_remaining_sec = 0.0
        self._mode_duration_sec = 0.0
        self._lidar_preprocess = "waiting"
        self._raw_lidar_points = 0
        self._used_lidar_points = 0
        self._road_trigger = False
        self._road_trigger_points = 0
        self._road_trigger_cluster_points = 0
        self._road_trigger_inside_ratio = 0.0
        self._road_trigger_min_ratio = 0.50
        self._road_trigger_x_min_m = 0.0
        self._road_trigger_x_max_m = 1.40
        self._road_trigger_inward_margin_m = 0.10
        self._road_reference = "unavailable"
        self._overtake_strategy = "cnn"
        self._obstacle_lane = "UNKNOWN"
        self._obstacle_lateral_offset_m: float | None = None
        self._obstacle_side_ambiguous = False
        self._hardcode_window_active = False
        self._post_cone_hardcode_remaining_sec = 0.0
        self._overtake_confirm_count = 0
        self._overtake_confirm_frames = 2
        self._overtake_trigger_armed = True
        self._overtake_suppressed_by_start_r = False
        self._left_streak = 0
        self._trigger_highlight_until = {
            "LEFT": -math.inf,
            "START_R": -math.inf,
            "OVERTAKE": -math.inf,
        }
        self._traffic_stop = False
        self._traffic_stop_seen = False
        self._race_state = "WAIT_GREEN"
        self._race_lap_count = 0
        self._race_target_laps = 3
        self._race_finish_enabled = True
        self._race_go = False
        self._race_encounter_armed = True
        self._race_cooldown_remaining_sec = 0.0
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
        self.create_subscription(
            String, self._yolo_state_topic, self._on_yolo_state, 10
        )
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
        stamp = _stamp_ns(message.header)
        points = self._poses(message)
        with self._lock:
            # BEV and path callbacks are delivered independently.  Requiring
            # their stamps to equal the *currently displayed* BEV made a valid
            # path disappear whenever the next BEV callback won the race.
            # Keep the newest path instead, matching simple_motion's cached
            # path behaviour during a perception gap.
            if stamp < self._path_stamps[name]:
                return
            self._path_stamps[name] = stamp
            if name == "_selected" and points.size == 0:
                return
            setattr(self, name, points)

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
                keys = (
                    "confidence",
                    "center_x_norm",
                    "center_y_norm",
                    "width_norm",
                    "height_norm",
                    "y2_norm",
                )
                parsed_item = {
                    key: float(item[key]) for key in keys if key in item
                }
                if (
                    "center_y_norm" in parsed_item
                    and "width_norm" in parsed_item
                    and all(math.isfinite(value) for value in parsed_item.values())
                ):
                    parsed_detections[str(name).upper()] = parsed_item
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

    def _on_yolo_state(self, message: String) -> None:
        """Cache the subscriber-gated, per-frame START_R viewer contract."""

        try:
            payload = json.loads(message.data)
            if payload.get("schema_version") != "yolo_state_v1_gpt":
                raise ValueError("unsupported YOLO state schema")
            traffic = payload.get("traffic_light", {})
            if not isinstance(traffic, dict):
                raise ValueError("traffic_light is not an object")
            traffic_detected = bool(traffic.get("detected", False))
            traffic_classifier_ran = bool(traffic.get("classifier_ran", False))
            parsed_traffic: dict[str, float] = {}
            for key in (
                "confidence",
                "center_x_norm",
                "center_y_norm",
                "width_norm",
                "height_norm",
            ):
                raw_value = traffic.get(key)
                if raw_value is None:
                    continue
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ValueError(f"non-finite traffic_light {key}")
                parsed_traffic[key] = value
            start_r = payload.get("start_r")
            if not isinstance(start_r, dict):
                raise ValueError("start_r is not an object")
            detected = bool(start_r.get("detected", False))
            parsed_detection: dict[str, float] = {}
            for key in (
                "confidence",
                "y2_norm",
                "y2_px",
                "approach_threshold_norm",
                "approach_threshold_px",
                "threshold_norm",
                "threshold_px",
            ):
                raw_value = start_r.get(key)
                if raw_value is None:
                    continue
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ValueError(f"non-finite START_R {key}")
                parsed_detection[key] = value
            if detected and "y2_norm" not in parsed_detection:
                raise ValueError("detected START_R is missing y2_norm")
            approach_ready = bool(start_r.get("approach_ready", False))
            approach_confirm_count = max(
                0, int(start_r.get("approach_confirm_count", 0))
            )
            approach_confirm_frames = max(
                1, int(start_r.get("approach_confirm_frames", 1))
            )
            approach_latch_active = bool(
                start_r.get("approach_latch_active", False)
            )
            ready = bool(start_r.get("enter_ready", False))
            confirm_count = max(0, int(start_r.get("confirm_count", 0)))
            confirm_frames = max(1, int(start_r.get("confirm_frames", 2)))
            latch_active = bool(start_r.get("latch_active", False))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warning(
                f"viewer YOLO state parse failed: {exc}",
                throttle_duration_sec=1.0,
            )
            return

        now = time.monotonic()
        with self._lock:
            self._traffic_detection = parsed_traffic if traffic_detected else {}
            self._traffic_classifier_ran = traffic_classifier_ran
            previous_y2 = self._cone_detection.get("y2_norm")
            next_y2 = parsed_detection.get("y2_norm")
            if (
                ready and not self._cone_enter_ready
                or confirm_count > self._cone_confirm_count
                or next_y2 is not None
                and (previous_y2 is None or next_y2 > previous_y2 + 0.005)
            ):
                self._trigger_highlight_until["START_R"] = now + 1.2
            self._cone_detection = parsed_detection if detected else {}
            self._cone_detection_seen_mono = now
            self._cone_approach_ready = approach_ready
            self._cone_approach_confirm_count = approach_confirm_count
            self._cone_approach_confirm_frames = approach_confirm_frames
            self._cone_approach_latch_active = approach_latch_active
            self._cone_enter_ready = ready
            self._cone_confirm_count = confirm_count
            self._cone_confirm_frames = confirm_frames
            self._cone_state_latch_active = latch_active

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
        now = time.monotonic()
        with self._lock:
            self._cnn_status = status
            mode = payload.get("cnn_mode") if isinstance(payload, dict) else None
            if isinstance(mode, str) and mode.strip():
                self._cnn_mode = mode.strip().upper()
            self._mode_remaining_sec = float(payload.get("mode_remaining_sec", 0.0))
            self._mode_duration_sec = float(payload.get("mode_duration_sec", 0.0))
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
            next_road_trigger = bool(road.get("trigger_detected", False))
            next_road_points = int(road.get("trigger_inside_points", 0))
            next_cluster_points = int(road.get("trigger_cluster_points", 0))
            next_inside_ratio = float(road.get("trigger_inside_ratio", 0.0))
            next_overtake_count = int(payload.get("overtake_confirm_count", 0))
            next_left_streak = int(payload.get("left_streak", 0))
            if next_left_streak > self._left_streak:
                self._trigger_highlight_until["LEFT"] = now + 1.2
            if (
                next_road_trigger and not self._road_trigger
                or next_road_points > self._road_trigger_points
                or next_overtake_count > self._overtake_confirm_count
            ):
                self._trigger_highlight_until["OVERTAKE"] = now + 1.2
            self._road_trigger = next_road_trigger
            self._road_trigger_points = next_road_points
            self._road_trigger_cluster_points = next_cluster_points
            self._road_trigger_inside_ratio = next_inside_ratio
            self._road_trigger_min_ratio = float(
                road.get("trigger_min_inside_ratio", self._road_trigger_min_ratio)
            )
            self._road_trigger_x_min_m = float(
                road.get("trigger_x_min_m", self._road_trigger_x_min_m)
            )
            self._road_trigger_x_max_m = float(
                road.get("trigger_x_max_m", self._road_trigger_x_max_m)
            )
            self._road_trigger_inward_margin_m = float(
                road.get(
                    "trigger_inward_margin_m",
                    self._road_trigger_inward_margin_m,
                )
            )
            self._overtake_min_raw_points = int(
                road.get("trigger_min_points", self._overtake_min_raw_points)
            )
            self._road_reference = str(road.get("reference", "unavailable"))
            self._overtake_strategy = str(
                payload.get("overtake_strategy", self._overtake_strategy)
            )
            self._obstacle_lane = str(
                road.get("obstacle_lane", self._obstacle_lane)
            ).strip().upper()
            lateral_offset = road.get("trigger_lateral_offset_m")
            self._obstacle_lateral_offset_m = (
                float(lateral_offset) if lateral_offset is not None else None
            )
            self._obstacle_side_ambiguous = bool(
                road.get("trigger_side_ambiguous", False)
            )
            self._hardcode_window_active = bool(
                payload.get("hardcode_window_active", False)
            )
            self._post_cone_hardcode_remaining_sec = float(
                payload.get("post_cone_hardcode_remaining_sec", 0.0)
            )
            self._overtake_confirm_count = next_overtake_count
            self._overtake_confirm_frames = int(
                payload.get("overtake_confirm_frames", 2)
            )
            self._overtake_trigger_armed = bool(
                payload.get("overtake_trigger_armed", True)
            )
            self._overtake_suppressed_by_start_r = bool(
                road.get("suppressed_by_start_r", False)
            )
            self._left_streak = next_left_streak
            self._race_state = str(
                payload.get("race_state", self._race_state)
            ).strip().upper()
            self._race_lap_count = int(
                payload.get("race_lap_count", self._race_lap_count)
            )
            self._race_target_laps = int(
                payload.get("race_target_laps", self._race_target_laps)
            )
            self._race_finish_enabled = bool(
                payload.get("race_finish_enabled", self._race_finish_enabled)
            )
            self._race_go = bool(payload.get("race_go", self._race_go))
            self._race_encounter_armed = bool(
                payload.get(
                    "race_encounter_armed", self._race_encounter_armed
                )
            )
            self._race_cooldown_remaining_sec = float(
                payload.get(
                    "race_cooldown_remaining_sec",
                    self._race_cooldown_remaining_sec,
                )
            )
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
            block_active = bool(payload.get("hardcoded_block_active", False))
            block_prefix = (
                f"BLOCK {payload.get('overtake_strategy', '?')} "
                f"active={int(block_active)} "
                f"{payload.get('hardcoded_block_direction', 'NONE')} "
                f"{payload.get('hardcoded_block_phase', 'IDLE')} "
                f"{int(payload.get('hardcoded_block_phase_tick', 0))}/"
                f"{int(payload.get('hardcoded_block_phase_ticks', 0))} "
                f"total={int(payload.get('hardcoded_block_total_tick', 0))}/"
                f"{int(payload.get('hardcoded_block_total_ticks', 0))} | "
            )
            text = block_prefix + (
                f"MOTION {payload.get('profile', '?')} "
                f"L={float(payload.get('lookahead_m', 0.0)):.2f} "
                f"G={float(payload.get('steer_gain', 0.0)):.2f} "
                f"V={float(payload.get('requested_speed', 0.0)):.1f} "
                f"drive={int(bool(payload.get('drive', False)))} "
                f"{payload.get('reason', '?')}"
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

    @staticmethod
    def _mode_color(mode: str) -> tuple[int, int, int]:
        return MODE_COLORS_BGR.get(str(mode).strip().upper(), PRIMARY_TEXT_BGR)

    @staticmethod
    def _put_segment(
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
        *,
        scale: float = 0.50,
        thickness: int = 1,
        gap: int = 14,
    ) -> int:
        """Draw one colored status segment and return the next x position."""

        label = str(text)
        cv2.putText(
            image,
            label,
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(scale),
            color,
            int(thickness),
            cv2.LINE_AA,
        )
        width = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            float(scale),
            int(thickness),
        )[0][0]
        return int(x) + int(width) + int(gap)

    def _bev_panel(
        self,
        bev: np.ndarray,
        lidar: np.ndarray,
        main: np.ndarray,
        shortcut: np.ndarray,
        selected: np.ndarray,
        cnn_mode: str,
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

        # Diagnostic path topics are neutral underlays.  The actual path sent to
        # motion (/center_path) is drawn last in the active mode's color.
        for points, color, thickness in (
            (main, (95, 95, 95), 1),
            (shortcut, (95, 95, 95), 1),
            (selected, self._mode_color(cnn_mode), 4),
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
            traffic_detection = dict(self._traffic_detection)
            traffic_classifier_ran = self._traffic_classifier_ran
            cone_detection = dict(self._cone_detection)
            cone_detection_seen_mono = self._cone_detection_seen_mono
            cone_approach_ready = self._cone_approach_ready
            cone_approach_confirm_count = self._cone_approach_confirm_count
            cone_approach_confirm_frames = self._cone_approach_confirm_frames
            cone_approach_latch_active = self._cone_approach_latch_active
            cone_enter_ready = self._cone_enter_ready
            cone_confirm_count = self._cone_confirm_count
            cone_confirm_frames = self._cone_confirm_frames
            cone_state_latch_active = self._cone_state_latch_active
            route = self._route_intent
            cone_mode = self._cone_mode
            cone_trigger_raw = self._cone_trigger_raw
            cnn_mode = self._cnn_mode
            mode_remaining_sec = self._mode_remaining_sec
            mode_duration_sec = self._mode_duration_sec
            lidar_preprocess = self._lidar_preprocess
            raw_lidar_points = self._raw_lidar_points
            used_lidar_points = self._used_lidar_points
            road_trigger = self._road_trigger
            road_trigger_points = self._road_trigger_points
            road_trigger_cluster_points = self._road_trigger_cluster_points
            road_trigger_inside_ratio = self._road_trigger_inside_ratio
            road_trigger_min_ratio = self._road_trigger_min_ratio
            road_trigger_x_min_m = self._road_trigger_x_min_m
            road_trigger_x_max_m = self._road_trigger_x_max_m
            road_trigger_inward_margin_m = self._road_trigger_inward_margin_m
            road_reference = self._road_reference
            overtake_strategy = self._overtake_strategy
            obstacle_lane = self._obstacle_lane
            obstacle_lateral_offset_m = self._obstacle_lateral_offset_m
            obstacle_side_ambiguous = self._obstacle_side_ambiguous
            hardcode_window_active = self._hardcode_window_active
            post_cone_hardcode_remaining_sec = (
                self._post_cone_hardcode_remaining_sec
            )
            overtake_min_raw_points = self._overtake_min_raw_points
            overtake_confirm_count = self._overtake_confirm_count
            overtake_confirm_frames = self._overtake_confirm_frames
            overtake_trigger_armed = self._overtake_trigger_armed
            overtake_suppressed_by_start_r = self._overtake_suppressed_by_start_r
            left_streak = self._left_streak
            trigger_highlight_until = dict(self._trigger_highlight_until)
            traffic_stop = self._traffic_stop
            traffic_stop_seen = self._traffic_stop_seen
            race_state = self._race_state
            race_lap_count = self._race_lap_count
            race_target_laps = self._race_target_laps
            race_finish_enabled = self._race_finish_enabled
            race_go = self._race_go
            race_encounter_armed = self._race_encounter_armed
            race_cooldown_remaining_sec = self._race_cooldown_remaining_sec
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
        bev_panel = self._bev_panel(
            bev, lidar, main, shortcut, selected, cnn_mode
        )
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
        center_x = detection.get("center_x_norm")
        center_y = detection.get("center_y_norm")
        box_width = detection.get("width_norm")
        decision_zone = bool(
            center_y is not None
            and box_width is not None
            and center_y <= self._decision_center_y_max
            and box_width >= self._decision_min_width
        )
        if race_state == "FINISHED":
            traffic_command = "FINISH STOP"
            traffic_color = (30, 30, 245)
        elif not race_go:
            traffic_command = "WAIT GREEN"
            traffic_color = (0, 210, 255)
        elif traffic_stop_seen:
            traffic_command = "STOP" if traffic_stop else "GO"
            traffic_color = (30, 30, 245) if traffic_stop else (30, 225, 30)
        else:
            traffic_command = "WAIT"
            traffic_color = INACTIVE_TEXT_BGR

        # Row 1: show every detected class immediately, but label the dominant
        # class as an actual decision only inside the stop-line image zone.
        signal_color = SIGNAL_COLORS_BGR.get(dominant_name, INACTIVE_TEXT_BGR)
        x = self._put_segment(
            canvas, "SIGNAL CONF", 14, 26, INACTIVE_TEXT_BGR,
            scale=0.54, thickness=1,
        )
        traffic_detector_conf = traffic_detection.get("confidence")
        x = self._put_segment(
            canvas,
            (
                f"TL_DET={traffic_detector_conf:.2f} CLS=RUN"
                if traffic_detector_conf is not None and traffic_classifier_ran
                else (
                    f"TL_DET={traffic_detector_conf:.2f} CLS=SKIP"
                    if traffic_detector_conf is not None
                    else "TL_DET=-- CLS=SKIP"
                )
            ),
            x,
            26,
            (255, 200, 30)
            if traffic_detector_conf is not None
            else INACTIVE_TEXT_BGR,
            scale=0.48,
            thickness=2 if traffic_classifier_ran else 1,
            gap=12,
        )
        for name in ("GREEN", "LEFT", "RED", "YELLOW"):
            confidence = signals.get(name)
            signal_label = "<- LEFT" if name == "LEFT" else name
            text = (
                f"{signal_label}={confidence:.2f}"
                if confidence is not None
                else f"{signal_label}<{self._signal_display_floor:.2f}"
            )
            x = self._put_segment(
                canvas,
                text,
                x,
                26,
                SIGNAL_COLORS_BGR[name]
                if confidence is not None
                else INACTIVE_TEXT_BGR,
                scale=0.50,
                thickness=2 if confidence is not None else 1,
                gap=12,
            )
        if dominant is None:
            decision_text = "DECISION=NONE"
            decision_color = INACTIVE_TEXT_BGR
        elif decision_zone:
            decision_text = f"DECISION={dominant_name} {dominant_conf:.2f}"
            decision_color = signal_color
        else:
            decision_text = f"DECISION=WAIT ({dominant_name} {dominant_conf:.2f})"
            decision_color = INACTIVE_TEXT_BGR
        x = self._put_segment(
            canvas, decision_text, x, 26, decision_color,
            scale=0.50, thickness=2, gap=14,
        )
        x = self._put_segment(
            canvas,
            f"CMD={traffic_command}",
            x,
            26,
            traffic_color,
            scale=0.62,
            thickness=2,
            gap=22,
        )
        signal_geometry = (
            f"SIGNAL_XY=({center_x:.3f},{center_y:.3f}) WIDTH={box_width:.3f}"
            if center_x is not None and center_y is not None and box_width is not None
            else "SIGNAL_XY=(--,--) WIDTH=--"
        )
        x = self._put_segment(
            canvas,
            signal_geometry,
            x,
            26,
            signal_color if decision_zone else INACTIVE_TEXT_BGR,
            scale=0.52,
            thickness=1,
        )
        self._put_segment(
            canvas,
            (
                f"TRAFFIC_ZONE={'IN' if decision_zone else 'OUT'} "
                f"cy<={self._decision_center_y_max:.2f} "
                f"w>={self._decision_min_width:.2f}"
            ),
            x,
            26,
            signal_color if decision_zone else INACTIVE_TEXT_BGR,
            scale=0.52,
            thickness=1,
        )

        # Row 2: the mode and countdown share the exact color of the selected
        # /center_path.  GENERAL has no timer by design.
        mode_color = self._mode_color(cnn_mode)
        timer_text = (
            "TIMER=--"
            if cnn_mode == "GENERAL"
            else (
                f"TIMER={mode_remaining_sec:.1f}/{mode_duration_sec:.1f}s"
                if mode_duration_sec > 0.0
                else f"TIMER={mode_remaining_sec:.1f}s"
            )
        )
        race_short = {
            "WAIT_GREEN": "WAIT",
            "RUNNING": "RUN",
            "FINISHED": "DONE",
        }.get(race_state, race_state or "WAIT")
        race_color = (
            (30, 30, 245)
            if race_state == "FINISHED"
            else ((30, 225, 30) if race_go else (0, 210, 255))
        )
        x = self._put_segment(
            canvas,
            (
                f"RACE={race_short} LAP={race_lap_count}/{race_target_laps} "
                f"FIN={'ON' if race_finish_enabled else 'OFF'}"
            ),
            14,
            55,
            race_color,
            scale=0.58,
            thickness=2,
            gap=20,
        )
        x = self._put_segment(
            canvas,
            f"MODE={cnn_mode}",
            x,
            55,
            mode_color,
            scale=0.68,
            thickness=2,
            gap=22,
        )
        x = self._put_segment(
            canvas,
            timer_text,
            x,
            55,
            mode_color,
            scale=0.62,
            thickness=2,
            gap=22,
        )
        x = self._put_segment(
            canvas,
            f"PATH={len(selected)} {route.upper()}",
            x,
            55,
            PRIMARY_TEXT_BGR,
            scale=0.50,
            thickness=1,
        )
        x = self._put_segment(
            canvas,
            f"LIDAR={raw_lidar_points}->{used_lidar_points} {lidar_preprocess}",
            x,
            55,
            PRIMARY_TEXT_BGR,
            scale=0.48,
            thickness=1,
        )
        self._put_segment(
            canvas,
            (
                f"{cnn_status} LAP_ARM={int(race_encounter_armed)} "
                f"CD={race_cooldown_remaining_sec:.1f}s"
            ),
            x,
            55,
            PRIMARY_TEXT_BGR,
            scale=0.46,
            thickness=1,
            gap=0,
        )

        # Row 3: trigger channels stay gray while idle.  Each one lights in the
        # color of the mode it can activate when evidence rises or remains true.
        now_mono = time.monotonic()
        shortcut_color = MODE_COLORS_BGR["SHORTCUT"]
        cone_color = MODE_COLORS_BGR["CONE"]
        overtake_color = MODE_COLORS_BGR["OVERTAKE"]
        left_active = bool(
            left_streak > 0
            or cnn_mode == "SHORTCUT"
            or now_mono < trigger_highlight_until["LEFT"]
        )
        cone_diag_age = now_mono - cone_detection_seen_mono
        cone_diag_fresh = bool(cone_detection and cone_diag_age <= 3.0)
        cone_y2 = cone_detection.get("y2_norm") if cone_diag_fresh else None
        cone_y2_px = cone_detection.get("y2_px") if cone_diag_fresh else None
        cone_approach_threshold_norm = cone_detection.get(
            "approach_threshold_norm", self._cone_approach_bottom_y_min
        )
        cone_threshold_norm = cone_detection.get(
            "threshold_norm", self._cone_enter_bottom_y_min
        )
        cone_seen = bool(
            cone_diag_fresh
            or cone_approach_ready
            or cone_approach_latch_active
            or cone_enter_ready
            or cone_trigger_raw
            or cone_state_latch_active
            or cone_mode
            or cnn_mode == "CONE"
            or now_mono < trigger_highlight_until["START_R"]
        )
        cone_threshold_pass = bool(
            cone_y2 is not None and cone_y2 >= cone_threshold_norm
        )
        cone_approach_pass = bool(
            cone_y2 is not None and cone_y2 >= cone_approach_threshold_norm
        )
        reference_ok = road_reference not in (
            "",
            "unavailable",
            "not_evaluated_in_cone",
        )
        points_ok = road_trigger_points >= overtake_min_raw_points
        ratio_ok = road_trigger_inside_ratio >= road_trigger_min_ratio
        overtake_mode_active = cnn_mode == "OVERTAKE"
        overtake_steps = sum(
            (reference_ok, points_ok, ratio_ok, road_trigger, overtake_mode_active)
        )
        overtake_active = bool(
            road_trigger
            or overtake_confirm_count > 0
            or overtake_mode_active
            or now_mono < trigger_highlight_until["OVERTAKE"]
        )

        x = self._put_segment(
            canvas,
            f"LEFT={left_streak}/2",
            14,
            84,
            shortcut_color if left_active else INACTIVE_TEXT_BGR,
            scale=0.50,
            thickness=2 if left_active else 1,
            gap=18,
        )
        x = self._put_segment(
            canvas,
            "START_R",
            x,
            84,
            cone_color if cone_seen else INACTIVE_TEXT_BGR,
            scale=0.50,
            thickness=2 if cone_seen else 1,
            gap=8,
        )
        x = self._put_segment(
            canvas,
            (
                f"Y2={cone_y2:.3f} ({cone_y2_px:.0f}px)"
                if cone_y2 is not None and cone_y2_px is not None
                else "Y2=--"
            ),
            x,
            84,
            cone_color if cone_approach_pass else INACTIVE_TEXT_BGR,
            scale=0.47,
            thickness=1,
            gap=8,
        )
        x = self._put_segment(
            canvas,
            (
                f"SLOW={int(cone_approach_latch_active)} "
                f">={cone_approach_threshold_norm:.2f} "
                f"[{cone_approach_confirm_count}/{cone_approach_confirm_frames}]"
            ),
            x,
            84,
            cone_color
            if (cone_approach_ready or cone_approach_latch_active)
            else INACTIVE_TEXT_BGR,
            scale=0.47,
            thickness=1,
            gap=8,
        )
        x = self._put_segment(
            canvas,
            (
                f"CONE={int(cone_trigger_raw or cone_state_latch_active)} "
                f">={cone_threshold_norm:.2f} "
                f"[{cone_confirm_count}/{cone_confirm_frames}]"
            ),
            x,
            84,
            cone_color
            if (cone_threshold_pass or cone_enter_ready or cone_trigger_raw or cone_state_latch_active)
            else INACTIVE_TEXT_BGR,
            scale=0.47,
            thickness=1,
            gap=18,
        )
        x = self._put_segment(
            canvas,
            f"OVERTAKE={overtake_steps}/5",
            x,
            84,
            overtake_color if overtake_active else INACTIVE_TEXT_BGR,
            scale=0.50,
            thickness=2 if overtake_active else 1,
            gap=9,
        )
        offset_text = (
            f"{obstacle_lateral_offset_m:+.3f}m"
            if obstacle_lateral_offset_m is not None
            else "--"
        )
        x = self._put_segment(
            canvas,
            (
                f"STRAT={overtake_strategy} LANE={obstacle_lane} "
                f"OFF={offset_text} AMB={int(obstacle_side_ambiguous)} "
                f"WIN={int(hardcode_window_active)}:"
                f"{post_cone_hardcode_remaining_sec:.1f}s"
            ),
            x,
            84,
            overtake_color if overtake_active else INACTIVE_TEXT_BGR,
            scale=0.43,
            thickness=1,
            gap=9,
        )
        if overtake_suppressed_by_start_r:
            x = self._put_segment(
                canvas,
                "BLOCK=START_R",
                x,
                84,
                cone_color,
                scale=0.47,
                thickness=2,
                gap=9,
            )
        for label, active in (
            ("REF", reference_ok),
            (
                f"PTS={road_trigger_points}/{road_trigger_cluster_points}"
                f">={overtake_min_raw_points}",
                points_ok,
            ),
            (
                f"RATIO={road_trigger_inside_ratio:.2f}/{road_trigger_min_ratio:.2f}",
                ratio_ok,
            ),
            (
                f"ROI={road_trigger_x_min_m:.1f}..{road_trigger_x_max_m:.1f}m"
                f"+{road_trigger_inward_margin_m:.2f}",
                road_trigger,
            ),
            (
                f"CONF={overtake_confirm_count}/{overtake_confirm_frames}",
                overtake_confirm_count > 0 or overtake_mode_active,
            ),
        ):
            x = self._put_segment(
                canvas,
                label,
                x,
                84,
                overtake_color if active else INACTIVE_TEXT_BGR,
                scale=0.45,
                thickness=1,
                gap=8,
            )
        self._put_segment(
            canvas,
            f"ARM={int(overtake_trigger_armed)}",
            x,
            84,
            INACTIVE_TEXT_BGR,
            scale=0.45,
            thickness=1,
            gap=0,
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
