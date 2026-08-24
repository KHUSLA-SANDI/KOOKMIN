#!/usr/bin/env python3
"""Combine full-topology camera BEV with mode-specific LiDAR and run the path CNN.

The input BEV message contains yellow/white channels only.  A white-boundary
filter creates the training-time ``lidar_filtered`` channel for GENERAL,
SHORTCUT, and OVERTAKE.  CONE receives the calibrated/self-masked raw LiDAR
without the outside-white removal.  Exactly one of four resident CNNs runs per
frame and publishes directly on ``/center_path``.  Confirmed LEFT,
conservative in-road LiDAR evidence, and START_R select fixed-duration modes.
RED/YELLOW stop remains an orthogonal drive-gate decision.
"""

from __future__ import annotations

from collections import deque
import json
import time
from typing import Deque, Optional, Tuple

import numpy as np

import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Bool, String

from .cnn_modes import (
    MODE_CONE,
    MODE_GENERAL,
    MODE_OVERTAKE,
    MODE_SHORTCUT,
    CnnModeController,
    ConfirmedOneShot,
)
from .hardcoded_overtake import (
    OBSTACLE_UNKNOWN,
    OVERTAKE_STRATEGIES,
    STRATEGY_HARDCODED_ALL,
    STRATEGY_HARDCODED_POST_CONE,
    DirectionalConfirmedOneShot,
    HardcodedOvertakeRequest,
    PostConeHardcodeWindow,
    avoidance_direction,
    classify_obstacle_lane,
)
from .lidar_geometry import (
    GRID_H,
    GRID_W,
    occupied_cell_count,
    points_to_grid,
    scan_to_grid,
)
from .path_contract import (
    INPUT_SHAPE,
    MODEL_KIND_LEGACY,
    PathGuardConfig,
    SanitizedPath,
    count_camera_evidence,
)
from .path_model import load_path_model
from .road_lidar_filter import filter_and_detect_in_road, reference_from_mid_grid
from .signal_mission import (
    ROUTE_MAIN,
    ROUTE_SHORTCUT,
    RaceLapController,
    TrafficMissionController,
    decode_signal_payload_full,
)


LATEST_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def path_message(path: SanitizedPath, header) -> PoseArray:
    message = PoseArray()
    # Do not alias and mutate the subscriber's Header object when replacing the
    # frame id.  Keep the source timestamp so downstream stale checks remain
    # tied to the BEV that produced this prediction.
    message.header.stamp = header.stamp
    message.header.frame_id = "lidar_frame"
    if not path.usable:
        return message
    for x, y in zip(path.x, path.y):
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = 0.0
        pose.orientation.w = 1.0
        message.poses.append(pose)
    return message


class CnnPathNode(Node):
    def __init__(self):
        super().__init__("cnn_path_node")
        p = self._parameter

        self._bev_topic = str(p("bev_topic", "/perception/bev"))
        self._scan_topic = str(p("scan_topic", "/scan"))
        main_topic = str(p("main_path_topic", "/cnn/path_main"))
        shortcut_topic = str(p("shortcut_path_topic", "/cnn/path_shortcut"))
        center_topic = str(p("center_path_topic", "/center_path"))
        cnn_input_bev_topic = str(
            p("cnn_input_bev_topic", "/debug/cnn_input_bev")
        )
        signal_topic = str(p("signal_topic", "/perception/signals"))
        route_intent_topic = str(p("route_intent_topic", "/route_intent"))
        route_reset_topic = str(p("route_reset_topic", "/route_intent_reset"))
        traffic_stop_topic = str(p("traffic_stop_topic", "/traffic_stop"))
        traffic_stop_reset_topic = str(
            p("traffic_stop_reset_topic", "/traffic_stop_reset")
        )
        race_go_topic = str(p("race_go_topic", "/race_go"))
        cone_trigger_topic = str(
            p("cone_trigger_topic", "/perception/cone_trigger")
        )
        start_r_detected_topic = str(
            p("start_r_detected_topic", "/perception/start_r_detected")
        )
        self._mode_topic = str(p("cnn_mode_topic", "/cnn_mode"))
        self._cone_mode_topic = str(p("cone_mode_topic", "/cone_mode"))
        self._overtake_strategy = str(
            p("overtake_strategy", STRATEGY_HARDCODED_ALL)
        ).strip().lower()
        if self._overtake_strategy not in OVERTAKE_STRATEGIES:
            raise ValueError(
                "overtake_strategy must be hardcoded_all or hardcoded_post_cone"
            )
        self._overtake_block_request_topic = str(
            p("overtake_block_request_topic", "/motion/overtake_block_request")
        )
        self._obstacle_lane_deadband = float(
            p("obstacle_lane_deadband_m", 0.10)
        )
        self._post_cone_hardcode_window_sec = float(
            p("post_cone_hardcode_window_sec", 5.0)
        )
        self._enable_center_path = bool(p("enable_center_path", True))
        shortcut_hold_sec = float(p("shortcut_hold_sec", 10.0))
        overtake_hold_sec = float(p("overtake_hold_sec", 7.0))
        cone_hold_sec = float(p("cone_hold_sec", 20.0))
        self._mode_hold_seconds = {
            MODE_GENERAL: 0.0,
            MODE_SHORTCUT: shortcut_hold_sec,
            MODE_OVERTAKE: overtake_hold_sec,
            MODE_CONE: cone_hold_sec,
        }
        signal_confirm_frames = int(p("left_confirm_frames", 2))
        signal_confidence = float(p("left_confidence", 0.25))
        decision_center_y_max = float(
            p("signal_decision_center_y_max", 0.22)
        )
        decision_min_width = float(p("signal_decision_min_width", 0.08))
        self._mission = TrafficMissionController(
            confirm_frames=signal_confirm_frames,
            confidence=signal_confidence,
            shortcut_hold_sec=shortcut_hold_sec,
            left_confirm_window_sec=float(p("left_confirm_window_sec", 1.5)),
            decision_center_y_max=decision_center_y_max,
            decision_min_width=decision_min_width,
            left_rearm_absent_frames=int(p("left_rearm_absent_frames", 3)),
        )
        self._race = RaceLapController(
            confirm_frames=signal_confirm_frames,
            confidence=signal_confidence,
            decision_center_y_max=decision_center_y_max,
            decision_min_width=decision_min_width,
            target_laps=int(p("race_target_laps", 3)),
            cooldown_sec=float(p("race_lap_cooldown_sec", 10.0)),
            clear_frames=int(p("race_lap_clear_frames", 5)),
            finish_enabled=bool(p("race_finish_enabled", True)),
        )
        self._mode_controller = CnnModeController(
            shortcut_hold_sec=shortcut_hold_sec,
            overtake_hold_sec=overtake_hold_sec,
            cone_hold_sec=cone_hold_sec,
        )
        overtake_confirm_frames = int(p("overtake_confirm_frames", 2))
        overtake_rearm_frames = int(p("overtake_rearm_clear_frames", 5))
        self._overtake_direction_latch = DirectionalConfirmedOneShot(
            confirm_frames=overtake_confirm_frames,
            rearm_clear_frames=overtake_rearm_frames,
        )
        self._overtake_cnn_latch = ConfirmedOneShot(
            confirm_frames=overtake_confirm_frames,
            rearm_clear_frames=overtake_rearm_frames,
        )
        # Include one process-unique prefix so a restarted perception node can
        # never reuse an event id still remembered by simple_motion.
        self._block_event_prefix = str(time.time_ns())
        self._block_event_counter = 0
        self._cone_trigger_high = False
        self._start_r_detected = False
        self._last_route_intent = ROUTE_MAIN
        self._last_published_mode = ""
        self._last_center_message: Optional[PoseArray] = None
        self._last_observed_mode = MODE_GENERAL
        self._post_cone_window = PostConeHardcodeWindow(
            window_sec=self._post_cone_hardcode_window_sec
        )
        legacy_model_path = str(p("model_path", ""))
        legacy_model_sha = str(p("expected_model_sha256", ""))
        general_model_path = str(p("general_model_path", legacy_model_path))
        general_model_sha = str(p("general_expected_sha256", legacy_model_sha))
        shortcut_model_path = str(p("shortcut_model_path", general_model_path))
        shortcut_model_sha = str(p("shortcut_expected_sha256", general_model_sha))
        overtake_model_path = str(p("overtake_model_path", general_model_path))
        overtake_model_sha = str(p("overtake_expected_sha256", general_model_sha))
        cone_model_path = str(p("cone_model_path", general_model_path))
        cone_model_sha = str(p("cone_expected_sha256", ""))
        device = str(p("device", "cpu"))
        torch_threads = int(p("torch_threads", 2))
        allow_unverified_legacy = bool(p("allow_unverified_legacy", False))

        self._range_scale = float(p("lidar_range_scale", 1.1476))
        self._range_min = float(p("lidar_min_range", 0.05))
        self._range_max = float(p("lidar_max_range", 8.0))
        self._self_x = float(p("lidar_self_x_abs", 0.25))
        self._self_y = float(p("lidar_self_y_abs", 0.15))
        self._lidar_radius = int(p("lidar_radius_cells", 1))
        self._min_camera_cells = int(p("min_camera_cells", 1))
        self._model_lidar_safety_margin = float(
            p("model_lidar_safety_margin_m", 0.025)
        )
        self._model_white_guard = float(p("model_white_guard_m", 0.025))
        # User-selected trigger contract: at least 10 cm inside both white
        # boundaries and before the x=1.4 m forward line.
        self._overtake_inward_margin = float(
            p("overtake_inward_margin_m", 0.10)
        )
        self._overtake_white_guard = float(p("overtake_white_guard_m", 0.05))
        self._overtake_x_min = float(p("overtake_x_min_m", 0.0))
        self._overtake_x_max = float(p("overtake_x_max_m", 1.40))
        self._overtake_min_points = int(p("overtake_min_raw_points", 3))
        self._overtake_min_inside_ratio = float(
            p("overtake_min_inside_ratio", 0.50)
        )
        self._overtake_min_road_width = float(
            p("overtake_min_road_width_m", 0.35)
        )
        self._overtake_max_road_width = float(
            p("overtake_max_road_width_m", 1.50)
        )
        self._overtake_cluster_gap_base = float(
            p("overtake_cluster_gap_base_m", 0.15)
        )
        self._overtake_cluster_gap_per_m = float(
            p("overtake_cluster_gap_per_m", 0.03)
        )
        self._overtake_max_cluster_extent = float(
            p("overtake_max_cluster_extent_m", 0.90)
        )
        self._camera_scan_offset_ns = int(
            float(p("camera_to_scan_offset_sec", 0.0)) * 1e9
        )
        self._sync_slop_ns = int(float(p("sync_slop_sec", 0.18)) * 1e9)
        self._history_ns = int(float(p("scan_history_sec", 1.0)) * 1e9)
        self._max_bev_age_ns = int(float(p("max_bev_age_sec", 0.35)) * 1e9)
        self._require_scan = bool(p("require_fresh_scan", True))

        if torch_threads < 1:
            raise ValueError("torch_threads must be positive")
        if self._sync_slop_ns <= 0 or self._history_ns <= 0:
            raise ValueError("sync_slop_sec and scan_history_sec must be positive")
        if self._max_bev_age_ns <= 0:
            raise ValueError("max_bev_age_sec must be positive")
        if self._history_ns < self._sync_slop_ns + abs(self._camera_scan_offset_ns):
            raise ValueError(
                "scan_history_sec must cover sync_slop_sec plus camera offset"
            )
        if self._lidar_radius < 0:
            raise ValueError("lidar_radius_cells must be non-negative")
        if self._min_camera_cells < 0:
            raise ValueError("min_camera_cells must be non-negative")
        if not np.isclose(self._overtake_inward_margin, 0.10, rtol=0.0, atol=1e-9):
            raise ValueError(
                "overtake_inward_margin_m is fixed to the approved 0.10 m contract"
            )
        if (
            not np.isfinite(self._overtake_x_min)
            or not np.isfinite(self._overtake_x_max)
            or self._overtake_x_min < 0.0
            or self._overtake_x_max <= self._overtake_x_min
            or self._overtake_x_max > 3.0
        ):
            raise ValueError(
                "overtake x window must satisfy 0 <= min < max <= 3.0 m"
            )
        if not np.isfinite(self._obstacle_lane_deadband) or not (
            0.02 <= self._obstacle_lane_deadband <= 0.50
        ):
            raise ValueError("obstacle_lane_deadband_m must be within 0.02..0.50 m")
        if (
            not np.isfinite(self._post_cone_hardcode_window_sec)
            or self._post_cone_hardcode_window_sec <= 0.0
        ):
            raise ValueError("post_cone_hardcode_window_sec must be positive and finite")

        guard = PathGuardConfig(
            valid_threshold=float(p("valid_threshold", 0.50)),
            shortcut_threshold=float(p("shortcut_threshold", 0.50)),
            min_points=int(p("min_valid_points", 6)),
            min_span_m=float(p("min_valid_span_m", 0.50)),
            max_abs_y_m=float(p("max_abs_y_m", 1.500001)),
            max_abs_slope=float(p("max_abs_slope", 3.1)),
            max_abs_curvature=float(p("max_abs_curvature", 12.0)),
        )

        import torch

        torch.set_num_threads(max(1, torch_threads))
        model_specs = {
            MODE_GENERAL: (general_model_path, general_model_sha),
            MODE_SHORTCUT: (shortcut_model_path, shortcut_model_sha),
            MODE_OVERTAKE: (overtake_model_path, overtake_model_sha),
            MODE_CONE: (cone_model_path, cone_model_sha),
        }
        cache = {}
        self._path_models = {}
        for mode, (path, expected_sha) in model_specs.items():
            key = (str(path), str(expected_sha).strip().lower())
            if key not in cache:
                cache[key] = load_path_model(
                    path,
                    device=device,
                    guard=guard,
                    expected_sha256=expected_sha,
                )
            model = cache[key]
            if model.model_kind == MODEL_KIND_LEGACY and not allow_unverified_legacy:
                raise RuntimeError(
                    f"legacy checkpoint execution is disabled for {mode}"
                )
            self._path_models[mode] = model

        # (receive time, scan-ordered calibrated points, raw raster cell count)
        self._scan_history: Deque[Tuple[int, np.ndarray, int]] = deque()
        self._last_road_reference: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._last_bev_stamp_ns = -1
        self._last_scan_error_log_ns = -1
        self._main_pub = self.create_publisher(PoseArray, main_topic, 1)
        self._shortcut_pub = self.create_publisher(PoseArray, shortcut_topic, 1)
        self._center_pub = self.create_publisher(PoseArray, center_topic, 1)
        self._route_pub = self.create_publisher(String, route_intent_topic, 1)
        self._traffic_stop_pub = self.create_publisher(Bool, traffic_stop_topic, 1)
        self._race_go_pub = self.create_publisher(Bool, race_go_topic, 1)
        self._mode_pub = self.create_publisher(String, self._mode_topic, 1)
        self._cone_mode_pub = self.create_publisher(Bool, self._cone_mode_topic, 1)
        self._overtake_block_request_pub = self.create_publisher(
            String, self._overtake_block_request_topic, 1
        )
        self._diag_pub = self.create_publisher(String, "/debug/cnn_path", 10)
        self._cnn_input_bev_pub = self.create_publisher(
            Image, cnn_input_bev_topic, LATEST_QOS
        )
        self.create_subscription(
            LaserScan,
            self._scan_topic,
            self._on_scan,
            qos_profile_sensor_data,
        )
        self.create_subscription(Image, self._bev_topic, self._on_bev, LATEST_QOS)
        self.create_subscription(String, signal_topic, self._on_signals, 1)
        self.create_subscription(Bool, cone_trigger_topic, self._on_cone_trigger, 1)
        self.create_subscription(
            Bool, start_r_detected_topic, self._on_start_r_detected, 1
        )
        self.create_subscription(Bool, route_reset_topic, self._on_route_reset, 1)
        self.create_subscription(
            Bool,
            traffic_stop_reset_topic,
            self._on_traffic_stop_reset,
            1,
        )
        kinds = {mode: model.model_kind for mode, model in self._path_models.items()}
        self.get_logger().info(
            f"four-mode CNN path ready kinds={kinds} input={INPUT_SHAPE} "
            f"device={device} direct_center_path={str(self._enable_center_path).lower()}"
        )

    def _parameter(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _current_route(self) -> str:
        mode = self._current_mode().mode
        return ROUTE_SHORTCUT if mode == MODE_SHORTCUT else ROUTE_MAIN

    def _observe_mode_snapshot(self, snapshot, *, now_sec: float) -> None:
        """Observe every externally used mode transition in one place.

        ``hardcoded_post_cone`` is armed only by CONE's natural timer expiry.
        Its five-second deadline is calculated from the original CONE
        deadline, not from a delayed camera callback.  An explicit reset never
        arms the fixed-obstacle window.
        """

        previous = self._last_observed_mode
        self._post_cone_window.observe_transition(
            previous_mode=previous,
            mode=snapshot.mode,
            changed=snapshot.changed,
            reason=snapshot.reason,
            remaining_sec=snapshot.remaining_sec,
            now_sec=now_sec,
        )
        if previous == MODE_CONE and snapshot.mode == MODE_GENERAL:
            if snapshot.changed and snapshot.reason == "timer_expired":
                remaining = self._post_cone_window.remaining_sec(now_sec)
                self.get_logger().warning(
                    "post-CONE fixed-obstacle hardcode window armed: "
                    f"remaining={remaining:.2f}s"
                )
        self._last_observed_mode = snapshot.mode

    def _post_cone_window_remaining(self, now_sec: Optional[float] = None) -> float:
        now = self._now_sec() if now_sec is None else float(now_sec)
        return self._post_cone_window.remaining_sec(now)

    def _hardcode_window_active(self, now_sec: Optional[float] = None) -> bool:
        if self._overtake_strategy == STRATEGY_HARDCODED_ALL:
            return True
        return self._post_cone_window_remaining(now_sec) > 0.0

    def _overtake_latch_diagnostics(self) -> dict:
        hardcode_active = self._hardcode_window_active()
        active_latch = (
            self._overtake_direction_latch
            if hardcode_active
            else self._overtake_cnn_latch
        )
        return {
            "overtake_confirm_count": active_latch.positive_count,
            "overtake_confirm_frames": active_latch.confirm_frames,
            "overtake_trigger_armed": active_latch.armed,
            "overtake_direction_confirm_count": (
                self._overtake_direction_latch.positive_count
            ),
            "overtake_cnn_confirm_count": self._overtake_cnn_latch.positive_count,
            "obstacle_lane_candidate": self._overtake_direction_latch.candidate,
            "hardcode_window_active": hardcode_active,
            "post_cone_hardcode_remaining_sec": round(
                self._post_cone_window_remaining(), 2
            ),
        }

    def _current_mode(self):
        now_sec = self._now_sec()
        snapshot = self._mode_controller.snapshot(now_sec)
        self._observe_mode_snapshot(snapshot, now_sec=now_sec)
        if snapshot.changed:
            self.get_logger().info("CNN mode timer ended: GENERAL")
        return snapshot

    def _publish_mode(self, snapshot=None) -> None:
        if snapshot is None:
            state = self._current_mode()
        else:
            state = snapshot
            self._observe_mode_snapshot(state, now_sec=self._now_sec())
        self._mode_pub.publish(String(data=state.mode))
        self._cone_mode_pub.publish(Bool(data=state.mode == MODE_CONE))
        if state.mode != self._last_published_mode:
            self.get_logger().info(
                f"CNN mode changed: {state.mode} reason={state.reason} "
                f"remaining={state.remaining_sec:.1f}s"
            )
            self._last_published_mode = state.mode

    def _publish_overtake_block_request(
        self,
        *,
        obstacle_lane: str,
        lateral_offset_m: float,
    ) -> HardcodedOvertakeRequest:
        """Publish exactly one versioned request; never change CNN mode."""

        self._block_event_counter += 1
        request = HardcodedOvertakeRequest(
            event_id=f"{self._block_event_prefix}-{self._block_event_counter}",
            direction=avoidance_direction(obstacle_lane),
            obstacle_lane=obstacle_lane,
            lateral_offset_m=lateral_offset_m,
        )
        self._overtake_block_request_pub.publish(String(data=request.to_json()))
        self.get_logger().warning(
            "hardcoded obstacle block requested: "
            f"event={request.event_id} obstacle={request.obstacle_lane} "
            f"move={request.direction} offset={request.lateral_offset_m:+.3f}m"
        )
        return request

    def _publish_route_intent(self) -> None:
        route = self._current_route()
        message = String()
        message.data = route
        self._route_pub.publish(message)
        if route != self._last_route_intent:
            self.get_logger().info(f"direct path route changed: {route}")
            self._last_route_intent = route

    def _publish_traffic_stop(self) -> None:
        message = Bool()
        # A completed race is a terminal STOP latch.  A later GREEN or a
        # manual traffic-stop reset may clear the signal mission, but cannot
        # release the three-lap finish.
        message.data = bool(self._mission.traffic_stop or self._race.finished)
        self._traffic_stop_pub.publish(message)

    def _publish_race_go(self) -> None:
        self._race_go_pub.publish(
            Bool(data=bool(self._race.started and not self._race.finished))
        )

    def _race_diagnostics(self) -> dict:
        now_sec = self._now_sec()
        return {
            "race_state": self._race.state_name,
            "race_started": bool(self._race.started),
            "race_finished": bool(self._race.finished),
            "race_go": bool(self._race.started and not self._race.finished),
            "race_lap_count": int(self._race.lap_count),
            "race_target_laps": int(self._race.target_laps),
            "race_finish_enabled": bool(self._race.finish_enabled),
            "race_encounter_armed": bool(self._race.encounter_armed),
            "race_cooldown_remaining_sec": round(
                self._race.cooldown_remaining_sec(now_sec), 2
            ),
        }

    def _on_signals(self, message: String) -> None:
        """Update the route latch from the same YOLO pass used for BEV."""

        try:
            sequence, signals, detections = decode_signal_payload_full(message.data)
        except ValueError as exc:
            self.get_logger().warning(
                f"rejected signal payload: {exc}",
                throttle_duration_sec=1.0,
            )
            return
        now_sec = self._now_sec()
        race_update = self._race.observe(
            sequence,
            signals,
            detections,
            now_sec=now_sec,
        )
        update = self._mission.observe(
            sequence,
            signals,
            detections,
            now_sec=now_sec,
        )
        if not update.accepted:
            return
        if update.source_restarted:
            self.get_logger().warning(
                "YOLO signal sequence restarted; signal confirmation streaks reset"
            )
        if update.route_changed:
            if update.route_intent == ROUTE_SHORTCUT:
                if self._race.started and not self._race.finished:
                    # Observe any natural timer expiry before asking the mode
                    # controller to enter another specialized mode.
                    self._current_mode()
                    mode = self._mode_controller.trigger(
                        MODE_SHORTCUT,
                        now_sec=now_sec,
                        reason="left_signal_confirmed",
                    )
                    self.get_logger().info(
                        "LEFT confirmed: shortcut selected for %.1f seconds"
                        % self._mission.shortcut_hold_sec
                    )
                    self._publish_mode(mode)
                else:
                    self.get_logger().info(
                        "LEFT ignored while race is not RUNNING"
                    )
            else:
                self.get_logger().info(
                    f"signal decision selected main: {update.dominant_signal}"
                )
            self._publish_route_intent()
        if update.stop_changed:
            state = "STOP" if update.traffic_stop else "GO"
            self.get_logger().warning(
                f"traffic decision changed: {state} signal={update.dominant_signal}"
            )
        if race_update.start_changed:
            self.get_logger().warning(
                "RACE START: initial GREEN confirmed; LAP=0"
            )
        if race_update.lap_changed:
            self.get_logger().warning(
                f"RACE LAP={race_update.lap_count}/{self._race.target_laps}"
            )
        if race_update.finish_changed:
            self.get_logger().warning(
                f"RACE FINISHED: {race_update.lap_count} laps; terminal STOP"
            )
        self._publish_race_go()
        self._publish_traffic_stop()

    def _on_cone_trigger(self, message: Bool) -> None:
        detected = bool(message.data)
        rising = detected and not self._cone_trigger_high
        self._cone_trigger_high = detected
        if not rising:
            return
        self._current_mode()
        mode = self._mode_controller.trigger(
            MODE_CONE,
            now_sec=self._now_sec(),
            reason="start_r_confirmed",
        )
        self._publish_mode(mode)

    def _on_start_r_detected(self, message: Bool) -> None:
        """Block OVERTAKE as soon as START_R is visible, before y2 entry."""

        self._start_r_detected = bool(message.data)

    def _on_route_reset(self, message: Bool) -> None:
        if not message.data:
            return
        self._mission.reset_route()
        mode = self._mode_controller.reset(
            now_sec=self._now_sec(), reason="route_reset"
        )
        # The previous shortcut must not remain cached in motion until the
        # next camera frame.  Invalidate it using the original source stamp;
        # never make an old path look new by assigning the reset time.
        if self._last_center_message is not None:
            invalid = PoseArray()
            invalid.header.stamp.sec = self._last_center_message.header.stamp.sec
            invalid.header.stamp.nanosec = (
                self._last_center_message.header.stamp.nanosec
            )
            invalid.header.frame_id = self._last_center_message.header.frame_id
            self._center_pub.publish(invalid)
            self._last_center_message = invalid
        self.get_logger().info("direct path route explicitly reset: main")
        self._publish_mode(mode)
        self._publish_route_intent()

    def _on_traffic_stop_reset(self, message: Bool) -> None:
        if not message.data:
            return
        self._mission.reset_stop()
        self._publish_traffic_stop()
        if self._race.finished:
            self.get_logger().warning(
                "traffic STOP reset requested, but race finish STOP remains latched"
            )
        else:
            self.get_logger().warning("traffic STOP explicitly reset")

    def _publish_center(self, path: SanitizedPath, header) -> SanitizedPath:
        published = path
        if not self._enable_center_path:
            published = self._invalid_path("center_path_output_disabled")
        message = path_message(published, header)
        self._center_pub.publish(message)
        self._last_center_message = message
        self._publish_mode()
        self._publish_route_intent()
        return published

    def _on_scan(self, message: LaserScan) -> None:
        received_ns = self.get_clock().now().nanoseconds
        try:
            grid, points = scan_to_grid(
                message.ranges,
                message.angle_min,
                message.angle_increment,
                radius_cells=self._lidar_radius,
                range_scale=self._range_scale,
                min_range_m=self._range_min,
                max_range_m=self._range_max,
                self_x_abs_m=self._self_x,
                self_y_abs_m=self._self_y,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            # A malformed scan must not tear down the process.  Do not append a
            # replacement zero grid: require_fresh_scan will fail closed until
            # a valid scan arrives.
            if (
                self._last_scan_error_log_ns < 0
                or received_ns - self._last_scan_error_log_ns >= 2_000_000_000
            ):
                self.get_logger().warning(f"dropping invalid LaserScan: {exc}")
                self._last_scan_error_log_ns = received_ns
            return
        self._scan_history.append(
            (received_ns, points, occupied_cell_count(grid))
        )
        oldest = received_ns - self._history_ns
        while self._scan_history and self._scan_history[0][0] < oldest:
            self._scan_history.popleft()

    def _nearest_scan(self, bev_received_ns: int):
        if not self._scan_history:
            return None
        target = bev_received_ns - self._camera_scan_offset_ns
        item = min(self._scan_history, key=lambda row: abs(row[0] - target))
        delta = abs(item[0] - target)
        if delta > self._sync_slop_ns:
            return None
        return item, delta

    @staticmethod
    def _decode_bev(message: Image) -> np.ndarray:
        if message.height != GRID_H or message.width != GRID_W:
            raise ValueError(
                f"BEV image size {message.height}x{message.width} != {GRID_H}x{GRID_W}"
            )
        # Channel order is semantic (yellow, white, lidar), not RGB/BGR.
        # Accepting an RGB-labelled image would silently change the contract.
        if message.encoding.lower() != "8uc3":
            raise ValueError(f"unsupported BEV encoding: {message.encoding}")
        expected_step = GRID_W * 3
        if int(message.step) != expected_step:
            raise ValueError(f"BEV step {message.step} != {expected_step}")
        flat = np.frombuffer(message.data, dtype=np.uint8)
        if flat.size != GRID_H * GRID_W * 3:
            raise ValueError(f"BEV byte count {flat.size} is invalid")
        hwc = flat.reshape(GRID_H, GRID_W, 3)
        chw = np.transpose(hwc, (2, 0, 1)).copy()
        if np.any(chw > 1):
            raise ValueError("BEV occupancy must be binary 0/1")
        if np.any(chw[2]):
            raise ValueError("incoming BEV lidar channel must be empty")
        return chw

    def _publish_cnn_input_bev(self, values: np.ndarray, header) -> None:
        """Publish the exact semantic 3-channel tensor before float conversion."""

        # This image is diagnostic-only.  Avoid a transpose, allocation,
        # serialization and DDS publish on every control frame when neither
        # the live viewer nor the recorder is connected.  Subscription counts
        # are checked each call, so a viewer started later begins receiving
        # frames without restarting the driving pipeline.
        if self._cnn_input_bev_pub.get_subscription_count() <= 0:
            return

        message = Image()
        message.header.stamp = header.stamp
        message.header.frame_id = "lidar_frame"
        message.height = GRID_H
        message.width = GRID_W
        message.encoding = "8UC3"
        message.is_bigendian = False
        message.step = GRID_W * 3
        hwc = np.ascontiguousarray(np.transpose(values, (1, 2, 0)))
        message.data = hwc.tobytes()
        self._cnn_input_bev_pub.publish(message)

    def _publish_empty(self, header, reason: str, **extra) -> None:
        empty = self._invalid_path(reason)
        self._main_pub.publish(path_message(empty, header))
        self._shortcut_pub.publish(path_message(empty, header))
        selected = self._publish_center(empty, header)
        mode_state = self._current_mode()
        route = ROUTE_SHORTCUT if mode_state.mode == MODE_SHORTCUT else ROUTE_MAIN
        payload = {
            "ok": False,
            "reason": reason,
            "cnn_mode": mode_state.mode,
            "mode_remaining_sec": round(mode_state.remaining_sec, 2),
            "mode_duration_sec": self._mode_hold_seconds[mode_state.mode],
            "mode_reason": mode_state.reason,
            "route_intent": route,
            "selected_route": route,
            "selected_usable": selected.usable,
            "selected_reason": selected.reason,
            "traffic_stop": bool(
                self._mission.traffic_stop or self._race.finished
            ),
            "left_streak": self._mission.left_streak,
            "cone_trigger_raw": bool(self._cone_trigger_high),
            "start_r_detected": bool(self._start_r_detected),
            "overtake_strategy": self._overtake_strategy,
            "overtake_block_request_topic": self._overtake_block_request_topic,
            "obstacle_lane_deadband_m": self._obstacle_lane_deadband,
            **self._overtake_latch_diagnostics(),
            **self._race_diagnostics(),
            **extra,
        }
        self._diag_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    @staticmethod
    def _invalid_path(reason: str) -> SanitizedPath:
        return SanitizedPath(
            usable=False,
            reason=reason,
            x=np.empty(0, np.float32),
            y=np.empty(0, np.float32),
            valid_probability=np.empty(0, np.float32),
        )

    def _on_bev(self, message: Image) -> None:
        callback_ns = self.get_clock().now().nanoseconds
        source_ns = stamp_to_ns(message.header.stamp)
        if source_ns <= 0:
            source_ns = callback_ns
        if source_ns <= self._last_bev_stamp_ns:
            return
        self._last_bev_stamp_ns = source_ns

        source_age_ns = callback_ns - source_ns
        if source_age_ns < -self._sync_slop_ns:
            self._publish_empty(message.header, "bev_stamp_is_in_future")
            return
        if source_age_ns > self._max_bev_age_ns:
            self._publish_empty(
                message.header,
                "stale_bev",
                source_age_ms=round(source_age_ns * 1e-6, 2),
            )
            return

        started = time.perf_counter()
        try:
            values = self._decode_bev(message)
        except ValueError as exc:
            self._publish_empty(message.header, f"bad_bev:{exc}")
            return
        decoded_at = time.perf_counter()
        camera_cells = count_camera_evidence(values)
        if camera_cells < self._min_camera_cells:
            self._publish_empty(
                message.header,
                "insufficient_camera_evidence",
                camera_cells=camera_cells,
                min_camera_cells=self._min_camera_cells,
            )
            return

        matched = self._nearest_scan(source_ns)
        if matched is None:
            if self._require_scan:
                self._publish_empty(message.header, "no_synchronized_scan")
                return
            raw_lidar_points = np.empty((0, 2), dtype=np.float32)
            raw_lidar_cells = 0
            scan_delta_ms: Optional[float] = None
        else:
            (_scan_ns, raw_lidar_points, raw_lidar_cells), delta_ns = matched
            scan_delta_ms = delta_ns * 1e-6

        mode_state = self._current_mode()
        fallback_x, fallback_y = reference_from_mid_grid(values[0])
        if fallback_x.size >= 2:
            reference = (fallback_x, fallback_y)
            reference_source = "current_mid"
        else:
            reference = self._last_road_reference
            reference_source = (
                "previous_selected_path" if reference is not None else "unavailable"
            )

        road_diag = {
            "reference": reference_source,
            "trigger_detected": False,
            "trigger_min_points": self._overtake_min_points,
            "trigger_min_inside_ratio": self._overtake_min_inside_ratio,
            "trigger_x_min_m": self._overtake_x_min,
            "trigger_x_max_m": self._overtake_x_max,
            "trigger_inward_margin_m": self._overtake_inward_margin,
            "strategy": self._overtake_strategy,
            "obstacle_lane_deadband_m": self._obstacle_lane_deadband,
            "obstacle_lane": OBSTACLE_UNKNOWN,
        }
        block_request: Optional[HardcodedOvertakeRequest] = None
        raw_lidar_grid = points_to_grid(
            raw_lidar_points, radius_cells=self._lidar_radius
        )
        filtered_points = np.empty((0, 2), dtype=np.float32)
        lidar_preprocess = "basic_range_self_only"
        if mode_state.mode == MODE_CONE:
            # The CONE checkpoint was trained with LiDAR before the
            # outside-white removal.  Range calibration and ego self-return
            # removal are sensor corrections and remain active.
            lidar = raw_lidar_grid
            lidar_points = int(raw_lidar_points.shape[0])
            road_diag["reference"] = "not_evaluated_in_cone"
            self._overtake_direction_latch.observe(None)
            self._overtake_cnn_latch.observe(False)
        elif reference is not None:
            lidar_preprocess = "outside_white_removed"
            try:
                road = filter_and_detect_in_road(
                    raw_lidar_points,
                    values[1],
                    reference[0],
                    reference[1],
                    model_safety_margin_m=self._model_lidar_safety_margin,
                    model_white_guard_m=self._model_white_guard,
                    trigger_inward_margin_m=self._overtake_inward_margin,
                    trigger_white_guard_m=self._overtake_white_guard,
                    trigger_x_min_m=self._overtake_x_min,
                    trigger_x_max_m=self._overtake_x_max,
                    trigger_min_points=self._overtake_min_points,
                    trigger_min_inside_ratio=self._overtake_min_inside_ratio,
                    trigger_min_road_width_m=self._overtake_min_road_width,
                    trigger_max_road_width_m=self._overtake_max_road_width,
                    trigger_lane_deadband_m=self._obstacle_lane_deadband,
                    cluster_gap_base_m=self._overtake_cluster_gap_base,
                    cluster_gap_per_m=self._overtake_cluster_gap_per_m,
                    max_cluster_extent_m=self._overtake_max_cluster_extent,
                )
            except ValueError as exc:
                self._publish_empty(
                    message.header,
                    "road_lidar_filter_error",
                    detail=str(exc),
                )
                return
            filtered_points = road.filtered_points
            road_diag.update(
                {
                    "trigger_detected": road.trigger_detected,
                    "trigger_cluster_points": road.trigger_cluster_points,
                    "trigger_inside_points": road.trigger_inside_points,
                    "trigger_inside_ratio": (
                        road.trigger_inside_points
                        / float(max(1, road.trigger_cluster_points))
                    ),
                    "reliable_boundary_rows": road.reliable_rows,
                    "removed_outside_points": road.removed_points,
                    "trigger_cluster_centroid_x_m": (
                        road.trigger_cluster_centroid_x_m
                    ),
                    "trigger_cluster_centroid_y_m": (
                        road.trigger_cluster_centroid_y_m
                    ),
                    "trigger_reference_y_m": road.trigger_reference_y_m,
                    "trigger_lateral_offset_m": road.trigger_lateral_offset_m,
                    "trigger_side_ambiguous": road.trigger_side_ambiguous,
                }
            )
            detected_before_start_r_suppression = bool(road.trigger_detected)
            suppressed_by_start_r = bool(
                detected_before_start_r_suppression and self._start_r_detected
            )
            effective_overtake_detected = bool(
                detected_before_start_r_suppression and not self._start_r_detected
            )
            road_diag.update(
                {
                    "trigger_detected_before_start_r_suppression": (
                        detected_before_start_r_suppression
                    ),
                    "suppressed_by_start_r": suppressed_by_start_r,
                    "trigger_detected": effective_overtake_detected,
                }
            )
            obstacle_lane = classify_obstacle_lane(
                road.trigger_lateral_offset_m if effective_overtake_detected else None,
                deadband_m=self._obstacle_lane_deadband,
            )
            trigger_allowed = (
                mode_state.mode == MODE_GENERAL and not self._cone_trigger_high
            )
            hardcode_window_active = self._hardcode_window_active()
            road_diag.update(
                {
                    "obstacle_lane": obstacle_lane,
                    "block_direction": (
                        avoidance_direction(obstacle_lane)
                        if obstacle_lane != OBSTACLE_UNKNOWN
                        else OBSTACLE_UNKNOWN
                    ),
                    "block_trigger_allowed": trigger_allowed,
                    "hardcode_window_active": hardcode_window_active,
                    "post_cone_hardcode_remaining_sec": (
                        self._post_cone_window_remaining()
                    ),
                }
            )
            if not detected_before_start_r_suppression:
                lane_observation = None
            elif (
                effective_overtake_detected
                and trigger_allowed
                and hardcode_window_active
            ):
                lane_observation = obstacle_lane
            else:
                # START_R, CONE, or another specialized mode is not a genuine
                # clear frame.  Reset a partial streak without re-arming a
                # block against the same visible object.
                lane_observation = OBSTACLE_UNKNOWN
            confirmed_lane = self._overtake_direction_latch.observe(
                lane_observation
            )
            cnn_overtake_event = self._overtake_cnn_latch.observe(
                effective_overtake_detected
            )
            if (
                hardcode_window_active
                and confirmed_lane is not None
                and trigger_allowed
            ):
                offset = road.trigger_lateral_offset_m
                if offset is None:
                    raise RuntimeError("confirmed obstacle lane has no lateral offset")
                block_request = self._publish_overtake_block_request(
                    obstacle_lane=confirmed_lane,
                    lateral_offset_m=offset,
                )
                road_diag["block_event_id"] = block_request.event_id
                # Hardcode execution deliberately stays GENERAL:
                # simple_motion temporarily owns steering exclusively.
            elif (
                not hardcode_window_active
                and cnn_overtake_event
                and trigger_allowed
            ):
                mode_state = self._mode_controller.trigger(
                    MODE_OVERTAKE,
                    now_sec=self._now_sec(),
                    reason="in_road_lidar_confirmed",
                )
                self._publish_mode(mode_state)
            lidar = points_to_grid(
                filtered_points, radius_cells=self._lidar_radius
            )
            lidar_points = int(filtered_points.shape[0])
        else:
            # No trustworthy road reference yet: keep the model channel clean
            # for this first frame and never fabricate OVERTAKE evidence.
            self._overtake_direction_latch.observe(None)
            self._overtake_cnn_latch.observe(False)
            road_diag["reference"] = "unavailable"
            lidar_preprocess = "outside_white_removed_no_reference"
            lidar = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
            lidar_points = 0

        values[2] = lidar
        lidar_cells = occupied_cell_count(lidar)
        preprocessed_at = time.perf_counter()
        self._publish_cnn_input_bev(values, message.header)
        debug_published_at = time.perf_counter()

        active_model = self._path_models[mode_state.mode]
        try:
            bundle = active_model.predict(values)
        except Exception as exc:
            self.get_logger().error(
                f"CNN inference failed mode={mode_state.mode}: {exc}"
            )
            self._publish_empty(message.header, "inference_error")
            return
        inferred_at = time.perf_counter()

        selected_path = bundle.main
        if mode_state.mode != MODE_CONE and selected_path.usable:
            self._last_road_reference = (
                selected_path.x.copy(),
                selected_path.y.copy(),
            )

        empty_other = self._invalid_path("inactive_mode_model")
        diagnostic_main = (
            empty_other if mode_state.mode == MODE_SHORTCUT else selected_path
        )
        diagnostic_shortcut = (
            selected_path if mode_state.mode == MODE_SHORTCUT else empty_other
        )
        self._main_pub.publish(path_message(diagnostic_main, message.header))
        self._shortcut_pub.publish(path_message(diagnostic_shortcut, message.header))
        published_selected = self._publish_center(selected_path, message.header)
        published_at = time.perf_counter()
        elapsed_ms = (published_at - started) * 1000.0
        payload = {
            "ok": bool(published_selected.usable),
            "cnn_mode": mode_state.mode,
            "mode_remaining_sec": round(mode_state.remaining_sec, 2),
            "mode_duration_sec": self._mode_hold_seconds[mode_state.mode],
            "mode_reason": mode_state.reason,
            "model_kind": active_model.model_kind,
            "selected_route": "shortcut" if mode_state.mode == MODE_SHORTCUT else "main",
            "selected_usable": published_selected.usable,
            "selected_reason": published_selected.reason,
            "selected_points": published_selected.point_count,
            "left_streak": self._mission.left_streak,
            "last_signal_sequence": self._mission.last_sequence,
            "traffic_stop": bool(
                self._mission.traffic_stop or self._race.finished
            ),
            "camera_cells": camera_cells,
            "lidar_preprocess_mode": lidar_preprocess,
            "road_lidar": road_diag,
            "cone_trigger_raw": bool(self._cone_trigger_high),
            "start_r_detected": bool(self._start_r_detected),
            "overtake_strategy": self._overtake_strategy,
            "overtake_block_request_topic": self._overtake_block_request_topic,
            "obstacle_lane_deadband_m": self._obstacle_lane_deadband,
            "block_request_event_id": (
                block_request.event_id if block_request is not None else None
            ),
            **self._overtake_latch_diagnostics(),
            **self._race_diagnostics(),
            "raw_lidar_points": int(raw_lidar_points.shape[0]),
            "raw_lidar_cells": raw_lidar_cells,
            "filtered_lidar_points": lidar_points,
            "used_lidar_points": lidar_points,
            "lidar_cells": lidar_cells,
            "cnn_passes": 1,
            "scan_delta_ms": None if scan_delta_ms is None else round(scan_delta_ms, 2),
            "source_age_ms": round(max(0, source_age_ns) * 1e-6, 2),
            "bev_decode_ms": round((decoded_at - started) * 1000.0, 3),
            "lidar_preprocess_ms": round(
                (preprocessed_at - decoded_at) * 1000.0, 3
            ),
            "debug_bev_publish_ms": round(
                (debug_published_at - preprocessed_at) * 1000.0, 3
            ),
            "cnn_inference_ms": round(
                (inferred_at - debug_published_at) * 1000.0, 3
            ),
            "path_publish_ms": round((published_at - inferred_at) * 1000.0, 3),
            "cnn_total_ms": round(elapsed_ms, 3),
        }
        self._diag_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = CnnPathNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        # ros2 launch stops child nodes with SIGINT.  Treat that expected
        # shutdown as a clean exit instead of emitting a misleading traceback.
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
