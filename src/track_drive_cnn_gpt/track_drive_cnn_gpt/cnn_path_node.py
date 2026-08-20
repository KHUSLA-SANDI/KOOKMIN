#!/usr/bin/env python3
"""Combine full-topology camera BEV with LiDAR and run the path CNN.

The input BEV message contains yellow/white channels only.  This node selects
the nearest received LaserScan, obtains a camera-only reference path, compacts
one unambiguous obstacle cluster into the training LiDAR domain, and publishes
sanitized main and shortcut paths.  It never republishes an old prediction
with a new stamp.
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
from std_msgs.msg import String

from .lidar_compact import compact_obstacle_points
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
        model_path = str(p("model_path", ""))
        expected_model_sha256 = str(p("expected_model_sha256", ""))
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
        self._min_lidar_cells = int(p("min_lidar_cells", 27))
        self._max_lidar_cells = int(p("max_lidar_cells", 67))
        self._lidar_preprocess_mode = str(
            p("lidar_preprocess_mode", "compact_reference")
        ).strip().lower()
        self._lidar_fov_deg = float(p("lidar_fov_deg", 110.0))
        self._lidar_corridor_half_width = float(
            p("lidar_corridor_half_width_m", 0.70)
        )
        self._lidar_cluster_gap_base = float(
            p("lidar_cluster_gap_base_m", 0.12)
        )
        self._lidar_cluster_gap_per_m = float(
            p("lidar_cluster_gap_per_m", 0.03)
        )
        self._lidar_min_cluster_points = int(p("lidar_min_cluster_points", 3))
        self._lidar_max_cluster_extent = float(
            p("lidar_max_cluster_extent_m", 0.72)
        )
        self._lidar_max_clusters = int(p("lidar_max_clusters", 1))
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
        if self._min_lidar_cells < 0 or self._max_lidar_cells < self._min_lidar_cells:
            raise ValueError("LiDAR cell limits must satisfy 0 <= min <= max")
        if self._lidar_preprocess_mode not in ("raw_gate", "compact_reference"):
            raise ValueError(
                "lidar_preprocess_mode must be raw_gate or compact_reference"
            )
        if not 0.0 < self._lidar_fov_deg <= 180.0:
            raise ValueError("lidar_fov_deg must be in (0, 180]")
        if not np.isfinite(
            [
                self._lidar_fov_deg,
                self._lidar_corridor_half_width,
                self._lidar_cluster_gap_base,
                self._lidar_cluster_gap_per_m,
                self._lidar_max_cluster_extent,
            ]
        ).all():
            raise ValueError("LiDAR compact geometry limits must be finite")
        if min(
            self._lidar_corridor_half_width,
            self._lidar_cluster_gap_base,
            self._lidar_max_cluster_extent,
        ) <= 0.0:
            raise ValueError("LiDAR compact geometry limits must be positive")
        if self._lidar_cluster_gap_per_m < 0.0:
            raise ValueError("lidar_cluster_gap_per_m must be non-negative")
        if self._lidar_min_cluster_points < 1 or self._lidar_max_clusters < 1:
            raise ValueError("LiDAR compact count limits must be positive")

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
        self._path_model = load_path_model(
            model_path,
            device=device,
            guard=guard,
            expected_sha256=expected_model_sha256,
        )
        if self._path_model.model_kind == MODEL_KIND_LEGACY and not allow_unverified_legacy:
            raise RuntimeError(
                "legacy checkpoint execution is disabled. Retrain/save a canonical "
                "dual checkpoint or set allow_unverified_legacy=true only for an "
                "offline comparison."
            )

        # (receive time, scan-ordered calibrated points, raw raster cell count)
        self._scan_history: Deque[Tuple[int, np.ndarray, int]] = deque()
        self._last_bev_stamp_ns = -1
        self._last_scan_error_log_ns = -1
        self._main_pub = self.create_publisher(PoseArray, main_topic, 1)
        self._shortcut_pub = self.create_publisher(PoseArray, shortcut_topic, 1)
        self._diag_pub = self.create_publisher(String, "/debug/cnn_path", 10)
        self.create_subscription(
            LaserScan,
            self._scan_topic,
            self._on_scan,
            qos_profile_sensor_data,
        )
        self.create_subscription(Image, self._bev_topic, self._on_bev, LATEST_QOS)
        self.get_logger().info(
            f"CNN path ready kind={self._path_model.model_kind} "
            f"input={INPUT_SHAPE} device={device} "
            f"lidar_mode={self._lidar_preprocess_mode}"
        )

    def _parameter(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

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

    def _publish_empty(self, header, reason: str, **extra) -> None:
        empty = self._invalid_path(reason)
        self._main_pub.publish(path_message(empty, header))
        self._shortcut_pub.publish(path_message(empty, header))
        payload = {"ok": False, "reason": reason, **extra}
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

        compact_diag = None
        cnn_passes = 1
        if self._lidar_preprocess_mode == "raw_gate":
            lidar = points_to_grid(
                raw_lidar_points, radius_cells=self._lidar_radius
            )
            lidar_points = int(raw_lidar_points.shape[0])
            values[2] = lidar
        else:
            # Pass 1 deliberately uses the all-zero clean channel found in
            # 8,022 training samples. Its main path defines a curve-aware road
            # corridor, so fixed |y| filtering is not used on bends.
            try:
                clean_bundle = self._path_model.predict(values)
            except Exception as exc:
                self.get_logger().error(f"CNN clean-reference inference failed: {exc}")
                self._publish_empty(message.header, "inference_error_clean_reference")
                return
            if not clean_bundle.main.usable:
                self._publish_empty(
                    message.header,
                    "camera_only_reference_unusable",
                    reference_reason=clean_bundle.main.reason,
                    raw_lidar_points=int(raw_lidar_points.shape[0]),
                    raw_lidar_cells=raw_lidar_cells,
                )
                return
            try:
                compact = compact_obstacle_points(
                    raw_lidar_points,
                    clean_bundle.main.x,
                    clean_bundle.main.y,
                    fov_deg=self._lidar_fov_deg,
                    corridor_half_width_m=self._lidar_corridor_half_width,
                    cluster_gap_base_m=self._lidar_cluster_gap_base,
                    cluster_gap_per_m=self._lidar_cluster_gap_per_m,
                    min_cluster_points=self._lidar_min_cluster_points,
                    max_cluster_extent_m=self._lidar_max_cluster_extent,
                    max_clusters=self._lidar_max_clusters,
                )
            except ValueError as exc:
                self._publish_empty(
                    message.header,
                    "lidar_preprocess_error",
                    detail=str(exc),
                    raw_lidar_points=int(raw_lidar_points.shape[0]),
                    raw_lidar_cells=raw_lidar_cells,
                )
                return
            compact_diag = {
                "reason": compact.reason,
                "raw_fov_points": compact.raw_fov_points,
                "corridor_points": compact.corridor_points,
                "corridor_clusters": compact.corridor_clusters,
                "selected_raw_points": compact.selected_raw_points,
                "selected_extent_m": round(compact.selected_extent_m, 3),
            }
            if not compact.usable:
                self._publish_empty(
                    message.header,
                    f"lidar_preprocess_ambiguous:{compact.reason}",
                    lidar_preprocess=compact_diag,
                    raw_lidar_points=int(raw_lidar_points.shape[0]),
                    raw_lidar_cells=raw_lidar_cells,
                    scan_delta_ms=(
                        None if scan_delta_ms is None else round(scan_delta_ms, 2)
                    ),
                )
                return
            lidar = points_to_grid(compact.points, radius_cells=self._lidar_radius)
            lidar_points = int(compact.points.shape[0])
            if compact.obstacle_present:
                values[2] = lidar
            else:
                bundle = clean_bundle

        lidar_cells = occupied_cell_count(lidar)
        if lidar_cells != 0 and not (
            self._min_lidar_cells <= lidar_cells <= self._max_lidar_cells
        ):
            self._publish_empty(
                message.header,
                "lidar_occupancy_out_of_training_domain",
                lidar_cells=lidar_cells,
                min_lidar_cells=self._min_lidar_cells,
                max_lidar_cells=self._max_lidar_cells,
                lidar_points=lidar_points,
                raw_lidar_points=int(raw_lidar_points.shape[0]),
                raw_lidar_cells=raw_lidar_cells,
                lidar_preprocess=compact_diag,
                scan_delta_ms=(
                    None if scan_delta_ms is None else round(scan_delta_ms, 2)
                ),
            )
            return

        if self._lidar_preprocess_mode == "raw_gate":
            try:
                bundle = self._path_model.predict(values)
            except Exception as exc:
                self.get_logger().error(f"CNN inference failed: {exc}")
                self._publish_empty(message.header, "inference_error")
                return
        elif lidar_points:
            try:
                bundle = self._path_model.predict(values)
            except Exception as exc:
                self.get_logger().error(f"CNN obstacle inference failed: {exc}")
                self._publish_empty(message.header, "inference_error_obstacle")
                return
            cnn_passes = 2

        # Availability is part of the canonical shortcut contract, but the ROS
        # interface carries paths only.  Publish an empty shortcut when the
        # availability gate is closed so the supervisor cannot select it.
        published_shortcut = (
            bundle.shortcut
            if bundle.shortcut_available
            else self._invalid_path("shortcut_unavailable")
        )
        self._main_pub.publish(path_message(bundle.main, message.header))
        self._shortcut_pub.publish(path_message(published_shortcut, message.header))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        payload = {
            "ok": bool(bundle.main.usable),
            "model_kind": self._path_model.model_kind,
            "main": {
                "usable": bundle.main.usable,
                "reason": bundle.main.reason,
                "points": bundle.main.point_count,
                "span_m": round(bundle.main.span_m, 3),
            },
            "shortcut": {
                "usable": bundle.shortcut_available,
                "reason": published_shortcut.reason,
                "points": published_shortcut.point_count,
                "probability": (
                    round(bundle.shortcut_probability, 4)
                    if bundle.shortcut_availability_source == "availability_head"
                    else None
                ),
                "availability_source": bundle.shortcut_availability_source,
            },
            "camera_cells": camera_cells,
            "lidar_preprocess_mode": self._lidar_preprocess_mode,
            "lidar_preprocess": compact_diag,
            "raw_lidar_points": int(raw_lidar_points.shape[0]),
            "raw_lidar_cells": raw_lidar_cells,
            "lidar_points": lidar_points,
            "lidar_cells": lidar_cells,
            "cnn_passes": cnn_passes,
            "scan_delta_ms": None if scan_delta_ms is None else round(scan_delta_ms, 2),
            "source_age_ms": round(max(0, source_age_ns) * 1e-6, 2),
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
