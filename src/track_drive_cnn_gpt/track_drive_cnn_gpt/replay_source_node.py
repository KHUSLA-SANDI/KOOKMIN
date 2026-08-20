#!/usr/bin/env python3
"""Replay recorded images as a live camera and optionally publish a clean scan.

Both publishers use sensor-data QoS with a one-message history.  Before they
publish, they wait for ROS graph discovery and fail if another publisher owns
the requested topic.  This prevents accidentally running ``usb_cam`` and the
replay source (or a real LiDAR and the clean scan) at the same time.
"""

from __future__ import annotations

from array import array
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String

from .replay_io import (
    FrameRecord,
    ScanRecord,
    load_frame_records_with_stats,
    load_scan_records_with_stats,
)


SENSOR_QOS_LATEST = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def _other_publishers(node: Node, topic: str) -> list[str]:
    """Describe publishers on ``topic`` other than this node itself."""

    own_name = node.get_name()
    own_namespace = node.get_namespace()
    others = []
    for endpoint in node.get_publishers_info_by_topic(topic):
        if endpoint.node_name == own_name and endpoint.node_namespace == own_namespace:
            continue
        others.append(f"{endpoint.node_namespace.rstrip('/')}/{endpoint.node_name}")
    return others


class ReplayImageNode(Node):
    """Publish a recorded camera/LiDAR session on a fresh replay timeline."""

    def __init__(self) -> None:
        super().__init__("replay_image_node")
        p = self._parameter
        self._source_path = Path(str(p("source_path", ""))).expanduser()
        self._image_topic = str(p("image_topic", "/image_raw"))
        self._frame_id = str(p("frame_id", "camera_optical_frame"))
        self._fps = float(p("fps", 30.0))
        self._playback_rate = float(p("playback_rate", 1.0))
        self._loop = bool(p("loop", True))
        self._start_index = int(p("start_index", 0))
        self._max_frames = int(p("max_frames", 0))
        self._expected_width = int(p("expected_width", 1920))
        self._expected_height = int(p("expected_height", 1080))
        self._guard_sec = float(p("startup_guard_sec", 1.5))
        self._fail_on_other = bool(p("fail_if_other_publisher", True))
        self._status_period = int(p("status_period_frames", 30))
        self._exit_when_done = bool(p("exit_when_done", False))
        requested_scan_mode = str(p("scan_mode", "auto")).strip().casefold()
        recorded_scan_path = str(p("recorded_scan_path", "")).strip()
        self._scan_topic = str(p("scan_topic", "/scan"))
        self._lidar_frame_id = str(p("lidar_frame_id", "lidar_frame"))
        self._clean_scan_rate = float(p("clean_scan_rate_hz", 10.0))
        self._clean_scan_samples = int(p("clean_scan_sample_count", 720))
        self._clean_range_min = float(p("clean_scan_range_min", 0.05))
        self._clean_range_max = float(p("clean_scan_range_max", 8.0))

        if self._fps <= 0.0 or self._playback_rate <= 0.0:
            raise ValueError("fps and playback_rate must be positive")
        if self._start_index < 0 or self._max_frames < 0:
            raise ValueError("start_index and max_frames must be non-negative")
        if self._expected_width <= 0 or self._expected_height <= 0:
            raise ValueError("expected image dimensions must be positive")
        if self._guard_sec < 0.25:
            raise ValueError("startup_guard_sec must be at least 0.25 seconds")
        if self._status_period < 1:
            raise ValueError("status_period_frames must be positive")
        if requested_scan_mode not in {"auto", "recorded", "clean", "none"}:
            raise ValueError("scan_mode must be auto, recorded, clean, or none")
        if self._clean_scan_rate <= 0.0 or self._clean_scan_samples < 2:
            raise ValueError("clean scan rate/samples are invalid")

        self._frames, self._timing_mode, frame_stats = load_frame_records_with_stats(
            self._source_path, self._fps
        )
        self._frame_skipped_lines = frame_stats.skipped_lines
        self._frame_clamped_lines = frame_stats.clamped_wall_lines
        if self._start_index >= len(self._frames):
            raise ValueError(
                f"start_index {self._start_index} is outside {len(self._frames)} frames"
            )
        session_dir = (
            self._source_path if self._source_path.is_dir() else self._source_path.parent
        )
        scan_path = (
            Path(recorded_scan_path).expanduser()
            if recorded_scan_path
            else session_dir / "scan.jsonl"
        )
        self._scan_mode = requested_scan_mode
        if self._scan_mode == "auto":
            self._scan_mode = "recorded" if scan_path.is_file() else "clean"
        self._recorded_scans: list[ScanRecord] = []
        self._scan_skipped_lines: tuple[int, ...] = ()
        if self._scan_mode == "recorded":
            self._recorded_scans, scan_stats = load_scan_records_with_stats(scan_path)
            self._scan_skipped_lines = scan_stats.skipped_lines

        end = len(self._frames)
        if self._max_frames:
            end = min(end, self._start_index + self._max_frames)
        self._cycle_frames = self._frames[self._start_index:end]
        self._wall_origin = self._cycle_frames[0].wall_sec
        self._frame_offsets = [
            (record.wall_sec - self._wall_origin) / self._playback_rate
            for record in self._cycle_frames
        ]
        self._cycle_scans = [
            record
            for record in self._recorded_scans
            if record.wall_sec >= self._wall_origin
            and record.wall_sec <= self._cycle_frames[-1].wall_sec
        ]
        self._scan_offsets = [
            (record.wall_sec - self._wall_origin) / self._playback_rate
            for record in self._cycle_scans
        ]
        frame_gaps = [
            later - earlier
            for earlier, later in zip(self._frame_offsets, self._frame_offsets[1:])
            if later > earlier
        ]
        trailing_gap = sorted(frame_gaps)[len(frame_gaps) // 2] if frame_gaps else 1.0 / self._fps
        self._cycle_period = self._frame_offsets[-1] + trailing_gap

        self._published = 0
        self._decode_errors = 0
        self._source_frames_skipped = 0
        self._started_mono: float | None = None
        self._cycle_started_mono: float | None = None
        self._pump_timer = None
        self._exit_deadline_mono: float | None = None
        self._completed = False
        self._frame_position = 0
        self._scan_position = 0
        self._clean_scan_position = 0

        self._publisher = self.create_publisher(
            Image, self._image_topic, SENSOR_QOS_LATEST
        )
        self._scan_publisher = (
            self.create_publisher(LaserScan, self._scan_topic, SENSOR_QOS_LATEST)
            if self._scan_mode != "none"
            else None
        )
        self._status_pub = self.create_publisher(String, "/replay/status", 10)
        self._guard_timer = self.create_timer(self._guard_sec, self._start)
        self._status(
            "waiting_for_topic_guard",
            source_path=str(self._source_path),
            image_count=len(self._cycle_frames),
            image_topic=self._image_topic,
            timing_mode=self._timing_mode,
            scan_mode=self._scan_mode,
            recorded_scan_count=len(self._cycle_scans),
            malformed_scan_lines=list(self._scan_skipped_lines),
            malformed_frame_lines=list(self._frame_skipped_lines),
            clamped_frame_wall_lines=list(self._frame_clamped_lines),
        )

    def _parameter(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _status(self, event: str, **extra) -> None:
        payload = {
            "event": event,
            "published_frames": self._published,
            "decode_errors": self._decode_errors,
            "source_frames_skipped": self._source_frames_skipped,
            **extra,
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self._status_pub.publish(String(data=encoded))
        self.get_logger().info(encoded)

    def _start(self) -> None:
        self._guard_timer.cancel()
        resolved_topic = self.resolve_topic_name(self._image_topic)
        others = _other_publishers(self, resolved_topic)
        if self._fail_on_other and others:
            self._status(
                "topic_collision",
                topic=resolved_topic,
                other_publishers=others,
            )
            self.get_logger().error(
                "Refusing replay because another image publisher is active: "
                + ", ".join(others)
            )
            self._finish("topic_collision")
            return
        if self._scan_publisher is not None:
            resolved_scan = self.resolve_topic_name(self._scan_topic)
            scan_others = _other_publishers(self, resolved_scan)
            if self._fail_on_other and scan_others:
                self._status(
                    "topic_collision",
                    topic=resolved_scan,
                    other_publishers=scan_others,
                )
                self.get_logger().error(
                    "Refusing replay because another scan publisher is active: "
                    + ", ".join(scan_others)
                )
                self._finish("topic_collision")
                return
        self._started_mono = time.monotonic()
        if self._scan_skipped_lines:
            self.get_logger().warning(
                "Skipped malformed scan.jsonl lines: "
                + ", ".join(str(value) for value in self._scan_skipped_lines)
            )
        if self._frame_skipped_lines:
            self.get_logger().warning(
                "Skipped malformed/missing frames.csv lines: "
                + ", ".join(str(value) for value in self._frame_skipped_lines)
            )
        if self._frame_clamped_lines:
            self.get_logger().warning(
                "Clamped backward frame wall times at lines: "
                + ", ".join(str(value) for value in self._frame_clamped_lines)
            )
        self._reset_cycle(self._started_mono)
        # A short pump interval preserves the recorded camera/LiDAR ordering;
        # the one-message sensor QoS still prevents downstream backlog.
        self._pump_timer = self.create_timer(0.002, self._pump)
        self._status(
            "replay_started",
            fps=self._fps,
            playback_rate=self._playback_rate,
            loop=self._loop,
            start_index=self._start_index,
            timing_mode=self._timing_mode,
            scan_mode=self._scan_mode,
        )

    def _reset_cycle(self, now_mono: float) -> None:
        self._cycle_started_mono = now_mono
        self._frame_position = 0
        self._scan_position = 0
        self._clean_scan_position = 0

    def _finish(self, reason: str) -> None:
        if self._completed:
            return
        self._completed = True
        if self._pump_timer is not None:
            self._pump_timer.cancel()
        elapsed = (
            None
            if self._started_mono is None
            else max(0.0, time.monotonic() - self._started_mono)
        )
        self._status(
            "replay_completed",
            reason=reason,
            elapsed_sec=None if elapsed is None else round(elapsed, 3),
            effective_fps=(
                None
                if not elapsed
                else round(self._published / elapsed, 3)
            ),
        )
        if self._exit_when_done:
            # Do not call rclpy.shutdown() from inside a timer callback.  On
            # Humble it can wait for the global executor that is currently
            # executing this callback and deadlock.  The outer spin loop polls
            # this deadline and shuts down only after the callback has returned.
            self._exit_deadline_mono = time.monotonic() + 0.25

    @property
    def exit_requested(self) -> bool:
        return (
            self._exit_deadline_mono is not None
            and time.monotonic() >= self._exit_deadline_mono
        )

    def _pump(self) -> None:
        if self._completed:
            return
        elapsed = time.monotonic() - (self._cycle_started_mono or time.monotonic())

        # Publish every due scan first when camera and scan share a wall time.
        # Limit each pump so malformed/dense logs cannot starve image callbacks.
        if self._scan_mode == "recorded":
            emitted = 0
            while (
                self._scan_position < len(self._cycle_scans)
                and self._scan_offsets[self._scan_position] <= elapsed
                and emitted < 8
            ):
                self._publish_recorded_scan(self._cycle_scans[self._scan_position])
                self._scan_position += 1
                emitted += 1
        elif self._scan_mode == "clean":
            clean_period = 1.0 / self._clean_scan_rate
            emitted = 0
            while self._clean_scan_position * clean_period <= elapsed and emitted < 4:
                self._publish_clean_scan()
                self._clean_scan_position += 1
                emitted += 1

        # If decoding briefly falls behind, emit only the newest due camera
        # frame.  This duplicates a real depth-1 camera stream instead of
        # replaying a stale queue at maximum speed.
        due_position = self._frame_position
        while (
            due_position + 1 < len(self._cycle_frames)
            and self._frame_offsets[due_position + 1] <= elapsed
        ):
            due_position += 1
        if (
            self._frame_position < len(self._cycle_frames)
            and self._frame_offsets[self._frame_position] <= elapsed
        ):
            self._source_frames_skipped += due_position - self._frame_position
            self._frame_position = due_position + 1
            self._publish_frame(self._cycle_frames[due_position])

        if self._frame_position >= len(self._cycle_frames) and elapsed >= self._cycle_period:
            if self._loop:
                self._reset_cycle(time.monotonic())
                self._status("replay_loop_restarted")
            else:
                self._finish("end_of_sequence")

    def _publish_frame(self, record: FrameRecord) -> None:
        path = record.path
        source_index = record.source_index
        decode_started = time.perf_counter()
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            self._decode_errors += 1
            self._status("frame_decode_failed", source_index=source_index, path=str(path))
            if self._decode_errors >= min(10, len(self._cycle_frames)):
                self._finish("too_many_decode_errors")
            return
        height, width = bgr.shape[:2]
        if (width, height) != (self._expected_width, self._expected_height):
            self._status(
                "resolution_mismatch",
                source_index=source_index,
                path=str(path),
                expected=[self._expected_width, self._expected_height],
                actual=[width, height],
            )
            self._finish("resolution_mismatch")
            return

        # usb_cam is configured as mjpeg2rgb, so reproduce its rgb8 wire
        # encoding exactly.  The YOLO node then performs the same RGB->BGR
        # conversion it does for the real camera.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        build_started = time.perf_counter()
        message = Image()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._frame_id
        message.height = height
        message.width = width
        message.encoding = "rgb8"
        message.is_bigendian = 0
        message.step = width * 3
        # Assign array('B') rather than bytes.  ROS 2's generated Python
        # setter validates a bytes object element-by-element (~350 ms for one
        # 1080p frame), while the native octet array takes well below 1 ms.
        message.data = array("B", np.ascontiguousarray(rgb).tobytes())
        build_ms = (time.perf_counter() - build_started) * 1000.0
        publish_started = time.perf_counter()
        self._publisher.publish(message)
        publish_ms = (time.perf_counter() - publish_started) * 1000.0
        self._published += 1

        if self._published == 1 or self._published % self._status_period == 0:
            elapsed = max(1e-9, time.monotonic() - (self._started_mono or time.monotonic()))
            self._status(
                "replay_progress",
                source_index=source_index,
                source_name=path.name,
                effective_fps=round(self._published / elapsed, 3),
                decode_ms=round(decode_ms, 3),
                message_build_ms=round(build_ms, 3),
                publish_ms=round(publish_ms, 3),
            )

    def _publish_recorded_scan(self, record: ScanRecord) -> None:
        if self._scan_publisher is None:
            return
        message = LaserScan()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._lidar_frame_id
        message.angle_min = record.angle_min
        message.angle_increment = record.angle_increment
        message.angle_max = record.angle_min + (len(record.ranges_m) - 1) * record.angle_increment
        message.scan_time = 0.1
        message.time_increment = message.scan_time / len(record.ranges_m)
        message.range_min = record.range_min
        message.range_max = record.range_max
        message.ranges = list(record.ranges_m)
        self._scan_publisher.publish(message)

    def _publish_clean_scan(self) -> None:
        if self._scan_publisher is None:
            return
        scan_time = 1.0 / self._clean_scan_rate
        message = LaserScan()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._lidar_frame_id
        message.angle_min = -math.pi
        message.angle_max = math.pi
        message.angle_increment = (message.angle_max - message.angle_min) / (
            self._clean_scan_samples - 1
        )
        message.time_increment = scan_time / self._clean_scan_samples
        message.scan_time = scan_time
        message.range_min = self._clean_range_min
        message.range_max = self._clean_range_max
        message.ranges = [math.inf] * self._clean_scan_samples
        self._scan_publisher.publish(message)


class CleanScanNode(Node):
    """Publish synchronized all-infinite scans for clean-road replay tests."""

    def __init__(self) -> None:
        super().__init__("replay_clean_scan_node")
        p = self._parameter
        self._scan_topic = str(p("scan_topic", "/scan"))
        self._frame_id = str(p("frame_id", "lidar_frame"))
        self._rate_hz = float(p("rate_hz", 10.0))
        self._sample_count = int(p("sample_count", 720))
        self._range_min = float(p("range_min", 0.05))
        self._range_max = float(p("range_max", 8.0))
        self._guard_sec = float(p("startup_guard_sec", 1.5))
        self._fail_on_other = bool(p("fail_if_other_publisher", True))
        if self._rate_hz <= 0.0 or self._sample_count < 2:
            raise ValueError("rate_hz must be positive and sample_count >= 2")
        if not 0.0 <= self._range_min < self._range_max:
            raise ValueError("scan range limits must satisfy 0 <= min < max")

        self._publisher = self.create_publisher(
            LaserScan, self._scan_topic, SENSOR_QOS_LATEST
        )
        self._scan_timer = None
        self._guard_timer = self.create_timer(self._guard_sec, self._start)

    def _parameter(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _start(self) -> None:
        self._guard_timer.cancel()
        resolved_topic = self.resolve_topic_name(self._scan_topic)
        others = _other_publishers(self, resolved_topic)
        if self._fail_on_other and others:
            self.get_logger().error(
                "Refusing clean scan because another scan publisher is active: "
                + ", ".join(others)
            )
            if rclpy.ok():
                rclpy.shutdown()
            return
        self._scan_timer = self.create_timer(1.0 / self._rate_hz, self._publish_scan)
        self.get_logger().info(
            f"Clean LaserScan replay started topic={resolved_topic} rate={self._rate_hz}Hz"
        )

    def _publish_scan(self) -> None:
        scan_time = 1.0 / self._rate_hz
        message = LaserScan()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._frame_id
        message.angle_min = -math.pi
        message.angle_max = math.pi
        message.angle_increment = (message.angle_max - message.angle_min) / (
            self._sample_count - 1
        )
        message.time_increment = scan_time / self._sample_count
        message.scan_time = scan_time
        message.range_min = self._range_min
        message.range_max = self._range_max
        message.ranges = [math.inf] * self._sample_count
        self._publisher.publish(message)


def _spin_until_exit(
    node,
    *,
    spin_once=rclpy.spin_once,
    context_ok=rclpy.ok,
    timeout_sec: float = 0.05,
) -> bool:
    """Spin until ROS shuts down or a replay node requests process exit."""

    while context_ok():
        spin_once(node, timeout_sec=timeout_sec)
        if bool(getattr(node, "exit_requested", False)):
            return True
    return False


def _spin(node_type, args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = node_type()
        _spin_until_exit(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main_images(args=None) -> None:
    _spin(ReplayImageNode, args=args)


def main_clean_scan(args=None) -> None:
    _spin(CleanScanNode, args=args)


if __name__ == "__main__":
    main_images()
