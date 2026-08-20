#!/usr/bin/env python3
"""Write a headless camera + BEV + dual-path replay preview MP4."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .path_contract import (
    GRID_RESOLUTION_M,
    GRID_X_BOUNDS_M,
    GRID_Y_BOUNDS_M,
)
from .replay_source_node import SENSOR_QOS_LATEST


def _stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def _decode_camera(message: Image) -> np.ndarray:
    encoding = message.encoding.casefold()
    if encoding not in {"rgb8", "bgr8", "8uc3"}:
        raise ValueError(f"unsupported camera encoding: {message.encoding}")
    if message.height <= 0 or message.width <= 0 or message.step < message.width * 3:
        raise ValueError("invalid camera geometry")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    required = int(message.height) * int(message.step)
    if raw.size < required:
        raise ValueError("camera data is truncated")
    rows = raw[:required].reshape(int(message.height), int(message.step))
    image = rows[:, : int(message.width) * 3].reshape(
        int(message.height), int(message.width), 3
    )
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(image).copy()


def _decode_bev(message: Image) -> np.ndarray:
    if message.encoding.casefold() != "8uc3":
        raise ValueError(f"unsupported BEV encoding: {message.encoding}")
    expected = int(message.height) * int(message.width) * 3
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size != expected or message.step != message.width * 3:
        raise ValueError("invalid BEV geometry/data")
    return raw.reshape(int(message.height), int(message.width), 3).copy()


class ReplayPreviewNode(Node):
    def __init__(self) -> None:
        super().__init__("replay_preview_node")
        p = self._parameter
        requested = Path(str(p("output_path", "/tmp/track_drive_replay_preview_gpt.mp4")))
        self._fps = float(p("video_fps", 10.0))
        self._camera_width = int(p("camera_panel_width", 960))
        self._canvas_height = int(p("canvas_height", 540))
        self._bev_scale = int(p("bev_scale", 4))
        overwrite = bool(p("overwrite", False))
        if self._fps <= 0.0 or self._camera_width <= 0 or self._canvas_height <= 0:
            raise ValueError("preview dimensions/fps must be positive")
        if self._bev_scale < 1:
            raise ValueError("bev_scale must be positive")
        self._output_path = self._available_path(requested.expanduser(), overwrite)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._camera: np.ndarray | None = None
        self._bev: np.ndarray | None = None
        self._bev_stamp = -1
        self._main = np.empty((0, 2), np.float32)
        self._shortcut = np.empty((0, 2), np.float32)
        self._selected = np.empty((0, 2), np.float32)
        self._route_intent = "unknown"
        self._diag = "waiting for CNN"
        self._writer = None
        self._written = 0
        self._started = time.monotonic()

        self.create_subscription(Image, "/image_raw", self._on_camera, SENSOR_QOS_LATEST)
        self.create_subscription(Image, "/perception/bev", self._on_bev, SENSOR_QOS_LATEST)
        self.create_subscription(PoseArray, "/cnn/path_main", self._on_main, 10)
        self.create_subscription(PoseArray, "/cnn/path_shortcut", self._on_shortcut, 10)
        self.create_subscription(PoseArray, "/center_path", self._on_selected, 10)
        self.create_subscription(String, "/route_intent", self._on_route_intent, 10)
        self.create_subscription(String, "/debug/cnn_path", self._on_diag, 10)
        self._timer = self.create_timer(1.0 / self._fps, self._write_frame)
        self.get_logger().info(f"Replay preview: {self._output_path}")

    def _parameter(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    @staticmethod
    def _available_path(requested: Path, overwrite: bool) -> Path:
        if overwrite or not requested.exists():
            return requested
        return requested.with_name(
            requested.stem + time.strftime("_%Y%m%d_%H%M%S") + requested.suffix
        )

    def _on_camera(self, message: Image) -> None:
        try:
            image = _decode_camera(message)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return
        with self._lock:
            self._camera = image

    def _on_bev(self, message: Image) -> None:
        try:
            bev = _decode_bev(message)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return
        stamp = _stamp_ns(message.header)
        with self._lock:
            self._bev = bev
            if stamp != self._bev_stamp:
                self._main = np.empty((0, 2), np.float32)
                self._shortcut = np.empty((0, 2), np.float32)
                self._selected = np.empty((0, 2), np.float32)
            self._bev_stamp = stamp

    @staticmethod
    def _poses(message: PoseArray) -> np.ndarray:
        return np.asarray(
            [[pose.position.x, pose.position.y] for pose in message.poses],
            dtype=np.float32,
        ).reshape(-1, 2)

    def _on_main(self, message: PoseArray) -> None:
        with self._lock:
            if _stamp_ns(message.header) == self._bev_stamp:
                self._main = self._poses(message)

    def _on_shortcut(self, message: PoseArray) -> None:
        with self._lock:
            if _stamp_ns(message.header) == self._bev_stamp:
                self._shortcut = self._poses(message)

    def _on_selected(self, message: PoseArray) -> None:
        with self._lock:
            if _stamp_ns(message.header) == self._bev_stamp:
                self._selected = self._poses(message)

    def _on_route_intent(self, message: String) -> None:
        intent = str(message.data).strip().casefold() or "unknown"
        with self._lock:
            self._route_intent = intent

    def _on_diag(self, message: String) -> None:
        route_intent = None
        try:
            payload = json.loads(message.data)
            route_intent = payload.get("route_intent")
            if payload.get("ok"):
                main = payload.get("main", {})
                shortcut = payload.get("shortcut", {})
                text = (
                    f"CNN {payload.get('cnn_total_ms', '?')} ms | "
                    f"main {main.get('points', 0)} | "
                    f"shortcut {shortcut.get('points', 0)} | "
                    f"lidar {payload.get('lidar_cells', 0)} cells"
                )
            else:
                text = f"CNN blocked: {payload.get('reason', 'unknown')}"
        except (TypeError, ValueError):
            text = "CNN diagnostic parse error"
        with self._lock:
            self._diag = text
            if isinstance(route_intent, str) and route_intent.strip():
                self._route_intent = route_intent.strip().casefold()

    def _bev_panel(
        self,
        bev: np.ndarray,
        main: np.ndarray,
        shortcut: np.ndarray,
        selected: np.ndarray,
    ) -> np.ndarray:
        height, width = bev.shape[:2]
        semantic = np.zeros((height, width, 3), dtype=np.uint8)
        yellow = bev[:, :, 0] > 0
        white = bev[:, :, 1] > 0
        semantic[yellow] = (0, 220, 255)
        semantic[white] = (245, 245, 245)

        x_max = GRID_X_BOUNDS_M[1]
        y_max = GRID_Y_BOUNDS_M[1]
        for points, color, thickness in (
            (main, (40, 255, 40), 2),
            (shortcut, (255, 80, 255), 2),
            # Draw last so the path actually sent on /center_path is obvious.
            (selected, (255, 255, 0), 3),
        ):
            pixels = []
            for x, y in points:
                row = int(np.floor((x_max - float(x)) / GRID_RESOLUTION_M))
                col = int(np.floor((y_max - float(y)) / GRID_RESOLUTION_M))
                if 0 <= row < height and 0 <= col < width:
                    pixels.append((col, row))
            if len(pixels) >= 2:
                cv2.polylines(
                    semantic,
                    [np.asarray(pixels, dtype=np.int32)],
                    False,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            elif pixels:
                cv2.circle(semantic, pixels[0], 2, color, -1)
        return cv2.resize(
            semantic,
            (width * self._bev_scale, height * self._bev_scale),
            interpolation=cv2.INTER_NEAREST,
        )

    def _write_frame(self) -> None:
        with self._lock:
            if self._camera is None or self._bev is None:
                return
            camera = self._camera.copy()
            bev = self._bev.copy()
            main = self._main.copy()
            shortcut = self._shortcut.copy()
            selected = self._selected.copy()
            route_intent = self._route_intent
            diagnostic = self._diag

        camera_panel = cv2.resize(
            camera,
            (self._camera_width, self._canvas_height),
            interpolation=cv2.INTER_AREA,
        )
        bev_image = self._bev_panel(bev, main, shortcut, selected)
        right_width = bev_image.shape[1]
        canvas = np.full(
            (self._canvas_height, self._camera_width + right_width, 3),
            24,
            dtype=np.uint8,
        )
        canvas[:, : self._camera_width] = camera_panel
        top = max(0, (self._canvas_height - bev_image.shape[0]) // 2)
        canvas[top : top + bev_image.shape[0], self._camera_width :] = bev_image
        cv2.putText(
            canvas,
            diagnostic[:95],
            (14, self._canvas_height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 255, 80),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "BEV: main=green | shortcut=magenta | selected=cyan",
            (self._camera_width + 8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"route={route_intent} | selected={len(selected)} pts",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if self._writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self._output_path), fourcc, self._fps, (canvas.shape[1], canvas.shape[0])
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"cannot open preview video: {self._output_path}")
        self._writer.write(canvas)
        self._written += 1

    def destroy_node(self) -> bool:
        if self._writer is not None:
            self._writer.release()
        elapsed = max(1e-9, time.monotonic() - self._started)
        self.get_logger().info(
            f"Replay preview complete frames={self._written} "
            f"effective_fps={self._written / elapsed:.2f} path={self._output_path}"
        )
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = ReplayPreviewNode()
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
