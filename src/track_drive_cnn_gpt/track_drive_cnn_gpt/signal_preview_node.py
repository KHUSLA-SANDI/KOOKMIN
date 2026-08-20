"""Tiny OpenCV window for the annotated YOLO signal preview."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage


def _preview_qos() -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


class SignalPreviewNode(Node):
    def __init__(self) -> None:
        super().__init__("signal_preview_node")
        self.declare_parameter(
            "preview_topic", "/debug/yolo_signal_preview/compressed"
        )
        self.declare_parameter("window_name", "best_0817 signal check")
        self.declare_parameter(
            "save_dir", "/home/xytron/signal_label_candidates_gpt"
        )
        self._topic = str(self.get_parameter("preview_topic").value)
        self._window_name = str(self.get_parameter("window_name").value)
        self._save_dir = Path(str(self.get_parameter("save_dir").value)).expanduser()
        self.quit_requested = False
        self._latest: np.ndarray | None = None
        self._frames = 0

        if os.name != "nt" and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            raise RuntimeError(
                "No graphical display is available. Run this viewer in the "
                "vehicle desktop terminal, or inspect /perception/signals."
            )
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        self._subscription = self.create_subscription(
            CompressedImage, self._topic, self._on_preview, _preview_qos()
        )
        self.get_logger().info(
            f"signal preview ready: topic={self._topic}; q/Esc quit, s save JPEG"
        )

    def _on_preview(self, message: CompressedImage) -> None:
        encoded = np.frombuffer(message.data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning("received an invalid preview JPEG")
            return
        self._latest = frame
        self._frames += 1
        cv2.imshow(self._window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self.quit_requested = True
        elif key == ord("s"):
            self._save_current()

    def _save_current(self) -> None:
        if self._latest is None:
            return
        self._save_dir.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime("signal_%Y%m%d_%H%M%S_%f.jpg")
        destination = self._save_dir / name
        if not cv2.imwrite(str(destination), self._latest):
            self.get_logger().error(f"failed to save {destination}")
            return
        self.get_logger().info(f"saved {destination}")

    def destroy_node(self) -> bool:
        cv2.destroyAllWindows()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: SignalPreviewNode | None = None
    try:
        node = SignalPreviewNode()
        while rclpy.ok() and not node.quit_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
