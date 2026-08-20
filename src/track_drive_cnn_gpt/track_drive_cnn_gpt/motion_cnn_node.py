"""CNN-specific overlay on the vehicle's latest legacy ``MotionNode``.

This module intentionally subclasses the installed vehicle controller instead
of copying or replacing it.  Vehicle-specific steering calibration and motor
conversion therefore remain owned by ``track_drive`` while CNN path semantics
are isolated here.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np

try:
    import rclpy
    from rclpy.time import Time
    from track_drive.motion_node import MotionNode as VehicleMotionNode
except ImportError:  # Allow ROS-free unit tests to import helper modules.
    rclpy = None
    Time = None
    VehicleMotionNode = object

from .motion_path_adapter import (
    prepare_ego_relative_path,
    stamp_to_ns,
    validate_source_stamp_ns,
)


# Read-only audit of xytron@192.168.50.1 on 2026-08-19.  A mismatch does not
# overwrite anything; it stops this overlay before the car can move.
AUDITED_VEHICLE_MOTION_SHA256 = (
    "1f2d3e6c2ea58073f0c6d917d587109d4055f4f6ec20153fb511c94d6cad089e"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CnnMotionNode(VehicleMotionNode):
    """Use source-stamped, fixed-x CNN paths with the vehicle controller."""

    def __init__(self) -> None:
        if rclpy is None or Time is None:
            raise RuntimeError("ROS 2 and the installed track_drive package are required")
        super().__init__()

        self.declare_parameter(
            "expected_vehicle_motion_sha256", AUDITED_VEHICLE_MOTION_SHA256
        )
        self.declare_parameter("cnn_path_frame_id", "lidar_frame")
        self.declare_parameter("path_future_tolerance_sec", 0.05)
        self.declare_parameter("verify_vehicle_motion_source", True)

        self._cnn_path_frame_id = str(
            self.get_parameter("cnn_path_frame_id").value
        )
        self._path_future_tolerance_sec = float(
            self.get_parameter("path_future_tolerance_sec").value
        )
        self._cnn_last_source_ns = -1
        self._verify_vehicle_motion_source()
        self.get_logger().info(
            "CNN motion overlay active: ego-relative absolute-x horizon "
            f"{self.LOOKAHEAD_X_MIN:.2f}..{self.LOOKAHEAD_X_MAX:.2f} m"
        )

    def _verify_vehicle_motion_source(self) -> None:
        if not bool(self.get_parameter("verify_vehicle_motion_source").value):
            self.get_logger().warning(
                "vehicle motion source verification disabled; use only for offline work"
            )
            return
        expected = str(
            self.get_parameter("expected_vehicle_motion_sha256").value
        ).strip().lower()
        source = inspect.getsourcefile(VehicleMotionNode)
        if source is None:
            raise RuntimeError("cannot locate installed vehicle MotionNode source")
        actual = _sha256_file(Path(source).resolve())
        if actual != expected:
            raise RuntimeError(
                "installed vehicle motion source changed; refusing unreviewed overlay: "
                f"expected={expected}, actual={actual}, source={source}"
            )

    def _clear_cnn_path(self, reason: str) -> None:
        self._path_xs = None
        self._path_ys = None
        self._path_stamp = None
        self.get_logger().warning(
            f"CNN path cleared: {reason}", throttle_duration_sec=1.0
        )

    def _on_path(self, msg) -> None:
        """Accept only fresh, source-stamped, strict ego-relative paths."""

        if str(msg.header.frame_id) != self._cnn_path_frame_id:
            self._clear_cnn_path(
                f"frame {msg.header.frame_id!r} != {self._cnn_path_frame_id!r}"
            )
            return
        try:
            source_ns = stamp_to_ns(msg.header.stamp)
        except ValueError as exc:
            self._clear_cnn_path(str(exc))
            return

        # An older delayed DDS sample must not erase a newer valid path.  Equal
        # stamps are accepted so an explicit empty invalidation (for example a
        # route reset) can clear the currently cached path without pretending
        # that the old camera observation is new.
        if source_ns < self._cnn_last_source_ns:
            self.get_logger().warning(
                "discarding out-of-order CNN path", throttle_duration_sec=1.0
            )
            return

        now = self.get_clock().now()
        try:
            validate_source_stamp_ns(
                source_ns,
                now.nanoseconds,
                stale_sec=self.PATH_STALE_SEC,
                future_tolerance_sec=self._path_future_tolerance_sec,
            )
            path = prepare_ego_relative_path(
                [pose.position.x for pose in msg.poses],
                [pose.position.y for pose in msg.poses],
                x_min=self.LOOKAHEAD_X_MIN,
                x_max=self.LOOKAHEAD_X_MAX,
            )
        except ValueError as exc:
            self._cnn_last_source_ns = max(self._cnn_last_source_ns, source_ns)
            self._clear_cnn_path(str(exc))
            return

        self._path_xs = path.x
        self._path_ys = path.y
        self._path_stamp = Time(
            nanoseconds=source_ns,
            clock_type=now.clock_type,
        )
        self._cnn_last_source_ns = source_ns

    def _reanchored_segment(self, xs, ys):
        """Return the fixed-x ego path; third vector is absolute forward x."""

        try:
            path = prepare_ego_relative_path(
                xs,
                ys,
                x_min=self.LOOKAHEAD_X_MIN,
                x_max=self.LOOKAHEAD_X_MAX,
            )
        except ValueError:
            return (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
        return path.x, path.y, path.focus_x


def main(args=None) -> None:
    if rclpy is None:
        raise RuntimeError("ROS 2 Python packages are required to run cnn_motion")
    rclpy.init(args=args)
    node = None
    try:
        node = CnnMotionNode()
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
