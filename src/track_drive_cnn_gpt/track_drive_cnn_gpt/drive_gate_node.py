"""Parallel STOP/GO gate for a CNN path that goes directly to motion.

This node never selects or republishes a path.  ``cnn_path_node`` is the only
publisher of ``/center_path``.  Path messages are observed for diagnostics but
are deliberately *not* a GO/STOP condition: a transient perception dropout
must not interrupt propulsion.  Race/manual arming, traffic/emergency stop,
and the launch-level drive enable remain the gate conditions, and the legacy
``/drive_cmd`` watchdog is refreshed at 20 Hz.  Normal race arming comes from
the confirmed GREEN lifecycle on ``/race_go``; ``/manual_go`` remains an
explicit operator override in either direction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from .motion_path_adapter import stamp_to_ns, validate_source_stamp_ns


OWNER_LANE = 0.0
OWNER_SAFETY_STOP = 3.0
STEER_PROFILE_NORMAL = 0.0


def encode_drive_command(drive_allowed: bool, speed_cap: float) -> list[float]:
    """Encode the legacy track_drive eight-float ``/drive_cmd`` contract."""

    cap = float(speed_cap)
    if not drive_allowed or not math.isfinite(cap) or cap <= 0.0:
        return [
            OWNER_SAFETY_STOP,
            STEER_PROFILE_NORMAL,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    return [
        OWNER_LANE,
        STEER_PROFILE_NORMAL,
        cap,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


@dataclass(frozen=True)
class DriveGateDecision:
    drive_allowed: bool
    reason: str


class DriveGateLogic:
    """ROS-independent automatic race arming and explicit-stop gate.

    ``path_source_time`` is retained only as diagnostic state.  Missing, stale,
    future, or malformed path messages never change the drive decision.
    """

    def __init__(self, *, enable_drive: bool = False, path_stale_sec: float = 0.25):
        if not math.isfinite(float(path_stale_sec)) or float(path_stale_sec) <= 0.0:
            raise ValueError("path_stale_sec must be positive and finite")
        self.enable_drive = bool(enable_drive)
        self.path_stale_sec = float(path_stale_sec)
        # ``None`` means automatic race arming owns START/WAIT.  Once an
        # operator publishes /manual_go, that explicit True/False overrides
        # the automatic value until the node is restarted.  This keeps the
        # existing field STOP/GO commands useful without requiring one for a
        # normal race start.
        self.manual_go = False
        self.manual_override: Optional[bool] = None
        self.race_go = False
        self.emergency_stop = False
        self.traffic_stop = False
        self.path_source_time: Optional[float] = None

    def set_manual_go(self, enabled: bool) -> None:
        self.manual_go = bool(enabled)
        self.manual_override = bool(enabled)

    def set_race_go(self, enabled: bool) -> None:
        self.race_go = bool(enabled)

    def set_emergency_stop(self, active: bool) -> None:
        self.emergency_stop = bool(active)

    def set_traffic_stop(self, active: bool) -> None:
        self.traffic_stop = bool(active)

    def update_path(self, source_time: Optional[float]) -> None:
        if source_time is None or not math.isfinite(float(source_time)):
            self.path_source_time = None
        else:
            self.path_source_time = float(source_time)

    def decide(self, now: float) -> DriveGateDecision:
        if not self.enable_drive:
            return DriveGateDecision(False, "drive_disabled")
        if self.emergency_stop:
            return DriveGateDecision(False, "emergency_stop")
        if self.traffic_stop:
            return DriveGateDecision(False, "traffic_signal_stop")
        if self.manual_override is False:
            return DriveGateDecision(False, "manual_stop")
        if self.manual_override is True:
            return DriveGateDecision(True, "ok_manual_override")
        if not self.race_go:
            return DriveGateDecision(False, "waiting_for_green")
        return DriveGateDecision(True, "ok_race")


try:
    import rclpy
    from geometry_msgs.msg import PoseArray
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float32MultiArray
except ImportError:
    rclpy = None
    Node = object


if rclpy is not None:

    class CnnDriveGateNode(Node):
        def __init__(self) -> None:
            super().__init__("cnn_drive_gate_node")
            self.declare_parameter("center_path_topic", "/center_path")
            self.declare_parameter("drive_cmd_topic", "/drive_cmd")
            self.declare_parameter("manual_go_topic", "/manual_go")
            self.declare_parameter("race_go_topic", "/race_go")
            self.declare_parameter("emergency_stop_topic", "/emergency_stop")
            self.declare_parameter("traffic_stop_topic", "/traffic_stop")
            self.declare_parameter("control_hz", 20.0)
            self.declare_parameter("path_stale_sec", 0.25)
            self.declare_parameter("path_future_tolerance_sec", 0.05)
            self.declare_parameter("path_frame_id", "lidar_frame")
            self.declare_parameter("enable_drive", False)
            self.declare_parameter("initial_manual_go", False)
            self.declare_parameter("speed_cap", 6.0)

            control_hz = float(self.get_parameter("control_hz").value)
            speed_cap = float(self.get_parameter("speed_cap").value)
            if not math.isfinite(control_hz) or control_hz <= 0.0:
                raise ValueError("control_hz must be positive and finite")
            if not math.isfinite(speed_cap) or speed_cap <= 0.0:
                raise ValueError("speed_cap must be positive and finite")
            self._speed_cap = speed_cap
            self._future_tolerance_sec = float(
                self.get_parameter("path_future_tolerance_sec").value
            )
            self._path_frame_id = str(self.get_parameter("path_frame_id").value)
            self._logic = DriveGateLogic(
                enable_drive=bool(self.get_parameter("enable_drive").value),
                path_stale_sec=float(self.get_parameter("path_stale_sec").value),
            )
            # false means "no manual override", so automatic GREEN arming can
            # take ownership.  true remains available for explicit dry-runs.
            if bool(self.get_parameter("initial_manual_go").value):
                self._logic.set_manual_go(True)
            self._last_source_ns = -1
            self._last_status = None

            self.create_subscription(
                PoseArray,
                str(self.get_parameter("center_path_topic").value),
                self._on_path,
                10,
            )
            self.create_subscription(
                Bool,
                str(self.get_parameter("manual_go_topic").value),
                lambda msg: self._logic.set_manual_go(msg.data),
                10,
            )
            self.create_subscription(
                Bool,
                str(self.get_parameter("race_go_topic").value),
                lambda msg: self._logic.set_race_go(msg.data),
                10,
            )
            self.create_subscription(
                Bool,
                str(self.get_parameter("emergency_stop_topic").value),
                lambda msg: self._logic.set_emergency_stop(msg.data),
                10,
            )
            self.create_subscription(
                Bool,
                str(self.get_parameter("traffic_stop_topic").value),
                lambda msg: self._logic.set_traffic_stop(msg.data),
                10,
            )
            self._drive_pub = self.create_publisher(
                Float32MultiArray,
                str(self.get_parameter("drive_cmd_topic").value),
                10,
            )
            self.create_timer(1.0 / control_hz, self._tick)
            self.get_logger().info(
                "CNN drive gate ready: observes /center_path only; "
                "does not select or republish paths; initial_manual_go=%s "
                "automatic GREEN arming enabled"
                % str(self._logic.manual_override is True).lower()
            )

        @staticmethod
        def _path_valid(message: PoseArray) -> bool:
            if len(message.poses) < 2:
                return False
            previous_x = None
            for pose in message.poses:
                x = float(pose.position.x)
                y = float(pose.position.y)
                if not (math.isfinite(x) and math.isfinite(y)):
                    return False
                if previous_x is not None and x <= previous_x:
                    return False
                previous_x = x
            return True

        def _invalidate(self, reason: str) -> None:
            self.get_logger().warning(
                f"drive gate ignoring invalid path: {reason}",
                throttle_duration_sec=1.0,
            )

        def _on_path(self, message: PoseArray) -> None:
            if str(message.header.frame_id) != self._path_frame_id:
                self._invalidate("wrong frame")
                return
            try:
                source_ns = stamp_to_ns(message.header.stamp)
            except ValueError as exc:
                self._invalidate(str(exc))
                return
            if source_ns < self._last_source_ns:
                return
            self._last_source_ns = source_ns
            try:
                validate_source_stamp_ns(
                    source_ns,
                    self.get_clock().now().nanoseconds,
                    stale_sec=self._logic.path_stale_sec,
                    future_tolerance_sec=self._future_tolerance_sec,
                )
            except ValueError as exc:
                self._invalidate(str(exc))
                return
            if not self._path_valid(message):
                self._invalidate("empty or malformed selected path")
                return
            self._logic.update_path(source_ns * 1e-9)

        def _tick(self) -> None:
            now = self.get_clock().now().nanoseconds * 1e-9
            decision = self._logic.decide(now)
            command = Float32MultiArray()
            command.data = encode_drive_command(
                decision.drive_allowed,
                self._speed_cap,
            )
            self._drive_pub.publish(command)
            status = (decision.drive_allowed, decision.reason)
            if status != self._last_status:
                # rclpy keys a log callsite by source location and rejects a
                # later call from that same line with a different severity.
                # Keep INFO and WARN on distinct callsites.
                if decision.drive_allowed:
                    self.get_logger().info(
                        f"drive gate DRIVE reason={decision.reason}"
                    )
                else:
                    self.get_logger().warning(
                        f"drive gate STOP reason={decision.reason}"
                    )
                self._last_status = status


def main(args=None) -> None:
    if rclpy is None:
        raise RuntimeError("ROS 2 Python packages are required to run cnn_drive_gate")
    rclpy.init(args=args)
    node = CnnDriveGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
