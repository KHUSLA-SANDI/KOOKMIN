#!/usr/bin/env python3
"""Minimal low-speed path follower for the CNN ``/center_path``.

Each GENERAL/SHORTCUT/OVERTAKE/CONE profile contains a fixed lookahead, one steering gain, one
exponential smoothing factor, and its requested speed.  Mission-specific simulation
logic (straight/curve switching, S-zone focus, boosts, preview speed, block
steering, and re-anchoring) is deliberately absent.

The node still keeps the safety and vehicle contracts that are not tuning
features: source-stamp freshness, the 20 Hz ``/drive_cmd`` watchdog, fail-stop
behaviour, and the installed vehicle ``CarInterface`` calibration/slew layer.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .cnn_modes import MODE_CONE, MODE_GENERAL, MODE_OVERTAKE, MODE_SHORTCUT, MODES
from .motion_path_adapter import (
    prepare_ego_relative_path,
    stamp_to_ns,
    validate_source_stamp_ns,
)


ANGLE_MIN = -100.0
ANGLE_MAX = 100.0
DEBUG_HZ = 2.0
AUDITED_CAR_INTERFACE_SHA256 = (
    "6554c716957cc93eb77b626bca3554653708b130640a6860da55d8811b889091"
)


def simple_steering_command(
    x_values: Any,
    y_values: Any,
    *,
    lookahead_m: float,
    steer_gain: float,
) -> tuple[float, float, float]:
    """Return ``(logical_angle, target_x, target_y)`` for one path.

    The lookahead is clamped to the valid path span instead of extrapolating a
    short prediction.  ``y`` is left-positive while the vehicle's historical
    logical steering convention is right-positive, hence the minus sign.
    """

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError("simple motion path must contain matching x/y vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("simple motion path contains NaN or infinity")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("simple motion path x must be strictly increasing")

    lookahead = float(lookahead_m)
    gain = float(steer_gain)
    if not math.isfinite(lookahead) or lookahead <= 0.0:
        raise ValueError("lookahead_m must be positive and finite")
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("steer_gain must be positive and finite")

    target_x = float(np.clip(lookahead, x[0], x[-1]))
    target_y = float(np.interp(target_x, x, y))
    bearing_deg = math.degrees(math.atan2(-target_y, max(target_x, 1e-6)))
    logical_angle = float(np.clip(gain * bearing_deg, ANGLE_MIN, ANGLE_MAX))
    return logical_angle, target_x, target_y


def smooth_steering(previous: float, target: float, alpha: float) -> float:
    """Apply one minimal IIR smoothing step in logical steering units."""

    previous = float(previous)
    target = float(target)
    alpha = float(alpha)
    if not math.isfinite(previous) or not math.isfinite(target):
        raise ValueError("steering values must be finite")
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("steer_smooth_alpha must be within (0, 1]")
    return float(np.clip(previous + alpha * (target - previous), ANGLE_MIN, ANGLE_MAX))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


try:
    import rclpy
    from geometry_msgs.msg import PoseArray
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, String
    from track_drive.lib import drive_cmd as dc
    from track_drive.lib.car_interface import CarInterface, DEFAULT_CFG
except ImportError:  # Pure steering helpers remain importable without ROS.
    rclpy = None
    Node = object
    PoseArray = Float32MultiArray = String = object
    dc = None
    CarInterface = None
    DEFAULT_CFG = {}


if rclpy is not None:

    class SimpleMotionNode(Node):
        """Low-speed controller with no simulation mission heuristics."""

        def __init__(self) -> None:
            # Keep the legacy node name so the audited track_drive/car.yaml
            # ``motion_node`` section supplies the exact measured calibration.
            super().__init__("motion_node")
            p = self._parameter

            self._control_hz = float(p("control_hz", 20.0))
            general = {
                "lookahead_m": float(p("lookahead_m", 1.0)),
                "steer_gain": float(p("steer_gain", 0.45)),
                "steer_smooth_alpha": float(p("steer_smooth_alpha", 0.40)),
                "speed_cmd": float(p("speed_cmd", 5.0)),
            }
            self._profiles = {
                MODE_GENERAL: general,
                MODE_SHORTCUT: {
                    "lookahead_m": float(p("shortcut_lookahead_m", general["lookahead_m"])),
                    "steer_gain": float(p("shortcut_steer_gain", general["steer_gain"])),
                    "steer_smooth_alpha": float(
                        p("shortcut_steer_smooth_alpha", general["steer_smooth_alpha"])
                    ),
                    "speed_cmd": float(p("shortcut_speed_cmd", general["speed_cmd"])),
                },
                MODE_OVERTAKE: {
                    "lookahead_m": float(p("overtake_lookahead_m", general["lookahead_m"])),
                    "steer_gain": float(p("overtake_steer_gain", general["steer_gain"])),
                    "steer_smooth_alpha": float(
                        p("overtake_steer_smooth_alpha", general["steer_smooth_alpha"])
                    ),
                    "speed_cmd": float(p("overtake_speed_cmd", general["speed_cmd"])),
                },
                MODE_CONE: {
                    "lookahead_m": float(p("cone_lookahead_m", 0.8)),
                    "steer_gain": float(p("cone_steer_gain", 0.55)),
                    "steer_smooth_alpha": float(p("cone_steer_smooth_alpha", 0.35)),
                    "speed_cmd": float(p("cone_speed_cmd", 3.5)),
                },
            }
            self._cnn_mode_topic = str(p("cnn_mode_topic", "/cnn_mode"))
            self._path_stale_sec = float(p("path_stale_sec", 0.25))
            self._future_tolerance_sec = float(
                p("path_future_tolerance_sec", 0.05)
            )
            self._path_frame_id = str(p("path_frame_id", "lidar_frame"))
            self._verify_car_source = bool(p("verify_car_interface_source", True))
            self._expected_car_sha256 = str(
                p(
                    "expected_car_interface_sha256",
                    AUDITED_CAR_INTERFACE_SHA256,
                )
            ).strip().lower()

            if not math.isfinite(self._control_hz) or self._control_hz <= 0.0:
                raise ValueError("control_hz must be positive and finite")
            for mode, profile in self._profiles.items():
                lookahead = profile["lookahead_m"]
                if not 0.3 <= lookahead <= 3.0:
                    raise ValueError(
                        f"{mode} lookahead_m must be within the CNN 0.3..3.0 m horizon"
                    )
                gain = profile["steer_gain"]
                if not math.isfinite(gain) or gain <= 0.0:
                    raise ValueError(f"{mode} steer_gain must be positive and finite")
                alpha = profile["steer_smooth_alpha"]
                if not 0.0 < alpha <= 1.0:
                    raise ValueError(f"{mode} steer_smooth_alpha must be within (0, 1]")
                speed = profile["speed_cmd"]
                if not math.isfinite(speed) or speed <= 0.0:
                    raise ValueError(f"{mode} speed_cmd must be positive and finite")
            if self._path_stale_sec <= 0.0 or self._future_tolerance_sec < 0.0:
                raise ValueError("path timestamp policy is invalid")

            self._verify_installed_car_interface()
            car_cfg = {key: p(key, DEFAULT_CFG[key]) for key in sorted(DEFAULT_CFG)}
            self._car = CarInterface(car_cfg)

            self._path_x: Optional[np.ndarray] = None
            self._path_y: Optional[np.ndarray] = None
            self._path_source_ns = -1
            self._last_seen_path_ns = -1
            self._cmd = dc.decode_drive_cmd([])
            self._cmd_receive_ns = -1
            self._angle_cmd = 0.0
            self._cnn_mode = MODE_GENERAL
            self._debug: dict[str, Any] = {
                "drive": False,
                "reason": "startup",
            }

            self.create_subscription(PoseArray, "/center_path", self._on_path, 10)
            self.create_subscription(
                Float32MultiArray, "/drive_cmd", self._on_drive_cmd, 10
            )
            self.create_subscription(
                String, self._cnn_mode_topic, self._on_cnn_mode, 10
            )
            self._motor_pub = self.create_publisher(
                Float32MultiArray, "/xycar_motor", 10
            )
            self._debug_pub = self.create_publisher(
                String, "/debug/arbitration", 10
            )
            self.create_timer(1.0 / self._control_hz, self._tick)
            self.create_timer(1.0 / DEBUG_HZ, self._tick_debug)
            summary = " ".join(
                f"{mode}=({profile['lookahead_m']:.2f}m,{profile['steer_gain']:.3f},"
                f"{profile['steer_smooth_alpha']:.2f},speed={profile['speed_cmd']:.1f})"
                for mode, profile in self._profiles.items()
            )
            self.get_logger().info(f"simple motion ready: {summary}")

        def _parameter(self, name: str, default: Any) -> Any:
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        def _verify_installed_car_interface(self) -> None:
            if not self._verify_car_source:
                self.get_logger().warning(
                    "CarInterface source verification disabled; offline use only"
                )
                return
            source = inspect.getsourcefile(CarInterface)
            if source is None:
                raise RuntimeError("cannot locate installed CarInterface source")
            actual = _sha256_file(Path(source).resolve())
            if actual != self._expected_car_sha256:
                raise RuntimeError(
                    "installed CarInterface changed; refusing unreviewed steering: "
                    f"expected={self._expected_car_sha256}, actual={actual}, "
                    f"source={source}"
                )

        def _clear_path(self, reason: str) -> None:
            self._path_x = None
            self._path_y = None
            self._path_source_ns = -1
            self.get_logger().warning(
                f"simple motion path cleared: {reason}",
                throttle_duration_sec=1.0,
            )

        def _on_path(self, message: PoseArray) -> None:
            if str(message.header.frame_id) != self._path_frame_id:
                self._clear_path("wrong frame")
                return
            try:
                source_ns = stamp_to_ns(message.header.stamp)
            except ValueError as exc:
                self._clear_path(str(exc))
                return
            if source_ns < self._last_seen_path_ns:
                return
            self._last_seen_path_ns = source_ns
            now_ns = self.get_clock().now().nanoseconds
            try:
                validate_source_stamp_ns(
                    source_ns,
                    now_ns,
                    stale_sec=self._path_stale_sec,
                    future_tolerance_sec=self._future_tolerance_sec,
                )
                path = prepare_ego_relative_path(
                    [pose.position.x for pose in message.poses],
                    [pose.position.y for pose in message.poses],
                    x_min=0.3,
                    x_max=3.0,
                )
            except ValueError as exc:
                self._clear_path(str(exc))
                return
            self._path_x = path.x
            self._path_y = path.y
            self._path_source_ns = source_ns

        def _on_drive_cmd(self, message: Float32MultiArray) -> None:
            self._cmd = dc.decode_drive_cmd(message.data)
            self._cmd_receive_ns = self.get_clock().now().nanoseconds

        def _on_cnn_mode(self, message: String) -> None:
            requested = str(message.data).strip().upper()
            if requested not in MODES:
                self.get_logger().warning(
                    f"ignoring unsupported CNN motion mode: {message.data!r}",
                    throttle_duration_sec=1.0,
                )
                return
            if requested != self._cnn_mode:
                self._cnn_mode = requested
                self.get_logger().info(f"simple motion profile: {self._cnn_mode}")

        def _fresh_path(self, now_ns: int) -> bool:
            if self._path_x is None or self._path_y is None:
                return False
            age = (now_ns - self._path_source_ns) * 1e-9
            return -self._future_tolerance_sec <= age <= self._path_stale_sec

        def _fresh_drive_cmd(self, now_ns: int) -> bool:
            if self._cmd_receive_ns <= 0:
                return False
            return (now_ns - self._cmd_receive_ns) * 1e-9 <= dc.DRIVE_CMD_STALE_SEC

        def _tick(self) -> None:
            now_ns = self.get_clock().now().nanoseconds
            path_fresh = self._fresh_path(now_ns)
            cmd_fresh = self._fresh_drive_cmd(now_ns)
            owner_lane = self._cmd.get("owner") == dc.Owner.LANE
            cmd_valid = bool(self._cmd.get("valid", False))
            speed_cap = float(self._cmd.get("speed_cap", 0.0))
            speed_cap_ok = math.isfinite(speed_cap) and speed_cap > 0.0
            drive = path_fresh and cmd_fresh and cmd_valid and owner_lane and speed_cap_ok

            profile = self._cnn_mode
            active = self._profiles[profile]
            lookahead_m = active["lookahead_m"]
            steer_gain = active["steer_gain"]
            steer_alpha = active["steer_smooth_alpha"]
            requested_speed = active["speed_cmd"]

            raw_angle = self._angle_cmd
            target_x = None
            target_y = None
            reason = "ok"
            if drive:
                try:
                    raw_angle, target_x, target_y = simple_steering_command(
                        self._path_x,
                        self._path_y,
                        lookahead_m=lookahead_m,
                        steer_gain=steer_gain,
                    )
                    self._angle_cmd = smooth_steering(
                        self._angle_cmd, raw_angle, steer_alpha
                    )
                    # /drive_cmd remains the global ceiling and STOP watchdog;
                    # the active motion profile owns the requested speed.
                    speed_cmd = min(speed_cap, requested_speed)
                except ValueError as exc:
                    drive = False
                    speed_cmd = 0.0
                    reason = f"steering_error:{exc}"
            else:
                speed_cmd = 0.0
                if not path_fresh:
                    reason = "path_missing_or_stale"
                elif not cmd_fresh:
                    reason = "drive_cmd_stale"
                elif not cmd_valid:
                    reason = "drive_cmd_invalid"
                elif not owner_lane:
                    reason = f"owner_{self._cmd.get('owner')}"
                else:
                    reason = "speed_cap_missing"

            # STOP retains the last logical steering angle.  CarInterface owns
            # the measured left/right scaling, mechanical clamps, and both
            # steering and speed slew at 20 Hz.
            angle_out, speed_out = self._car.to_motor(self._angle_cmd, speed_cmd)
            output = Float32MultiArray()
            output.data = [float(angle_out), float(speed_out)]
            self._motor_pub.publish(output)

            self._debug = {
                "controller": "simple_motion_v1_gpt",
                "profile": profile,
                "drive": drive,
                "reason": reason,
                "path_fresh": path_fresh,
                "cmd_fresh": cmd_fresh,
                "lookahead_m": lookahead_m,
                "steer_gain": steer_gain,
                "steer_smooth_alpha": steer_alpha,
                "target_x": target_x,
                "target_y": target_y,
                "raw_angle": raw_angle,
                "angle_cmd": self._angle_cmd,
                "angle_out": angle_out,
                "speed_cap": speed_cap,
                "requested_speed": requested_speed,
                "speed_out": speed_out,
            }

        def _tick_debug(self) -> None:
            message = String()
            message.data = json.dumps(self._debug, ensure_ascii=False)
            self._debug_pub.publish(message)


def main(args=None) -> None:
    if rclpy is None:
        raise RuntimeError("ROS 2 and installed track_drive are required")
    rclpy.init(args=args)
    node = None
    try:
        node = SimpleMotionNode()
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
