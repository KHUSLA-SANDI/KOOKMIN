#!/usr/bin/env python3
"""Minimal low-speed path follower for the CNN ``/center_path``.

Each GENERAL/SHORTCUT/OVERTAKE/CONE profile contains a fixed lookahead, one steering gain, one
exponential smoothing factor, and its requested speed.  Mission-specific simulation
logic (straight/curve switching, S-zone focus, boosts, preview speed, block
steering, and re-anchoring) is deliberately absent.

The node keeps publishing at 20 Hz and owns the motor-speed command: it ramps
from zero by the measured per-tick rise toward the active mode speed.  An
explicit STOP still becomes zero on the same tick.  The installed
vehicle ``CarInterface`` continues to own steering calibration/slew.  Path
validity and timestamps are used only when accepting a replacement path.  A
rejected or missing replacement never clears the last usable path and never
stops propulsion.
"""

from __future__ import annotations

from collections import deque
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .cnn_modes import MODE_CONE, MODE_GENERAL, MODE_OVERTAKE, MODE_SHORTCUT, MODES
from .hardcoded_overtake import (
    OVERTAKE_STRATEGIES,
    STRATEGY_HARDCODED_ALL,
    HardcodedOvertakeRequest,
)
from .motion_path_adapter import (
    prepare_ego_relative_path,
    stamp_to_ns,
    validate_source_stamp_ns,
)
from .overtake_block import (
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
    PHASE_IDLE,
    HardcodedOvertakeBlock,
    OvertakePulseProfile,
)


ANGLE_MIN = -100.0
ANGLE_MAX = 100.0
DEBUG_HZ = 10.0
AUDITED_CAR_INTERFACE_SHA256 = (
    "595628d4c08bed8ddc59ad7e53bd5393084256f74b80815994544f957935d9f8"
)
# ``simple_motion`` shapes speed itself.  This value disables only the legacy
# symmetric speed limiter inside this node's CarInterface instance; steering
# calibration/slew and every other CarInterface user remain unchanged.
CAR_SPEED_SLEW_BYPASS_PER_TICK = 100.0


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


def motion_drive_allowed(
    *,
    cmd_fresh: bool,
    cmd_valid: bool,
    owner_lane: bool,
    speed_cap: float,
) -> bool:
    """Return whether explicit drive authorization is currently usable.

    Path availability is intentionally absent from this decision.  During a
    perception dropout the controller continues with its cached path, or with
    the last steering command when no path has ever been accepted.
    """

    cap = float(speed_cap)
    return bool(
        cmd_fresh
        and cmd_valid
        and owner_lane
        and math.isfinite(cap)
        and cap > 0.0
    )


def select_requested_speed(
    profile_speed: float,
    *,
    cnn_mode: str,
    cone_approach_active: bool,
    cone_approach_speed: float,
) -> tuple[float, bool]:
    """Apply the START_R approach limit without changing the active CNN mode."""

    requested = float(profile_speed)
    approach = float(cone_approach_speed)
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("profile_speed must be positive and finite")
    if not math.isfinite(approach) or approach <= 0.0:
        raise ValueError("cone_approach_speed must be positive and finite")
    limited = bool(cnn_mode == MODE_GENERAL and cone_approach_active)
    return (min(requested, approach) if limited else requested), limited


class TwoStageSpeedController:
    """Shape motor speed while the node continues publishing every 20 Hz tick.

    A stopped-to-driving transition starts at ``startup_speed_cmd`` (zero is
    supported) and holds it for ``startup_hold_sec``.  It then moves
    toward the current target by the configured up/down amount per timer tick.
    Target changes while already driving never restart the startup hold.  A
    drive-disable request always returns zero immediately and rearms startup for
    the next genuine departure.
    """

    def __init__(
        self,
        *,
        startup_speed_cmd: float,
        startup_hold_sec: float,
        slew_speed_up_per_tick: float,
        slew_speed_down_per_tick: float,
    ) -> None:
        self.startup_speed_cmd = float(startup_speed_cmd)
        self.startup_hold_sec = float(startup_hold_sec)
        self.slew_speed_up_per_tick = float(slew_speed_up_per_tick)
        self.slew_speed_down_per_tick = float(slew_speed_down_per_tick)
        if not math.isfinite(self.startup_speed_cmd) or self.startup_speed_cmd < 0.0:
            raise ValueError("startup_speed_cmd must be non-negative and finite")
        if not math.isfinite(self.startup_hold_sec) or self.startup_hold_sec < 0.0:
            raise ValueError("startup_hold_sec must be non-negative and finite")
        if (
            not math.isfinite(self.slew_speed_up_per_tick)
            or self.slew_speed_up_per_tick <= 0.0
        ):
            raise ValueError("slew_speed_up_per_tick must be positive and finite")
        if (
            not math.isfinite(self.slew_speed_down_per_tick)
            or self.slew_speed_down_per_tick <= 0.0
        ):
            raise ValueError("slew_speed_down_per_tick must be positive and finite")

        self._driving = False
        self._command = 0.0
        self._startup_until_sec = 0.0
        self._phase = "STOPPED"

    @property
    def command(self) -> float:
        return self._command

    @property
    def phase(self) -> str:
        return self._phase

    def startup_remaining_sec(self, now_sec: float) -> float:
        if self._phase != "STARTUP_HOLD":
            return 0.0
        return max(0.0, self._startup_until_sec - float(now_sec))

    def update(self, *, drive: bool, target_speed: float, now_sec: float) -> float:
        now = float(now_sec)
        target = float(target_speed)
        if not math.isfinite(now):
            raise ValueError("now_sec must be finite")
        if not math.isfinite(target) or target < 0.0:
            raise ValueError("target_speed must be non-negative and finite")

        if not drive:
            self._driving = False
            self._command = 0.0
            self._startup_until_sec = now
            self._phase = "STOPPED"
            return self._command

        if not self._driving:
            self._driving = True
            self._command = min(self.startup_speed_cmd, target)
            self._startup_until_sec = now + self.startup_hold_sec
            self._phase = (
                "STARTUP_HOLD" if self.startup_hold_sec > 0.0 else "RUNNING"
            )
            return self._command

        # If a new ceiling falls below the held command, obey it using the
        # normal down-slew instead of waiting for startup hold to expire.
        if target < self._command:
            self._command -= min(
                self._command - target, self.slew_speed_down_per_tick
            )
            self._startup_until_sec = now
            self._phase = "RUNNING"
            return self._command

        if now < self._startup_until_sec:
            self._phase = "STARTUP_HOLD"
            return self._command

        delta = target - self._command
        if delta > 0.0:
            self._command += min(delta, self.slew_speed_up_per_tick)
        self._phase = "RUNNING"
        return self._command


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
    from std_msgs.msg import Bool, Float32MultiArray, String
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
            self._debug_hz = float(p("debug_hz", DEBUG_HZ))
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
            self._cone_approach_topic = str(
                p("cone_approach_topic", "/perception/cone_approach")
            )
            self._cone_approach_speed_cmd = float(
                p("cone_approach_speed_cmd", 8.0)
            )
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
            block_lane_change_speed = float(
                p("hardcoded_block_lane_change_speed_cmd", 16.0)
            )
            block_pass_speed = float(p("hardcoded_block_pass_speed_cmd", 16.0))
            block_pass_ticks = int(p("hardcoded_block_pass_ticks", 20))
            self._block_profiles = {
                DIRECTION_RIGHT: OvertakePulseProfile(
                    direction=DIRECTION_RIGHT,
                    shift_angle_cmd=float(
                        p("hardcoded_block_right_shift_angle_cmd", 60.0)
                    ),
                    shift_ticks=int(p("hardcoded_block_right_shift_ticks", 6)),
                    counter_angle_cmd=float(
                        p("hardcoded_block_right_counter_angle_cmd", -60.0)
                    ),
                    counter_ticks=int(
                        p("hardcoded_block_right_counter_ticks", 10)
                    ),
                    lane_change_speed_cmd=block_lane_change_speed,
                    pass_speed_cmd=block_pass_speed,
                    pass_ticks=block_pass_ticks,
                ),
                DIRECTION_LEFT: OvertakePulseProfile(
                    direction=DIRECTION_LEFT,
                    shift_angle_cmd=float(
                        p("hardcoded_block_left_shift_angle_cmd", -60.0)
                    ),
                    shift_ticks=int(p("hardcoded_block_left_shift_ticks", 7)),
                    counter_angle_cmd=float(
                        p("hardcoded_block_left_counter_angle_cmd", 60.0)
                    ),
                    counter_ticks=int(
                        p("hardcoded_block_left_counter_ticks", 10)
                    ),
                    lane_change_speed_cmd=block_lane_change_speed,
                    pass_speed_cmd=block_pass_speed,
                    pass_ticks=block_pass_ticks,
                ),
            }
            self._startup_speed_cmd = float(p("startup_speed_cmd", 5.0))
            self._startup_hold_sec = float(p("startup_hold_sec", 1.0))
            self._slew_speed_up_per_tick = float(
                p("slew_speed_up_per_tick", 0.75)
            )
            self._slew_speed_down_per_tick = float(
                p("slew_speed_down_per_tick", 2.0)
            )
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
            if (
                not math.isfinite(self._debug_hz)
                or self._debug_hz <= 0.0
                or self._debug_hz > self._control_hz
            ):
                raise ValueError("debug_hz must be within 0..control_hz")
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
            if (
                not math.isfinite(self._cone_approach_speed_cmd)
                or self._cone_approach_speed_cmd <= 0.0
            ):
                raise ValueError(
                    "cone_approach_speed_cmd must be positive and finite"
                )

            self._speed_controller = TwoStageSpeedController(
                startup_speed_cmd=self._startup_speed_cmd,
                startup_hold_sec=self._startup_hold_sec,
                slew_speed_up_per_tick=self._slew_speed_up_per_tick,
                slew_speed_down_per_tick=self._slew_speed_down_per_tick,
            )

            self._verify_installed_car_interface()
            car_cfg = {key: p(key, DEFAULT_CFG[key]) for key in sorted(DEFAULT_CFG)}
            # The controller above owns the measured motor ramp.  Bypass both
            # CarInterface's legacy slew and dead-zone snap for this instance:
            # sub-dead-zone commands are intentional during a smooth launch.
            car_cfg["slew_speed_per_tick"] = CAR_SPEED_SLEW_BYPASS_PER_TICK
            car_cfg["speed_deadzone"] = 0.0
            self._car = CarInterface(car_cfg)

            self._path_x: Optional[np.ndarray] = None
            self._path_y: Optional[np.ndarray] = None
            self._path_source_ns = -1
            self._last_seen_path_ns = -1
            self._cmd = dc.decode_drive_cmd([])
            self._cmd_receive_ns = -1
            self._angle_cmd = 0.0
            self._cnn_mode = MODE_GENERAL
            self._cone_approach_active = False
            self._overtake_block = HardcodedOvertakeBlock()
            self._block_pending: Optional[HardcodedOvertakeRequest] = None
            self._block_seen_ids: set[str] = set()
            self._block_seen_order: deque[str] = deque()
            self._block_event_id = ""
            self._block_direction = "NONE"
            self._block_obstacle_lane = "NONE"
            self._block_phase = PHASE_IDLE
            self._block_phase_tick = 0
            self._block_phase_ticks = 0
            self._block_total_tick = 0
            self._block_total_ticks = 0
            self._block_cancel_reason = ""
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
            self.create_subscription(
                Bool, self._cone_approach_topic, self._on_cone_approach, 10
            )
            self.create_subscription(
                String,
                self._overtake_block_request_topic,
                self._on_overtake_block_request,
                10,
            )
            self._motor_pub = self.create_publisher(
                Float32MultiArray, "/xycar_motor", 10
            )
            self._debug_pub = self.create_publisher(
                String, "/debug/arbitration", 10
            )
            self.create_timer(1.0 / self._control_hz, self._tick)
            self.create_timer(1.0 / self._debug_hz, self._tick_debug)
            summary = " ".join(
                f"{mode}=({profile['lookahead_m']:.2f}m,{profile['steer_gain']:.3f},"
                f"{profile['steer_smooth_alpha']:.2f},speed={profile['speed_cmd']:.1f})"
                for mode, profile in self._profiles.items()
            )
            self.get_logger().info(f"simple motion ready: {summary}")
            self.get_logger().info(
                "START_R approach: GENERAL speed limited to "
                f"{self._cone_approach_speed_cmd:.1f} on "
                f"{self._cone_approach_topic}"
            )
            self.get_logger().info(
                "motor speed ramp: start="
                f"{self._startup_speed_cmd:.1f} hold={self._startup_hold_sec:.1f}s, "
                f"up={self._slew_speed_up_per_tick:.2f}/tick, "
                f"down={self._slew_speed_down_per_tick:.2f}/tick, STOP=immediate"
            )
            block_summary = " ".join(
                f"{direction}=({profile.shift_angle_cmd:+.0f}x{profile.shift_ticks},"
                f"{profile.counter_angle_cmd:+.0f}x{profile.counter_ticks},"
                f"straightx{profile.pass_ticks})"
                for direction, profile in self._block_profiles.items()
            )
            self.get_logger().info(
                "hardcoded obstacle strategy ready: "
                f"immediate-next-tick speed={block_lane_change_speed:.1f} "
                f"{block_summary}"
            )

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

        def _reject_path(self, reason: str) -> None:
            self.get_logger().warning(
                f"simple motion ignoring path: {reason}; keeping last steering/path",
                throttle_duration_sec=1.0,
            )

        def _on_path(self, message: PoseArray) -> None:
            if str(message.header.frame_id) != self._path_frame_id:
                self._reject_path("wrong frame")
                return
            try:
                source_ns = stamp_to_ns(message.header.stamp)
            except ValueError as exc:
                self._reject_path(str(exc))
                return
            if source_ns < self._last_seen_path_ns:
                return
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
                self._reject_path(str(exc))
                return
            # Advance ordering only after a replacement path is accepted.  A
            # malformed message with a far-future stamp must not block later
            # valid paths from restoring steering updates.
            self._last_seen_path_ns = source_ns
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
                if requested == MODE_CONE:
                    self._cancel_hardcoded_block("cone_mode")

        def _on_cone_approach(self, message: Bool) -> None:
            active = bool(message.data)
            if active != self._cone_approach_active:
                self._cone_approach_active = active
                self.get_logger().info(
                    "START_R approach speed limit: "
                    + ("ACTIVE" if active else "released")
                )

        def _remember_block_event(self, event_id: str) -> bool:
            """Return False for a duplicate while keeping bounded history."""

            if event_id in self._block_seen_ids:
                return False
            self._block_seen_ids.add(event_id)
            self._block_seen_order.append(event_id)
            while len(self._block_seen_order) > 64:
                expired = self._block_seen_order.popleft()
                self._block_seen_ids.discard(expired)
            return True

        def _on_overtake_block_request(self, message: String) -> None:
            try:
                request = HardcodedOvertakeRequest.from_json(message.data)
            except ValueError as exc:
                self.get_logger().warning(
                    f"rejected hardcoded obstacle request: {exc}",
                    throttle_duration_sec=1.0,
                )
                return
            if not self._remember_block_event(request.event_id):
                self.get_logger().info(
                    f"ignored duplicate hardcoded obstacle event {request.event_id}"
                )
                return
            if (
                self._block_pending is not None
                or self._overtake_block.active
                or self._cnn_mode == MODE_CONE
            ):
                self.get_logger().warning(
                    "ignored hardcoded obstacle reentry: "
                    f"event={request.event_id} phase={self._block_phase} "
                    f"mode={self._cnn_mode}"
                )
                return
            self._block_pending = request
            self._block_event_id = request.event_id
            self._block_direction = request.direction
            self._block_obstacle_lane = request.obstacle_lane
            self._block_phase = "QUEUED"
            self._block_phase_tick = 0
            self._block_phase_ticks = 0
            self._block_total_tick = 0
            self._block_total_ticks = self._block_profiles[
                request.direction
            ].shift_ticks + self._block_profiles[
                request.direction
            ].counter_ticks + self._block_profiles[request.direction].pass_ticks
            self._block_cancel_reason = ""
            self.get_logger().warning(
                "accepted hardcoded obstacle event: "
                f"{request.event_id} obstacle={request.obstacle_lane} "
                f"move={request.direction}"
            )

        def _cancel_hardcoded_block(self, reason: str) -> None:
            if self._block_pending is None and not self._overtake_block.active:
                return
            self._block_pending = None
            self._overtake_block.cancel()
            self._block_phase = "CANCELLED"
            self._block_phase_tick = 0
            self._block_phase_ticks = 0
            self._block_total_tick = 0
            self._block_total_ticks = 0
            self._block_cancel_reason = str(reason)
            self.get_logger().warning(
                f"hardcoded obstacle block cancelled: {reason}"
            )

        def _fresh_path(self, now_ns: int) -> bool:
            if self._path_x is None or self._path_y is None:
                return False
            age = (now_ns - self._path_source_ns) * 1e-9
            return -self._future_tolerance_sec <= age <= self._path_stale_sec

        def _has_path(self) -> bool:
            return self._path_x is not None and self._path_y is not None

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
            path_available = self._has_path()
            drive = motion_drive_allowed(
                cmd_fresh=cmd_fresh,
                cmd_valid=cmd_valid,
                owner_lane=owner_lane,
                speed_cap=speed_cap,
            )

            profile = self._cnn_mode
            active = self._profiles[profile]
            lookahead_m = active["lookahead_m"]
            steer_gain = active["steer_gain"]
            steer_alpha = active["steer_smooth_alpha"]
            profile_speed = active["speed_cmd"]
            requested_speed, approach_speed_limited = select_requested_speed(
                profile_speed,
                cnn_mode=profile,
                cone_approach_active=self._cone_approach_active,
                cone_approach_speed=self._cone_approach_speed_cmd,
            )

            raw_angle = self._angle_cmd
            target_x = None
            target_y = None
            reason = "ok"
            block_command_published = False
            block_present = bool(
                self._block_pending is not None or self._overtake_block.active
            )
            if block_present and (not drive or profile == MODE_CONE):
                self._cancel_hardcoded_block(
                    "cone_mode" if profile == MODE_CONE else "drive_stop"
                )
                block_present = False

            if drive and block_present:
                # Exclusive block ownership: path callbacks continue caching,
                # but path steering and the IIR are bypassed.  CarInterface is
                # still called exactly once below, so measured calibration,
                # clamps, and steering slew remain active.
                target_x = None
                target_y = None
                if self._block_pending is not None:
                    request = self._block_pending
                    self._block_pending = None
                    self._overtake_block.start(
                        self._block_profiles[request.direction]
                    )
                    block_command = self._overtake_block.step()
                    block_command_published = True
                    self._angle_cmd = block_command.angle_cmd
                    raw_angle = block_command.angle_cmd
                    requested_speed = block_command.speed_cmd
                    target_speed = min(speed_cap, requested_speed)
                    self._block_phase = block_command.phase
                    self._block_phase_tick = block_command.phase_tick
                    self._block_phase_ticks = block_command.phase_ticks
                    self._block_total_tick = block_command.total_tick
                    self._block_total_ticks = block_command.total_ticks
                    reason = f"hardcoded_block_{block_command.phase.lower()}"
                else:
                    block_command = self._overtake_block.step()
                    block_command_published = True
                    self._angle_cmd = block_command.angle_cmd
                    raw_angle = block_command.angle_cmd
                    requested_speed = block_command.speed_cmd
                    target_speed = min(speed_cap, requested_speed)
                    self._block_phase = block_command.phase
                    self._block_phase_tick = block_command.phase_tick
                    self._block_phase_ticks = block_command.phase_ticks
                    self._block_total_tick = block_command.total_tick
                    self._block_total_ticks = block_command.total_ticks
                    reason = f"hardcoded_block_{block_command.phase.lower()}"
            elif drive:
                # The preceding final PASS publish kept PASS N/N in that
                # tick's debug payload.  Only this following path tick returns
                # the stable state to IDLE.
                self._block_phase = PHASE_IDLE
                self._block_phase_tick = 0
                self._block_phase_ticks = 0
                self._block_total_tick = 0
                self._block_total_ticks = 0
                # /drive_cmd remains the global ceiling and explicit STOP
                # watchdog; the active motion profile owns requested speed.
                target_speed = min(speed_cap, requested_speed)
                if path_available:
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
                        if not path_fresh:
                            reason = "holding_last_path"
                    except ValueError as exc:
                        # Cached-path failures also retain the last steering and
                        # speed instead of producing a one-cycle STOP command.
                        reason = f"holding_last_steering:{exc}"
                else:
                    reason = "holding_last_steering_no_path"
            else:
                target_speed = 0.0
                if not cmd_fresh:
                    reason = "drive_cmd_stale"
                elif not cmd_valid:
                    reason = "drive_cmd_invalid"
                elif not owner_lane:
                    reason = f"owner_{self._cmd.get('owner')}"
                else:
                    reason = "speed_cap_missing"

            now_sec = now_ns * 1e-9
            speed_cmd = self._speed_controller.update(
                drive=drive,
                target_speed=target_speed,
                now_sec=now_sec,
            )

            # STOP retains the last logical steering angle.  CarInterface owns
            # the measured left/right scaling, mechanical clamps, and steering
            # slew.  The two-stage controller above owns motor-speed slew while
            # this node continues publishing every 20 Hz tick.
            angle_out, speed_out = self._car.to_motor(self._angle_cmd, speed_cmd)
            output = Float32MultiArray()
            output.data = [float(angle_out), float(speed_out)]
            self._motor_pub.publish(output)

            self._debug = {
                "controller": "simple_motion_v1_gpt",
                "profile": profile,
                "drive": drive,
                "reason": reason,
                "path_available": path_available,
                "path_fresh": path_fresh,
                "path_reused": path_available and not path_fresh,
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
                "profile_speed": profile_speed,
                "requested_speed": requested_speed,
                "target_speed": target_speed,
                "shaped_speed_cmd": speed_cmd,
                "speed_out": speed_out,
                "speed_phase": self._speed_controller.phase,
                "startup_remaining_sec": self._speed_controller.startup_remaining_sec(
                    now_sec
                ),
                "startup_speed_cmd": self._startup_speed_cmd,
                "startup_hold_sec": self._startup_hold_sec,
                "slew_speed_up_per_tick": self._slew_speed_up_per_tick,
                "slew_speed_down_per_tick": self._slew_speed_down_per_tick,
                "cone_approach_active": self._cone_approach_active,
                "cone_approach_speed_limited": approach_speed_limited,
                "cone_approach_speed_cmd": self._cone_approach_speed_cmd,
                "overtake_strategy": self._overtake_strategy,
                "hardcoded_block_active": bool(
                    self._block_pending is not None
                    or self._overtake_block.active
                    or block_command_published
                ),
                "hardcoded_block_queued": self._block_pending is not None,
                "hardcoded_block_event_id": self._block_event_id,
                "hardcoded_block_direction": self._block_direction,
                "hardcoded_block_obstacle_lane": self._block_obstacle_lane,
                "hardcoded_block_phase": self._block_phase,
                "hardcoded_block_phase_tick": self._block_phase_tick,
                "hardcoded_block_phase_ticks": self._block_phase_ticks,
                "hardcoded_block_total_tick": self._block_total_tick,
                "hardcoded_block_total_ticks": self._block_total_ticks,
                "hardcoded_block_cancel_reason": self._block_cancel_reason,
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
