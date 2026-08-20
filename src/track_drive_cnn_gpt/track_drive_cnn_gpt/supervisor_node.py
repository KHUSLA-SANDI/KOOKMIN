"""Fail-closed route supervisor between CNN paths and the legacy motion node."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

from .motion_path_adapter import stamp_to_ns, validate_source_stamp_ns


ROUTE_MAIN = "main"
ROUTE_SHORTCUT = "shortcut"
OWNER_LANE = 0.0
OWNER_SAFETY_STOP = 3.0
STEER_PROFILE_NORMAL = 0.0


@dataclass(frozen=True)
class SupervisorDecision:
    drive_allowed: bool
    selected_route: Optional[str]
    reason: str


@dataclass(frozen=True)
class PairedPaths:
    """Main and shortcut samples produced from one BEV source stamp."""

    stamp_ns: int
    main: Any
    shortcut: Any
    main_usable: bool
    shortcut_usable: bool


class PathPairAssembler:
    """Atomically assemble the two route topics by their source timestamp."""

    def __init__(self, max_pending: int = 8) -> None:
        if int(max_pending) < 2:
            raise ValueError("max_pending must be at least two")
        self.max_pending = int(max_pending)
        self._pending: dict[int, dict[str, tuple[Any, bool]]] = {}
        self.last_committed_stamp_ns = -1

    def clear(self) -> None:
        self._pending.clear()

    def add(
        self,
        route: str,
        stamp_ns: int,
        value: Any,
        *,
        usable: bool,
    ) -> Optional[PairedPaths]:
        if route not in (ROUTE_MAIN, ROUTE_SHORTCUT):
            raise ValueError(f"unsupported route: {route}")
        stamp = int(stamp_ns)
        if stamp <= 0:
            raise ValueError("path source stamp must be positive")
        if stamp <= self.last_committed_stamp_ns:
            return None

        row = self._pending.setdefault(stamp, {})
        row[route] = (value, bool(usable))
        while len(self._pending) > self.max_pending:
            del self._pending[min(self._pending)]
        if ROUTE_MAIN not in row or ROUTE_SHORTCUT not in row:
            return None

        main, main_usable = row[ROUTE_MAIN]
        shortcut, shortcut_usable = row[ROUTE_SHORTCUT]
        self.last_committed_stamp_ns = stamp
        for old_stamp in tuple(self._pending):
            if old_stamp <= stamp:
                del self._pending[old_stamp]
        return PairedPaths(
            stamp_ns=stamp,
            main=main,
            shortcut=shortcut,
            main_usable=main_usable,
            shortcut_usable=shortcut_usable,
        )


class SupervisorLogic:
    """ROS-independent arming, route selection, and freshness gate."""

    def __init__(
        self,
        *,
        enable_drive: bool = False,
        path_stale_sec: float = 0.25,
        shortcut_fallback_to_main: bool = False,
    ) -> None:
        if path_stale_sec <= 0.0:
            raise ValueError("path_stale_sec must be positive")
        self.enable_drive = bool(enable_drive)
        self.path_stale_sec = float(path_stale_sec)
        self.shortcut_fallback_to_main = bool(shortcut_fallback_to_main)
        self.manual_go = False
        self.emergency_stop = False
        self.route_intent = ROUTE_MAIN
        self._main_path_time: Optional[float] = None
        self._shortcut_path_time: Optional[float] = None

    def set_manual_go(self, enabled: bool) -> None:
        self.manual_go = bool(enabled)

    def set_emergency_stop(self, active: bool) -> None:
        self.emergency_stop = bool(active)

    def set_route_intent(self, intent: str) -> None:
        normalized = str(intent).strip().lower()
        aliases = {
            "main": ROUTE_MAIN,
            "normal": ROUTE_MAIN,
            "default": ROUTE_MAIN,
            "shortcut": ROUTE_SHORTCUT,
            "short": ROUTE_SHORTCUT,
            "left": ROUTE_SHORTCUT,
        }
        self.route_intent = aliases.get(normalized, normalized)

    def update_path(self, route: str, received_at: Optional[float]) -> None:
        if received_at is not None and not math.isfinite(float(received_at)):
            received_at = None
        if route == ROUTE_MAIN:
            self._main_path_time = None if received_at is None else float(received_at)
        elif route == ROUTE_SHORTCUT:
            self._shortcut_path_time = None if received_at is None else float(received_at)
        else:
            raise ValueError(f"unsupported route: {route}")

    def _fresh(self, timestamp: Optional[float], now: float) -> bool:
        if timestamp is None or not math.isfinite(float(now)):
            return False
        age = float(now) - timestamp
        return 0.0 <= age <= self.path_stale_sec

    def decide(self, now: float) -> SupervisorDecision:
        if not self.enable_drive:
            return SupervisorDecision(False, None, "drive_disabled")
        if self.emergency_stop:
            return SupervisorDecision(False, None, "emergency_stop")
        if not self.manual_go:
            return SupervisorDecision(False, None, "manual_go_required")
        if self.route_intent not in (ROUTE_MAIN, ROUTE_SHORTCUT):
            return SupervisorDecision(False, None, "invalid_route_intent")

        main_fresh = self._fresh(self._main_path_time, now)
        if self.route_intent == ROUTE_MAIN:
            if not main_fresh:
                return SupervisorDecision(False, None, "main_path_stale")
            return SupervisorDecision(True, ROUTE_MAIN, "ok")

        if self._fresh(self._shortcut_path_time, now):
            return SupervisorDecision(True, ROUTE_SHORTCUT, "ok")
        if self.shortcut_fallback_to_main and main_fresh:
            return SupervisorDecision(True, ROUTE_MAIN, "shortcut_fallback_main")
        return SupervisorDecision(False, None, "shortcut_path_stale")


def encode_drive_command(drive_allowed: bool, speed_cap: float) -> list[float]:
    """Encode the legacy track_drive `/drive_cmd` eight-float contract."""

    if not drive_allowed:
        return [OWNER_SAFETY_STOP, STEER_PROFILE_NORMAL, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0]
    cap = float(speed_cap)
    if not math.isfinite(cap) or cap <= 0.0:
        return [OWNER_SAFETY_STOP, STEER_PROFILE_NORMAL, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0]
    return [OWNER_LANE, STEER_PROFILE_NORMAL, cap, 0.0,
            0.0, 0.0, 0.0, 0.0]


try:
    import rclpy
    from geometry_msgs.msg import PoseArray
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float32MultiArray, String
except ImportError:
    rclpy = None
    Node = object


if rclpy is not None:

    class CnnSupervisorNode(Node):
        def __init__(self) -> None:
            super().__init__("cnn_supervisor_node")

            self.declare_parameter("main_path_topic", "/cnn/path_main")
            self.declare_parameter("shortcut_path_topic", "/cnn/path_shortcut")
            self.declare_parameter("center_path_topic", "/center_path")
            self.declare_parameter("drive_cmd_topic", "/drive_cmd")
            self.declare_parameter("route_intent_topic", "/route_intent")
            self.declare_parameter("manual_go_topic", "/manual_go")
            self.declare_parameter("emergency_stop_topic", "/emergency_stop")
            self.declare_parameter("control_hz", 20.0)
            self.declare_parameter("path_stale_sec", 0.25)
            self.declare_parameter("path_future_tolerance_sec", 0.05)
            self.declare_parameter("path_frame_id", "lidar_frame")
            self.declare_parameter("shortcut_fallback_to_main", False)
            self.declare_parameter("enable_drive", False)
            self.declare_parameter("speed_cap", 6.0)

            control_hz = float(self.get_parameter("control_hz").value)
            stale_sec = float(self.get_parameter("path_stale_sec").value)
            speed_cap = float(self.get_parameter("speed_cap").value)
            enable_drive = bool(self.get_parameter("enable_drive").value)
            if control_hz <= 0.0:
                raise ValueError("control_hz must be positive")
            if speed_cap <= 0.0 or not math.isfinite(speed_cap):
                self.get_logger().error("invalid speed_cap; drive gate forced off")
                enable_drive = False

            self._speed_cap = speed_cap
            self._path_stale_sec = stale_sec
            self._path_future_tolerance_sec = float(
                self.get_parameter("path_future_tolerance_sec").value
            )
            self._path_frame_id = str(self.get_parameter("path_frame_id").value)
            self._logic = SupervisorLogic(
                enable_drive=enable_drive,
                path_stale_sec=stale_sec,
                shortcut_fallback_to_main=bool(
                    self.get_parameter("shortcut_fallback_to_main").value
                ),
            )
            self._paths = {ROUTE_MAIN: None, ROUTE_SHORTCUT: None}
            self._paired_messages = {ROUTE_MAIN: None, ROUTE_SHORTCUT: None}
            self._path_pairs = PathPairAssembler()
            self._last_status = None

            main_topic = str(self.get_parameter("main_path_topic").value)
            shortcut_topic = str(self.get_parameter("shortcut_path_topic").value)
            self.create_subscription(PoseArray, main_topic, self._on_main_path, 10)
            self.create_subscription(
                PoseArray, shortcut_topic, self._on_shortcut_path, 10
            )
            self.create_subscription(
                String,
                str(self.get_parameter("route_intent_topic").value),
                self._on_route_intent,
                10,
            )
            self.create_subscription(
                Bool,
                str(self.get_parameter("manual_go_topic").value),
                self._on_manual_go,
                10,
            )
            self.create_subscription(
                Bool,
                str(self.get_parameter("emergency_stop_topic").value),
                self._on_emergency_stop,
                10,
            )

            self._center_pub = self.create_publisher(
                PoseArray, str(self.get_parameter("center_path_topic").value), 10
            )
            self._drive_pub = self.create_publisher(
                Float32MultiArray,
                str(self.get_parameter("drive_cmd_topic").value),
                10,
            )
            self.create_timer(1.0 / control_hz, self._tick)
            self.get_logger().info(
                "CNN supervisor ready: enable_drive=%s; manual_go and fresh path required"
                % str(enable_drive).lower()
            )

        def _now(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

        @staticmethod
        def _path_valid(msg: PoseArray) -> bool:
            if len(msg.poses) < 2:
                return False
            previous_x = None
            for pose in msg.poses:
                x_value = float(pose.position.x)
                y_value = float(pose.position.y)
                if not (math.isfinite(x_value) and math.isfinite(y_value)):
                    return False
                if previous_x is not None and x_value <= previous_x:
                    return False
                previous_x = x_value
            return True

        def _clear_cached_paths(self, reason: str) -> None:
            """Drop the active pair without discarding a new pending pair.

            An empty route message is an explicit negative result for one BEV
            frame.  Keeping the preceding active pair until its peer arrives
            would let an old path drive after the CNN has already said that the
            new result is unusable.  Clear both active routes immediately so
            the cache always represents either one atomic source pair or no
            pair at all.
            """

            self._paths = {ROUTE_MAIN: None, ROUTE_SHORTCUT: None}
            self._paired_messages = {ROUTE_MAIN: None, ROUTE_SHORTCUT: None}
            self._logic.update_path(ROUTE_MAIN, None)
            self._logic.update_path(ROUTE_SHORTCUT, None)

            self.get_logger().warning(
                f"active CNN path pair cleared: {reason}",
                throttle_duration_sec=1.0,
            )

        def _invalidate_all_paths(self, reason: str) -> None:
            self._clear_cached_paths(reason)
            self._path_pairs.clear()

        def _commit_pair(self, pair: PairedPaths) -> None:
            source_time = pair.stamp_ns * 1e-9
            self._paired_messages[ROUTE_MAIN] = pair.main
            self._paired_messages[ROUTE_SHORTCUT] = pair.shortcut
            self._paths[ROUTE_MAIN] = pair.main if pair.main_usable else None
            self._paths[ROUTE_SHORTCUT] = (
                pair.shortcut if pair.shortcut_usable else None
            )
            self._logic.update_path(
                ROUTE_MAIN, source_time if pair.main_usable else None
            )
            self._logic.update_path(
                ROUTE_SHORTCUT, source_time if pair.shortcut_usable else None
            )

        def _store_path(self, route: str, msg: PoseArray) -> None:
            if str(msg.header.frame_id) != self._path_frame_id:
                self._invalidate_all_paths(
                    f"{route} frame {msg.header.frame_id!r} != {self._path_frame_id!r}"
                )
                self._center_pub.publish(msg)
                return
            try:
                source_ns = stamp_to_ns(msg.header.stamp)
            except ValueError as exc:
                self._invalidate_all_paths(f"{route}: {exc}")
                self._center_pub.publish(msg)
                return

            # A delayed sample from an already committed frame is harmless and
            # must not clear the newer active pair merely because it became
            # stale or is empty.  Zero is never a valid source timestamp.
            if source_ns > 0:
                if source_ns <= self._path_pairs.last_committed_stamp_ns:
                    return

            try:
                validate_source_stamp_ns(
                    source_ns,
                    self.get_clock().now().nanoseconds,
                    stale_sec=self._path_stale_sec,
                    future_tolerance_sec=self._path_future_tolerance_sec,
                )
            except ValueError as exc:
                self._invalidate_all_paths(f"{route}: {exc}")
                # The motion overlay must see an explicit invalidation as well
                # as the STOP command; otherwise its previous path array would
                # remain cached until timeout.  It validates frame/stamp again.
                self._center_pub.publish(msg)
                return

            usable = self._path_valid(msg)
            if not usable:
                # Do not let a previously cached pair survive an explicit
                # empty/invalid result.  Keep the pending entries, however, so
                # main and shortcut for this stamp can still commit together.
                self._clear_cached_paths(
                    f"unusable {route} path for stamp={source_ns}"
                )
                if route == self._logic.route_intent:
                    self._center_pub.publish(msg)
            pair = self._path_pairs.add(
                route,
                source_ns,
                msg,
                usable=usable,
            )
            if pair is not None:
                self._commit_pair(pair)

        def _on_main_path(self, msg: PoseArray) -> None:
            self._store_path(ROUTE_MAIN, msg)

        def _on_shortcut_path(self, msg: PoseArray) -> None:
            self._store_path(ROUTE_SHORTCUT, msg)

        def _on_route_intent(self, msg: String) -> None:
            self._logic.set_route_intent(msg.data)
            selected = self._logic.route_intent
            # If the newly requested route is explicitly unavailable in the
            # latest atomic pair, forward that empty message once so motion
            # clears any path cached under the previous route immediately.
            invalid = self._paired_messages.get(selected)
            if selected in self._paths and self._paths[selected] is None:
                if invalid is not None:
                    self._center_pub.publish(invalid)

        def _on_manual_go(self, msg: Bool) -> None:
            self._logic.set_manual_go(msg.data)

        def _on_emergency_stop(self, msg: Bool) -> None:
            self._logic.set_emergency_stop(msg.data)

        def _tick(self) -> None:
            decision = self._logic.decide(self._now())
            drive_allowed = decision.drive_allowed
            selected_path = self._paths.get(decision.selected_route)
            if drive_allowed and selected_path is not None:
                self._center_pub.publish(selected_path)
            else:
                drive_allowed = False

            command = Float32MultiArray()
            command.data = encode_drive_command(drive_allowed, self._speed_cap)
            self._drive_pub.publish(command)

            status = (drive_allowed, decision.selected_route, decision.reason)
            if status != self._last_status:
                level = self.get_logger().info if drive_allowed else self.get_logger().warning
                level(
                    "supervisor %s route=%s reason=%s"
                    % (
                        "DRIVE" if drive_allowed else "STOP",
                        decision.selected_route or "-",
                        decision.reason,
                    )
                )
                self._last_status = status


def main(args=None) -> None:
    if rclpy is None:
        raise RuntimeError("ROS 2 Python packages are required to run cnn_supervisor")
    rclpy.init(args=args)
    node = CnnSupervisorNode()
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
