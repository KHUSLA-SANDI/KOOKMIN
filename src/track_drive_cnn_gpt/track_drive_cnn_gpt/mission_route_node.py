"""Minimal explicit-reset mission latch for CNN main/shortcut selection."""

from __future__ import annotations

from .signal_mission import (
    RouteIntentLatch,
    decode_signal_payload,
)


try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, String
except ImportError:
    rclpy = None
    Node = object


if rclpy is not None:

    class MissionRouteNode(Node):
        """Publish ``main`` until confirmed LEFT, then latch ``shortcut``.

        There is deliberately no time-based automatic return: track timing is
        not calibrated yet.  ``/route_intent_reset`` must explicitly request
        the safe return to ``main``.
        """

        def __init__(self) -> None:
            super().__init__("mission_route_node")
            self.declare_parameter("signal_topic", "/perception/signals")
            self.declare_parameter("route_intent_topic", "/route_intent")
            self.declare_parameter("route_reset_topic", "/route_intent_reset")
            self.declare_parameter("left_confirm_frames", 2)
            self.declare_parameter("left_confidence", 0.25)
            self.declare_parameter("publish_hz", 5.0)

            publish_hz = float(self.get_parameter("publish_hz").value)
            if publish_hz <= 0.0:
                raise ValueError("publish_hz must be positive")
            self._logic = RouteIntentLatch(
                left_confirm_frames=int(
                    self.get_parameter("left_confirm_frames").value
                ),
                left_confidence=float(self.get_parameter("left_confidence").value),
            )
            self._last_sequence = -1
            self._publisher = self.create_publisher(
                String,
                str(self.get_parameter("route_intent_topic").value),
                1,
            )
            self.create_subscription(
                String,
                str(self.get_parameter("signal_topic").value),
                self._on_signals,
                1,
            )
            self.create_subscription(
                Bool,
                str(self.get_parameter("route_reset_topic").value),
                self._on_reset,
                1,
            )
            self.create_timer(1.0 / publish_hz, self._publish)
            self.get_logger().info(
                "mission route latch ready: main -> shortcut on confirmed LEFT; "
                "return requires explicit route_intent_reset"
            )

        def _publish(self) -> None:
            message = String()
            message.data = self._logic.route_intent
            self._publisher.publish(message)

        def _on_signals(self, message: String) -> None:
            try:
                sequence, signals = decode_signal_payload(message.data)
            except ValueError as exc:
                self.get_logger().warning(
                    f"rejected signal payload: {exc}",
                    throttle_duration_sec=1.0,
                )
                return
            # A duplicate must not create a false two-hit confirmation.  A
            # lower sequence means the YOLO node restarted; reset only the
            # confirmation streak and accept the new source without clearing
            # an already-latched route decision.
            if sequence == self._last_sequence:
                return
            if sequence < self._last_sequence:
                self._logic.reset_observation_streak()
                self.get_logger().warning(
                    "YOLO signal sequence restarted; LEFT confirmation streak reset"
                )
            self._last_sequence = sequence
            before = self._logic.route_intent
            after = self._logic.observe(signals)
            if after != before:
                self.get_logger().info("route intent latched: shortcut")
                self._publish()

        def _on_reset(self, message: Bool) -> None:
            if not message.data:
                return
            self._logic.reset_main()
            self._last_sequence = -1
            self.get_logger().info("route intent explicitly reset: main")
            self._publish()


def main(args=None) -> None:
    if rclpy is None:
        raise RuntimeError("ROS 2 Python packages are required to run mission_route")
    rclpy.init(args=args)
    node = MissionRouteNode()
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
