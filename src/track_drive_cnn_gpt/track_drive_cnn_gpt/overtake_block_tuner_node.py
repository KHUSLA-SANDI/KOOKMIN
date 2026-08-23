#!/usr/bin/env python3
"""Run one fixed steering pulse block on the real car, then stop and exit.

Sequence at 20 Hz (selected by ``move_direction``):

    wait -> accelerate STRAIGHT to the configured speed
         -> SHIFT for N ticks -> COUNTER for N ticks -> STRAIGHT for N ticks
         -> publish zero speed -> process exit

There are no runtime commands and no path/CNN subscriptions.  Edit only
``config/overtake_block_tuner.yaml`` and rerun this executable for each trial.
Never run it together with race drive or teleop because this executable itself
is the sole ``/xycar_motor`` publisher during the trial.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from track_drive.lib.car_interface import CarInterface, DEFAULT_CFG

from .overtake_block import (
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
    HardcodedOvertakeBlock,
    OvertakePulseProfile,
)
from .simple_motion_node import (
    CAR_SPEED_SLEW_BYPASS_PER_TICK,
    TwoStageSpeedController,
)


STOP_PUBLISH_TICKS = 3


class OvertakeBlockTunerNode(Node):
    """Single-run lane shift -> counter steer -> straight pulse executor."""

    def __init__(self) -> None:
        # Reuse the measured ``motion_node`` steering section in car.yaml.
        super().__init__("motion_node")
        p = self._parameter
        self._control_hz = float(p("control_hz", 20.0))
        self._start_delay_sec = float(p("start_delay_sec", 2.0))
        self._motor_topic = str(p("motor_topic", "/xycar_motor"))
        self._move_direction = str(p("move_direction", DIRECTION_LEFT)).strip().upper()

        self._left_angle_cmd = float(p("left_angle_cmd", -60.0))
        self._left_ticks = int(p("left_ticks", 10))
        self._right_angle_cmd = float(p("right_angle_cmd", 60.0))
        self._right_ticks = int(p("right_ticks", 10))
        self._straight_ticks = int(p("straight_ticks", 20))
        self._speed_cmd = float(p("speed_cmd", 12.0))

        self._startup_speed_cmd = float(p("startup_speed_cmd", 5.0))
        self._startup_hold_sec = float(p("startup_hold_sec", 1.0))
        self._slew_speed_up_per_tick = float(
            p("slew_speed_up_per_tick", 0.75)
        )
        self._slew_speed_down_per_tick = float(
            p("slew_speed_down_per_tick", 2.0)
        )

        if not math.isfinite(self._control_hz) or self._control_hz <= 0.0:
            raise ValueError("control_hz must be positive and finite")
        if not math.isfinite(self._start_delay_sec) or self._start_delay_sec < 0.0:
            raise ValueError("start_delay_sec must be non-negative and finite")
        if self._move_direction == DIRECTION_LEFT:
            shift_angle = self._left_angle_cmd
            shift_ticks = self._left_ticks
            counter_angle = self._right_angle_cmd
            counter_ticks = self._right_ticks
        elif self._move_direction == DIRECTION_RIGHT:
            shift_angle = self._right_angle_cmd
            shift_ticks = self._right_ticks
            counter_angle = self._left_angle_cmd
            counter_ticks = self._left_ticks
        else:
            raise ValueError("move_direction must be LEFT or RIGHT")

        self._profile = OvertakePulseProfile(
            direction=self._move_direction,
            shift_angle_cmd=shift_angle,
            shift_ticks=shift_ticks,
            counter_angle_cmd=counter_angle,
            counter_ticks=counter_ticks,
            lane_change_speed_cmd=self._speed_cmd,
            pass_speed_cmd=self._speed_cmd,
            pass_ticks=self._straight_ticks,
        )

        self._speed_controller = TwoStageSpeedController(
            startup_speed_cmd=self._startup_speed_cmd,
            startup_hold_sec=self._startup_hold_sec,
            slew_speed_up_per_tick=self._slew_speed_up_per_tick,
            slew_speed_down_per_tick=self._slew_speed_down_per_tick,
        )
        car_cfg = {key: p(key, DEFAULT_CFG[key]) for key in sorted(DEFAULT_CFG)}
        # Use the exact same ownership split as production simple_motion:
        # this node shapes motor speed; CarInterface owns steering calibration
        # and steering slew exactly once.
        car_cfg["slew_speed_per_tick"] = CAR_SPEED_SLEW_BYPASS_PER_TICK
        self._car = CarInterface(car_cfg)

        self._block = HardcodedOvertakeBlock()
        self._ownership_checked = False
        self._speed_ready = False
        self._last_phase: Optional[str] = None
        self._stop_ticks_remaining = 0
        self._finished = False
        self._start_ns = (
            self.get_clock().now().nanoseconds
            + int(round(self._start_delay_sec * 1e9))
        )
        self._motor_pub = self.create_publisher(
            Float32MultiArray, self._motor_topic, 10
        )
        self.create_timer(1.0 / self._control_hz, self._tick)

        self.get_logger().warning(
            f"ONE-SHOT {self._move_direction} shift armed: "
            f"SHIFT {shift_angle:+.1f} x{shift_ticks} -> "
            f"COUNTER {counter_angle:+.1f} x{counter_ticks} -> "
            f"STRAIGHT x{self._straight_ticks}; starts in "
            f"{self._start_delay_sec:.1f}s after reaching speed "
            f"{self._speed_cmd:.1f}"
        )

    def _parameter(self, name: str, default: Any) -> Any:
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _publish_motor(self, angle_cmd: float, speed_cmd: float) -> None:
        angle_out, speed_out = self._car.to_motor(angle_cmd, speed_cmd)
        message = Float32MultiArray()
        message.data = [float(angle_out), float(speed_out)]
        self._motor_pub.publish(message)

    def _finish_after_stop(self) -> None:
        self._finished = True
        self.get_logger().warning("ONE-SHOT complete: motor speed=0, exiting")
        rclpy.shutdown()

    def _tick(self) -> None:
        if self._finished:
            return
        now_ns = self.get_clock().now().nanoseconds
        now_sec = now_ns * 1e-9

        if self._stop_ticks_remaining > 0:
            stopped_speed = self._speed_controller.update(
                drive=False, target_speed=0.0, now_sec=now_sec
            )
            self._publish_motor(0.0, stopped_speed)
            self._stop_ticks_remaining -= 1
            if self._stop_ticks_remaining == 0:
                self._finish_after_stop()
            return

        if now_ns < self._start_ns:
            stopped_speed = self._speed_controller.update(
                drive=False, target_speed=0.0, now_sec=now_sec
            )
            self._publish_motor(0.0, stopped_speed)
            return

        if not self._ownership_checked:
            # One publisher endpoint is this node.  Refuse the trial if a race
            # motion or teleop publisher is still present.
            publishers = self.get_publishers_info_by_topic(self._motor_topic)
            if len(publishers) > 1:
                self.get_logger().error(
                    "another /xycar_motor publisher exists; trial cancelled"
                )
                self._stop_ticks_remaining = STOP_PUBLISH_TICKS
                return
            self._ownership_checked = True

        if not self._speed_ready:
            shaped_speed = self._speed_controller.update(
                drive=True,
                target_speed=self._speed_cmd,
                now_sec=now_sec,
            )
            self._publish_motor(0.0, shaped_speed)
            if shaped_speed >= self._speed_cmd - 1e-6:
                self._speed_ready = True
                self._block.start(self._profile)
                self.get_logger().warning(
                    f"speed ready at {shaped_speed:.1f}; "
                    f"{self._move_direction} shift starts next tick"
                )
            return

        command = self._block.step()
        if command.phase != self._last_phase:
            self._last_phase = command.phase
            self.get_logger().warning(
                f"phase={command.phase} angle={command.angle_cmd:+.1f} "
                f"ticks={command.phase_ticks} speed={command.speed_cmd:.1f}"
            )
        shaped_speed = self._speed_controller.update(
            drive=True,
            target_speed=command.speed_cmd,
            now_sec=now_sec,
        )
        self._publish_motor(command.angle_cmd, shaped_speed)
        if command.completes_after_publish:
            # The final straight command remains on the motor for one complete
            # 20 Hz tick; zero begins on the next tick.
            self._stop_ticks_remaining = STOP_PUBLISH_TICKS

    def destroy_node(self) -> bool:
        if not self._finished:
            message = Float32MultiArray()
            message.data = [0.0, 0.0]
            self._motor_pub.publish(message)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[OvertakeBlockTunerNode] = None
    try:
        node = OvertakeBlockTunerNode()
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
