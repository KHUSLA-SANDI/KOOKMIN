"""Pure fixed-pulse lane-change block shared by tuner and race motion.

The class owns only phase/tick selection.  It never publishes motor commands,
so the caller remains the single motor owner and can apply the normal vehicle
calibration, steering slew, and speed ramp exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


DIRECTION_LEFT = "LEFT"
DIRECTION_RIGHT = "RIGHT"
DIRECTIONS = (DIRECTION_LEFT, DIRECTION_RIGHT)

PHASE_SHIFT = "SHIFT"
PHASE_COUNTER = "COUNTER"
PHASE_PASS = "PASS"
PHASE_IDLE = "IDLE"


@dataclass(frozen=True)
class OvertakePulseProfile:
    """One immutable block snapshot captured at the trigger boundary."""

    direction: str
    shift_angle_cmd: float
    shift_ticks: int
    counter_angle_cmd: float
    counter_ticks: int
    lane_change_speed_cmd: float
    pass_speed_cmd: float
    pass_ticks: int

    def __post_init__(self) -> None:
        direction = str(self.direction).strip().upper()
        if direction not in DIRECTIONS:
            raise ValueError(f"unsupported overtake direction: {self.direction!r}")
        object.__setattr__(self, "direction", direction)

        finite_values = (
            self.shift_angle_cmd,
            self.counter_angle_cmd,
            self.lane_change_speed_cmd,
            self.pass_speed_cmd,
        )
        if not all(math.isfinite(float(value)) for value in finite_values):
            raise ValueError("overtake block commands must be finite")
        if not -100.0 <= float(self.shift_angle_cmd) <= 100.0:
            raise ValueError("shift_angle_cmd must be within [-100, 100]")
        if not -100.0 <= float(self.counter_angle_cmd) <= 100.0:
            raise ValueError("counter_angle_cmd must be within [-100, 100]")
        if float(self.lane_change_speed_cmd) <= 0.0:
            raise ValueError("lane_change_speed_cmd must be positive")
        if float(self.pass_speed_cmd) <= 0.0:
            raise ValueError("pass_speed_cmd must be positive")
        for name in ("shift_ticks", "counter_ticks", "pass_ticks"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")

        if direction == DIRECTION_RIGHT:
            if self.shift_angle_cmd <= 0.0 or self.counter_angle_cmd >= 0.0:
                raise ValueError("RIGHT requires positive shift and negative counter")
        elif self.shift_angle_cmd >= 0.0 or self.counter_angle_cmd <= 0.0:
            raise ValueError("LEFT requires negative shift and positive counter")


@dataclass(frozen=True)
class OvertakeBlockCommand:
    direction: str
    phase: str
    phase_tick: int
    phase_ticks: int
    total_tick: int
    total_ticks: int
    angle_cmd: float
    speed_cmd: float
    completes_after_publish: bool


class HardcodedOvertakeBlock:
    """Exclusive three-phase block: SHIFT -> COUNTER -> PASS."""

    def __init__(self) -> None:
        self._profile: Optional[OvertakePulseProfile] = None
        self._tick = 0

    @property
    def active(self) -> bool:
        return self._profile is not None

    @property
    def direction(self) -> Optional[str]:
        return None if self._profile is None else self._profile.direction

    @property
    def total_tick(self) -> int:
        return int(self._tick)

    @property
    def total_ticks(self) -> int:
        profile = self._profile
        if profile is None:
            return 0
        return profile.shift_ticks + profile.counter_ticks + profile.pass_ticks

    def start(self, profile: OvertakePulseProfile) -> None:
        if self.active:
            raise RuntimeError("overtake block is already active")
        self._profile = profile
        self._tick = 0

    def cancel(self) -> None:
        self._profile = None
        self._tick = 0

    def step(self) -> OvertakeBlockCommand:
        profile = self._profile
        if profile is None:
            raise RuntimeError("overtake block is not active")

        shift_end = profile.shift_ticks
        counter_end = shift_end + profile.counter_ticks
        total = counter_end + profile.pass_ticks
        tick = self._tick
        if tick < shift_end:
            phase = PHASE_SHIFT
            phase_tick = tick
            phase_ticks = profile.shift_ticks
            angle = profile.shift_angle_cmd
            speed = profile.lane_change_speed_cmd
        elif tick < counter_end:
            phase = PHASE_COUNTER
            phase_tick = tick - shift_end
            phase_ticks = profile.counter_ticks
            angle = profile.counter_angle_cmd
            speed = profile.lane_change_speed_cmd
        else:
            phase = PHASE_PASS
            phase_tick = tick - counter_end
            phase_ticks = profile.pass_ticks
            angle = 0.0
            speed = profile.pass_speed_cmd

        completes = tick + 1 >= total
        command = OvertakeBlockCommand(
            direction=profile.direction,
            phase=phase,
            phase_tick=phase_tick + 1,
            phase_ticks=phase_ticks,
            total_tick=tick + 1,
            total_ticks=total,
            angle_cmd=float(angle),
            speed_cmd=float(speed),
            completes_after_publish=completes,
        )
        self._tick += 1
        if completes:
            self._profile = None
            self._tick = 0
        return command


__all__ = [
    "DIRECTION_LEFT",
    "DIRECTION_RIGHT",
    "DIRECTIONS",
    "PHASE_SHIFT",
    "PHASE_COUNTER",
    "PHASE_PASS",
    "PHASE_IDLE",
    "OvertakePulseProfile",
    "OvertakeBlockCommand",
    "HardcodedOvertakeBlock",
]
