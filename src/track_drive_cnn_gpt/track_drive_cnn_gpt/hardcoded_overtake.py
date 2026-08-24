"""Pure contract between LiDAR obstacle classification and simple motion.

Version 3 keeps the CNN mode GENERAL and emits one directional request after
two consecutive, same-side obstacle frames.  The motion node remains the sole
``/xycar_motor`` publisher and consumes the request exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Optional

from .overtake_block import DIRECTION_LEFT, DIRECTION_RIGHT, DIRECTIONS


STRATEGY_HARDCODED_ALL = "hardcoded_all"
STRATEGY_HARDCODED_POST_CONE = "hardcoded_post_cone"
OVERTAKE_STRATEGIES = (
    STRATEGY_HARDCODED_ALL,
    STRATEGY_HARDCODED_POST_CONE,
)


class PostConeHardcodeWindow:
    """Five-second fixed-obstacle window tied to CONE's true deadline."""

    def __init__(self, *, window_sec: float = 5.0) -> None:
        self.window_sec = float(window_sec)
        if not math.isfinite(self.window_sec) or self.window_sec <= 0.0:
            raise ValueError("post-cone window must be positive and finite")
        self._cone_deadline_sec = -math.inf
        self._until_sec = -math.inf

    def observe_transition(
        self,
        *,
        previous_mode: str,
        mode: str,
        changed: bool,
        reason: str,
        remaining_sec: float,
        now_sec: float,
    ) -> None:
        now = float(now_sec)
        remaining = float(remaining_sec)
        if not math.isfinite(now) or not math.isfinite(remaining):
            raise ValueError("mode transition time must be finite")
        previous = str(previous_mode).strip().upper()
        current = str(mode).strip().upper()
        transition_reason = str(reason)
        if current == "CONE" and bool(changed):
            self._cone_deadline_sec = now + max(0.0, remaining)
            self._until_sec = -math.inf
        elif previous == "CONE" and current == "GENERAL":
            if bool(changed) and transition_reason == "timer_expired":
                self._until_sec = self._cone_deadline_sec + self.window_sec
            else:
                # Explicit route/reset changes must never arm the window.
                self._until_sec = -math.inf

    def remaining_sec(self, now_sec: float) -> float:
        now = float(now_sec)
        if not math.isfinite(now):
            raise ValueError("now_sec must be finite")
        return max(0.0, self._until_sec - now)

    def active(self, now_sec: float) -> bool:
        return self.remaining_sec(now_sec) > 0.0
OBSTACLE_LANE_1 = "LANE1_LEFT"
OBSTACLE_LANE_2 = "LANE2_RIGHT"
OBSTACLE_UNKNOWN = "UNKNOWN"
OBSTACLE_LANES = (OBSTACLE_LANE_1, OBSTACLE_LANE_2)


def classify_obstacle_lane(
    lateral_offset_m: float | None,
    *,
    deadband_m: float,
) -> str:
    """Classify the obstacle relative to the yellow/mid road reference.

    BEV uses positive ``y`` to the vehicle's left.  The measured data clusters
    lane 1 at positive offsets and lane 2 at negative offsets.  Values inside
    the deadband are deliberately UNKNOWN and cannot start a block.
    """

    deadband = float(deadband_m)
    if not math.isfinite(deadband) or deadband <= 0.0:
        raise ValueError("obstacle_lane_deadband_m must be positive and finite")
    if lateral_offset_m is None:
        return OBSTACLE_UNKNOWN
    offset = float(lateral_offset_m)
    if not math.isfinite(offset):
        return OBSTACLE_UNKNOWN
    if offset >= deadband:
        return OBSTACLE_LANE_1
    if offset <= -deadband:
        return OBSTACLE_LANE_2
    return OBSTACLE_UNKNOWN


def avoidance_direction(obstacle_lane: str) -> str:
    """Return vehicle shift direction away from the occupied lane."""

    lane = str(obstacle_lane).strip().upper()
    if lane == OBSTACLE_LANE_1:
        return DIRECTION_RIGHT
    if lane == OBSTACLE_LANE_2:
        return DIRECTION_LEFT
    raise ValueError(f"ambiguous obstacle lane cannot start a block: {obstacle_lane!r}")


class DirectionalConfirmedOneShot:
    """Confirm one lane label N times and re-arm only after true clear frames.

    UNKNOWN means an obstacle was detected but its side is ambiguous: it resets
    the same-side streak, but does not count as a clear frame.  ``None`` means
    no effective obstacle and is the only observation that re-arms a fired
    event.  This prevents a noisy left/right flip on the same obstacle from
    becoming a second maneuver.
    """

    def __init__(self, *, confirm_frames: int = 2, rearm_clear_frames: int = 5):
        if int(confirm_frames) < 1 or int(rearm_clear_frames) < 1:
            raise ValueError("trigger frame counts must be positive")
        self.confirm_frames = int(confirm_frames)
        self.rearm_clear_frames = int(rearm_clear_frames)
        self._candidate = OBSTACLE_UNKNOWN
        self._positive = 0
        self._clear = 0
        self._armed = True

    def observe(self, obstacle_lane: Optional[str]) -> Optional[str]:
        if obstacle_lane is None:
            self._candidate = OBSTACLE_UNKNOWN
            self._positive = 0
            if not self._armed:
                self._clear += 1
                if self._clear >= self.rearm_clear_frames:
                    self._armed = True
                    self._clear = 0
            return None

        lane = str(obstacle_lane).strip().upper()
        self._clear = 0
        if lane not in OBSTACLE_LANES:
            self._candidate = OBSTACLE_UNKNOWN
            self._positive = 0
            return None
        if not self._armed:
            return None
        if lane != self._candidate:
            self._candidate = lane
            self._positive = 1
        else:
            self._positive += 1
        if self._positive < self.confirm_frames:
            return None
        confirmed = self._candidate
        self._candidate = OBSTACLE_UNKNOWN
        self._positive = 0
        self._armed = False
        return confirmed

    @property
    def positive_count(self) -> int:
        return int(self._positive)

    @property
    def candidate(self) -> str:
        return str(self._candidate)

    @property
    def armed(self) -> bool:
        return bool(self._armed)


@dataclass(frozen=True)
class HardcodedOvertakeRequest:
    event_id: str
    direction: str
    obstacle_lane: str
    lateral_offset_m: float

    def __post_init__(self) -> None:
        event_id = str(self.event_id).strip()
        direction = str(self.direction).strip().upper()
        obstacle_lane = str(self.obstacle_lane).strip().upper()
        offset = float(self.lateral_offset_m)
        if not event_id:
            raise ValueError("hardcoded block event_id is empty")
        if direction not in DIRECTIONS:
            raise ValueError(f"invalid hardcoded block direction: {direction!r}")
        if obstacle_lane not in OBSTACLE_LANES:
            raise ValueError(f"invalid obstacle lane: {obstacle_lane!r}")
        if not math.isfinite(offset):
            raise ValueError("hardcoded block lateral offset must be finite")
        if avoidance_direction(obstacle_lane) != direction:
            raise ValueError("hardcoded block direction does not avoid obstacle lane")
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "obstacle_lane", obstacle_lane)
        object.__setattr__(self, "lateral_offset_m", offset)

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema": "hardcoded_overtake_request_v1_gpt",
                "event_id": self.event_id,
                "direction": self.direction,
                "obstacle_lane": self.obstacle_lane,
                "lateral_offset_m": round(self.lateral_offset_m, 4),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, value: str) -> "HardcodedOvertakeRequest":
        try:
            payload = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid hardcoded block JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("hardcoded block payload must be an object")
        if payload.get("schema") != "hardcoded_overtake_request_v1_gpt":
            raise ValueError("unsupported hardcoded block payload schema")
        try:
            return cls(
                event_id=payload["event_id"],
                direction=payload["direction"],
                obstacle_lane=payload["obstacle_lane"],
                lateral_offset_m=payload["lateral_offset_m"],
            )
        except KeyError as exc:
            raise ValueError(f"hardcoded block payload missing {exc.args[0]}") from exc


__all__ = [
    "STRATEGY_HARDCODED_ALL",
    "STRATEGY_HARDCODED_POST_CONE",
    "OVERTAKE_STRATEGIES",
    "PostConeHardcodeWindow",
    "OBSTACLE_LANE_1",
    "OBSTACLE_LANE_2",
    "OBSTACLE_UNKNOWN",
    "OBSTACLE_LANES",
    "classify_obstacle_lane",
    "avoidance_direction",
    "DirectionalConfirmedOneShot",
    "HardcodedOvertakeRequest",
]
