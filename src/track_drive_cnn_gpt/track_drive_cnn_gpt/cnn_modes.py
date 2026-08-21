"""Pure four-CNN mode state machine.

Only the active model is inferred.  Durations are wall-clock seconds, not
camera frames, so skipped frames cannot stretch or shorten a course section.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


MODE_GENERAL = "GENERAL"
MODE_SHORTCUT = "SHORTCUT"
MODE_OVERTAKE = "OVERTAKE"
MODE_CONE = "CONE"
MODES = (MODE_GENERAL, MODE_SHORTCUT, MODE_OVERTAKE, MODE_CONE)
_PRIORITY = {
    MODE_GENERAL: 0,
    MODE_SHORTCUT: 1,
    MODE_OVERTAKE: 2,
    MODE_CONE: 3,
}


@dataclass(frozen=True)
class ModeSnapshot:
    mode: str
    changed: bool
    remaining_sec: float
    reason: str


class CnnModeController:
    """GENERAL plus fixed, non-extending specialized-mode windows."""

    def __init__(
        self,
        *,
        shortcut_hold_sec: float = 10.0,
        overtake_hold_sec: float = 7.0,
        cone_hold_sec: float = 20.0,
    ) -> None:
        durations = {
            MODE_SHORTCUT: float(shortcut_hold_sec),
            MODE_OVERTAKE: float(overtake_hold_sec),
            MODE_CONE: float(cone_hold_sec),
        }
        if not all(math.isfinite(value) and value > 0.0 for value in durations.values()):
            raise ValueError("specialized mode durations must be positive and finite")
        self._durations = durations
        self._mode = MODE_GENERAL
        self._until_sec = -math.inf
        self._reason = "startup"

    def snapshot(self, now_sec: float) -> ModeSnapshot:
        now = self._finite_time(now_sec)
        changed = False
        reason = self._reason
        if self._mode != MODE_GENERAL and now >= self._until_sec:
            self._mode = MODE_GENERAL
            self._until_sec = -math.inf
            self._reason = "timer_expired"
            reason = self._reason
            changed = True
        remaining = (
            0.0
            if self._mode == MODE_GENERAL
            else max(0.0, self._until_sec - now)
        )
        return ModeSnapshot(self._mode, changed, remaining, reason)

    def trigger(self, mode: str, *, now_sec: float, reason: str) -> ModeSnapshot:
        now = self._finite_time(now_sec)
        requested = str(mode).strip().upper()
        if requested not in self._durations:
            raise ValueError(f"unsupported specialized mode: {mode!r}")
        current = self.snapshot(now)
        # A repeated detection never extends its own fixed window.  A higher
        # priority course mode may pre-empt a lower one (CONE > OVERTAKE >
        # SHORTCUT), while lower-priority noise is ignored.
        if requested == self._mode or _PRIORITY[requested] <= _PRIORITY[self._mode]:
            return ModeSnapshot(
                self._mode,
                current.changed,
                current.remaining_sec,
                self._reason,
            )
        self._mode = requested
        self._until_sec = now + self._durations[requested]
        self._reason = str(reason)
        return ModeSnapshot(
            self._mode,
            True,
            self._durations[requested],
            self._reason,
        )

    def reset(self, *, now_sec: float, reason: str = "explicit_reset") -> ModeSnapshot:
        self._finite_time(now_sec)
        changed = self._mode != MODE_GENERAL
        self._mode = MODE_GENERAL
        self._until_sec = -math.inf
        self._reason = str(reason)
        return ModeSnapshot(self._mode, changed, 0.0, self._reason)

    @staticmethod
    def _finite_time(value: float) -> float:
        now = float(value)
        if not math.isfinite(now):
            raise ValueError("now_sec must be finite")
        return now


class ConfirmedOneShot:
    """Emit once after N positive frames; re-arm after M negative frames."""

    def __init__(self, *, confirm_frames: int = 2, rearm_clear_frames: int = 5):
        if int(confirm_frames) < 1 or int(rearm_clear_frames) < 1:
            raise ValueError("trigger frame counts must be positive")
        self.confirm_frames = int(confirm_frames)
        self.rearm_clear_frames = int(rearm_clear_frames)
        self._positive = 0
        self._clear = 0
        self._armed = True

    def observe(self, detected: bool) -> bool:
        if bool(detected):
            self._clear = 0
            if not self._armed:
                return False
            self._positive += 1
            if self._positive >= self.confirm_frames:
                self._positive = 0
                self._armed = False
                return True
            return False
        self._positive = 0
        if not self._armed:
            self._clear += 1
            if self._clear >= self.rearm_clear_frames:
                self._armed = True
                self._clear = 0
        return False

    @property
    def positive_count(self) -> int:
        """Current confirmation progress for diagnostics only."""

        return int(self._positive)

    @property
    def armed(self) -> bool:
        """Whether a new event may currently be emitted."""

        return bool(self._armed)


__all__ = [
    "MODE_GENERAL",
    "MODE_SHORTCUT",
    "MODE_OVERTAKE",
    "MODE_CONE",
    "MODES",
    "ModeSnapshot",
    "CnnModeController",
    "ConfirmedOneShot",
]
