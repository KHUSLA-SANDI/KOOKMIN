"""Pure helpers for same-pass YOLO signals and route-intent latching.

The production segmentation model contains the four signal classes together
with ``lane`` and ``mid``.  Keeping extraction here makes it testable without
ROS or an Ultralytics installation and guarantees that signal handling never
runs a second model inference.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


SIGNAL_SCHEMA = "yolo_signal_v1_gpt"
SIGNAL_NAMES = ("GREEN", "LEFT", "RED", "YELLOW")
ROUTE_MAIN = "main"
ROUTE_SHORTCUT = "shortcut"


def optional_class_id(names_or_model: Any, class_name: str) -> int | None:
    """Resolve one optional model class by case-insensitive exact name."""

    desired = str(class_name).strip().upper()
    if not desired:
        return None
    names = getattr(names_or_model, "names", names_or_model)
    if isinstance(names, (list, tuple)):
        pairs = enumerate(names)
    elif isinstance(names, Mapping):
        pairs = names.items()
    else:
        raise TypeError("class names must be a mapping/list or an object with .names")
    for index, name in pairs:
        if str(name).strip().upper() == desired:
            return int(index)
    return None


def select_route_value(route_intent: str, main: Any, shortcut: Any) -> Any:
    """Return exactly the requested route without an implicit fallback."""

    normalized = str(route_intent).strip().lower()
    if normalized == ROUTE_MAIN:
        return main
    if normalized == ROUTE_SHORTCUT:
        return shortcut
    raise ValueError(f"unsupported route intent: {route_intent!r}")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def signal_class_ids(names_or_model: Any) -> dict[str, int]:
    """Resolve all required signal class IDs case-insensitively.

    Failing closed is intentional.  A lane-only checkpoint must not appear to
    work while silently removing the mission signals.
    """

    names = getattr(names_or_model, "names", names_or_model)
    if isinstance(names, (list, tuple)):
        pairs = enumerate(names)
    elif isinstance(names, Mapping):
        pairs = names.items()
    else:
        raise TypeError("class names must be a mapping/list or an object with .names")

    normalized = {str(name).strip().upper(): int(index) for index, name in pairs}
    missing = [name for name in SIGNAL_NAMES if name not in normalized]
    if missing:
        raise ValueError(
            "segmentation model is missing required signal classes: "
            + ", ".join(missing)
        )
    return {name: normalized[name] for name in SIGNAL_NAMES}


def extract_signal_confidences(
    result: Any,
    class_ids: Mapping[str, int],
    *,
    min_confidence: float = 0.25,
) -> dict[str, float]:
    """Return the highest confidence for every signal present in one result."""

    threshold = float(min_confidence)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("min_confidence must be within [0, 1]")
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return {}
    classes_value = getattr(boxes, "cls", None)
    confidences_value = getattr(boxes, "conf", None)
    if classes_value is None or confidences_value is None:
        return {}

    classes = _to_numpy(classes_value).astype(np.int64, copy=False).reshape(-1)
    confidences = _to_numpy(confidences_value).astype(np.float64, copy=False).reshape(-1)
    if classes.size != confidences.size:
        raise RuntimeError(
            f"YOLO returned {classes.size} class IDs but {confidences.size} confidences"
        )

    output: dict[str, float] = {}
    for name in SIGNAL_NAMES:
        class_id = int(class_ids[name])
        selected = confidences[classes == class_id]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            best = float(np.max(selected))
            if best >= threshold:
                output[name] = best
    return output


def extract_signal_detections(
    result: Any,
    class_ids: Mapping[str, int],
    *,
    image_hw: tuple[int, int],
    min_confidence: float = 0.25,
) -> dict[str, dict[str, float]]:
    """Return the best box and confidence for each signal class.

    Coordinates are normalized to the original camera image.  Keeping this
    metadata in the same-pass signal message lets mission logic distinguish a
    far-away RED/YELLOW from the same signal at the stop line without a second
    detector.
    """

    height, width = (int(image_hw[0]), int(image_hw[1]))
    if height <= 0 or width <= 0:
        raise ValueError("image_hw must contain positive height and width")
    threshold = float(min_confidence)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("min_confidence must be within [0, 1]")
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return {}
    classes_value = getattr(boxes, "cls", None)
    confidences_value = getattr(boxes, "conf", None)
    xyxy_value = getattr(boxes, "xyxy", None)
    if classes_value is None or confidences_value is None or xyxy_value is None:
        return {}

    classes = _to_numpy(classes_value).astype(np.int64, copy=False).reshape(-1)
    confidences = _to_numpy(confidences_value).astype(np.float64, copy=False).reshape(-1)
    xyxy = _to_numpy(xyxy_value).astype(np.float64, copy=False).reshape(-1, 4)
    if not (classes.size == confidences.size == xyxy.shape[0]):
        raise RuntimeError("YOLO signal box arrays have inconsistent lengths")

    output: dict[str, dict[str, float]] = {}
    for name in SIGNAL_NAMES:
        indices = np.flatnonzero(classes == int(class_ids[name]))
        if not indices.size:
            continue
        finite = indices[np.isfinite(confidences[indices])]
        if not finite.size:
            continue
        best_index = int(finite[np.argmax(confidences[finite])])
        confidence = float(confidences[best_index])
        if confidence < threshold:
            continue
        x1, y1, x2, y2 = (float(value) for value in xyxy[best_index])
        if not np.isfinite([x1, y1, x2, y2]).all():
            continue
        x1 = min(max(x1 / width, 0.0), 1.0)
        x2 = min(max(x2 / width, 0.0), 1.0)
        y1 = min(max(y1 / height, 0.0), 1.0)
        y2 = min(max(y2 / height, 0.0), 1.0)
        if x2 < x1 or y2 < y1:
            continue
        output[name] = {
            "confidence": confidence,
            "x1_norm": x1,
            "y1_norm": y1,
            "x2_norm": x2,
            "y2_norm": y2,
            "center_x_norm": 0.5 * (x1 + x2),
            "center_y_norm": 0.5 * (y1 + y2),
            "width_norm": x2 - x1,
            "height_norm": y2 - y1,
        }
    return output


def extract_class_detection(
    result: Any,
    class_id: int | None,
    *,
    image_hw: tuple[int, int],
    min_confidence: float = 0.25,
) -> dict[str, float] | None:
    """Return normalized metadata for the best instance of one class."""

    if class_id is None:
        return None
    temporary_ids = {name: -1 for name in SIGNAL_NAMES}
    temporary_ids["GREEN"] = int(class_id)
    result_map = extract_signal_detections(
        result,
        temporary_ids,
        image_hw=image_hw,
        min_confidence=min_confidence,
    )
    return result_map.get("GREEN")


@dataclass(frozen=True)
class TimedTriggerObservation:
    active: bool
    changed: bool


class TimedTriggerLatch:
    """Confirm entry for N frames and release after a detection-loss timeout."""

    def __init__(self, *, enter_confirm_frames: int = 2, exit_hold_sec: float = 1.0):
        if isinstance(enter_confirm_frames, bool) or int(enter_confirm_frames) < 1:
            raise ValueError("enter_confirm_frames must be positive")
        if not math.isfinite(float(exit_hold_sec)) or float(exit_hold_sec) <= 0.0:
            raise ValueError("exit_hold_sec must be positive and finite")
        self.enter_confirm_frames = int(enter_confirm_frames)
        self.exit_hold_sec = float(exit_hold_sec)
        self.active = False
        self.enter_streak = 0
        self.last_seen_sec = -math.inf

    def observe(self, detected: bool, *, now_sec: float) -> TimedTriggerObservation:
        now = float(now_sec)
        if not math.isfinite(now):
            raise ValueError("now_sec must be finite")
        before = self.active
        if bool(detected):
            self.last_seen_sec = now
            if not self.active:
                self.enter_streak += 1
                if self.enter_streak >= self.enter_confirm_frames:
                    self.active = True
                    self.enter_streak = 0
        else:
            if not self.active:
                self.enter_streak = 0
            elif now - self.last_seen_sec >= self.exit_hold_sec:
                self.active = False
                self.enter_streak = 0
        return TimedTriggerObservation(self.active, self.active != before)


def encode_signal_payload(
    sequence: int,
    signals: Mapping[str, float],
    detections: Mapping[str, Mapping[str, float]] | None = None,
) -> str:
    """Encode the small, versioned ``std_msgs/String`` signal contract."""

    sequence = int(sequence)
    if sequence < 0:
        raise ValueError("sequence must be non-negative")
    clean: dict[str, float] = {}
    for raw_name, raw_confidence in signals.items():
        name = str(raw_name).strip().upper()
        if name not in SIGNAL_NAMES:
            raise ValueError(f"unsupported signal name: {raw_name!r}")
        confidence = float(raw_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"invalid confidence for {name}: {raw_confidence!r}")
        clean[name] = confidence
    payload = {
        "schema_version": SIGNAL_SCHEMA,
        "sequence": sequence,
        "signals": clean,
    }
    if detections is not None:
        clean_detections: dict[str, dict[str, float]] = {}
        required = (
            "confidence",
            "x1_norm",
            "y1_norm",
            "x2_norm",
            "y2_norm",
            "center_x_norm",
            "center_y_norm",
            "width_norm",
            "height_norm",
        )
        for raw_name, raw_detection in detections.items():
            name = str(raw_name).strip().upper()
            if name not in SIGNAL_NAMES or not isinstance(raw_detection, Mapping):
                raise ValueError(f"invalid signal detection: {raw_name!r}")
            item = {key: float(raw_detection[key]) for key in required}
            if not all(math.isfinite(value) for value in item.values()):
                raise ValueError(f"non-finite detection metadata for {name}")
            if not all(0.0 <= item[key] <= 1.0 for key in required):
                raise ValueError(f"detection metadata outside [0, 1] for {name}")
            if item["x2_norm"] < item["x1_norm"] or item["y2_norm"] < item["y1_norm"]:
                raise ValueError(f"inverted signal box for {name}")
            clean_detections[name] = item
        payload["detections"] = clean_detections
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def decode_signal_payload(encoded: str) -> tuple[int, dict[str, float]]:
    """Decode and strictly validate a signal payload."""

    try:
        payload = json.loads(str(encoded))
    except json.JSONDecodeError as exc:
        raise ValueError("signal payload is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SIGNAL_SCHEMA:
        raise ValueError("unsupported signal payload schema")
    if "sequence" not in payload or "signals" not in payload:
        raise ValueError("signal payload is missing sequence or signals")
    sequence = payload["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("signal sequence must be a non-negative integer")
    signals = payload["signals"]
    if not isinstance(signals, dict):
        raise ValueError("signals must be an object")
    # Reuse the encoder's validation, then return ordinary floats.
    normalized = json.loads(encode_signal_payload(sequence, signals))["signals"]
    return sequence, {str(name): float(value) for name, value in normalized.items()}


def decode_signal_payload_full(
    encoded: str,
) -> tuple[int, dict[str, float], dict[str, dict[str, float]]]:
    """Decode confidences plus optional normalized box metadata."""

    sequence, signals = decode_signal_payload(encoded)
    payload = json.loads(str(encoded))
    raw_detections = payload.get("detections", {})
    if not isinstance(raw_detections, dict):
        raise ValueError("detections must be an object")
    normalized = json.loads(
        encode_signal_payload(sequence, signals, raw_detections)
    ).get("detections", {})
    return sequence, signals, {
        str(name): {str(key): float(value) for key, value in item.items()}
        for name, item in normalized.items()
    }


@dataclass(frozen=True)
class TrafficMissionObservation:
    accepted: bool
    source_restarted: bool
    route_changed: bool
    stop_changed: bool
    route_intent: str
    traffic_stop: bool
    dominant_signal: str | None
    decision_zone: bool


class TrafficMissionController:
    """Sequence-aware timed shortcut and stop-line signal controller.

    LEFT starts one fixed-duration shortcut window.  RED/YELLOW only latch a
    stop after their detected box reaches the configured image zone.  Once
    stopped, a confirmed GREEN or LEFT in that same zone releases the stop;
    detection loss alone never releases it.
    """

    def __init__(
        self,
        *,
        confirm_frames: int = 2,
        confidence: float = 0.25,
        shortcut_hold_sec: float = 10.0,
        left_confirm_window_sec: float = 1.5,
        decision_center_y_max: float = 0.22,
        decision_min_width: float = 0.08,
        left_rearm_absent_frames: int = 3,
    ) -> None:
        if isinstance(confirm_frames, bool) or int(confirm_frames) < 1:
            raise ValueError("confirm_frames must be a positive integer")
        if isinstance(left_rearm_absent_frames, bool) or int(left_rearm_absent_frames) < 1:
            raise ValueError("left_rearm_absent_frames must be positive")
        values = (
            float(confidence),
            float(shortcut_hold_sec),
            float(left_confirm_window_sec),
            float(decision_center_y_max),
            float(decision_min_width),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("traffic mission parameters must be finite")
        if not 0.0 <= values[0] <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if values[1] <= 0.0 or values[2] <= 0.0:
            raise ValueError("shortcut and LEFT confirmation times must be positive")
        if not 0.0 <= values[3] <= 1.0 or not 0.0 <= values[4] <= 1.0:
            raise ValueError("decision-zone thresholds must be within [0, 1]")
        self.confirm_frames = int(confirm_frames)
        self.confidence = values[0]
        self.shortcut_hold_sec = values[1]
        self.left_confirm_window_sec = values[2]
        self.decision_center_y_max = values[3]
        self.decision_min_width = values[4]
        self.left_rearm_absent_frames = int(left_rearm_absent_frames)
        self.last_sequence = -1
        self.left_streak = 0
        self._left_hit_times: list[float] = []
        self._left_absent_streak = 0
        self._left_armed = True
        self._shortcut_until = -math.inf
        self._decision_name: str | None = None
        self._decision_streak = 0
        self.traffic_stop = False

    def route_intent(self, now_sec: float) -> str:
        now = float(now_sec)
        if not math.isfinite(now):
            raise ValueError("now_sec must be finite")
        return ROUTE_SHORTCUT if now < self._shortcut_until else ROUTE_MAIN

    def shortcut_remaining_sec(self, now_sec: float) -> float:
        return max(0.0, self._shortcut_until - float(now_sec))

    def _dominant(self, signals: Mapping[str, float]) -> str | None:
        valid = []
        for raw_name, raw_confidence in signals.items():
            name = str(raw_name).strip().upper()
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                continue
            if name in SIGNAL_NAMES and math.isfinite(confidence) and confidence >= self.confidence:
                valid.append((confidence, name))
        return max(valid)[1] if valid else None

    def _in_decision_zone(
        self,
        name: str | None,
        detections: Mapping[str, Mapping[str, float]],
    ) -> bool:
        if name is None or name not in detections:
            return False
        item = detections[name]
        try:
            center_y = float(item["center_y_norm"])
            width = float(item["width_norm"])
        except (KeyError, TypeError, ValueError):
            return False
        return (
            math.isfinite(center_y)
            and math.isfinite(width)
            and center_y <= self.decision_center_y_max
            and width >= self.decision_min_width
        )

    def observe(
        self,
        sequence: int,
        signals: Mapping[str, float],
        detections: Mapping[str, Mapping[str, float]],
        *,
        now_sec: float,
    ) -> TrafficMissionObservation:
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("signal sequence must be a non-negative integer")
        now = float(now_sec)
        if not math.isfinite(now):
            raise ValueError("now_sec must be finite")
        before_route = self.route_intent(now)
        before_stop = self.traffic_stop
        if sequence == self.last_sequence:
            return TrafficMissionObservation(
                False, False, False, False, before_route, before_stop, None, False
            )
        source_restarted = sequence < self.last_sequence
        if source_restarted:
            self.left_streak = 0
            self._left_hit_times.clear()
            self._decision_name = None
            self._decision_streak = 0
        self.last_sequence = sequence

        dominant = self._dominant(signals)
        left_seen = dominant == "LEFT"
        oldest_left_hit = now - self.left_confirm_window_sec
        self._left_hit_times = [
            timestamp
            for timestamp in self._left_hit_times
            if timestamp >= oldest_left_hit
        ]
        if left_seen:
            self._left_absent_streak = 0
            if self._left_armed and before_route != ROUTE_SHORTCUT:
                self._left_hit_times.append(now)
                self.left_streak = len(self._left_hit_times)
                if self.left_streak >= self.confirm_frames:
                    self._shortcut_until = now + self.shortcut_hold_sec
                    self._left_armed = False
                    self.left_streak = 0
                    self._left_hit_times.clear()
        else:
            self.left_streak = len(self._left_hit_times)
            self._left_absent_streak += 1
            if self._left_absent_streak >= self.left_rearm_absent_frames:
                self._left_armed = True

        decision_zone = self._in_decision_zone(dominant, detections)
        decision_name = dominant if decision_zone else None
        if decision_name is None:
            self._decision_name = None
            self._decision_streak = 0
        else:
            if decision_name == self._decision_name:
                self._decision_streak += 1
            else:
                self._decision_name = decision_name
                self._decision_streak = 1
            if self._decision_streak >= self.confirm_frames:
                if decision_name in ("RED", "YELLOW"):
                    self._shortcut_until = -math.inf
                    self.traffic_stop = True
                elif decision_name == "GREEN":
                    self._shortcut_until = -math.inf
                    self.traffic_stop = False
                elif decision_name == "LEFT":
                    self.traffic_stop = False

        after_route = self.route_intent(now)
        return TrafficMissionObservation(
            True,
            source_restarted,
            after_route != before_route,
            self.traffic_stop != before_stop,
            after_route,
            self.traffic_stop,
            dominant,
            decision_zone,
        )

    def reset_route(self) -> str:
        self._shortcut_until = -math.inf
        self.left_streak = 0
        self._left_hit_times.clear()
        self._left_armed = True
        self.last_sequence = -1
        return ROUTE_MAIN

    def reset_stop(self) -> None:
        self.traffic_stop = False
        self._decision_name = None
        self._decision_streak = 0


class RouteIntentLatch:
    """Latch shortcut after consecutive LEFT observations until explicit reset."""

    def __init__(
        self,
        *,
        left_confirm_frames: int = 2,
        left_confidence: float = 0.25,
    ) -> None:
        if isinstance(left_confirm_frames, bool) or int(left_confirm_frames) < 1:
            raise ValueError("left_confirm_frames must be a positive integer")
        left_confidence = float(left_confidence)
        if not math.isfinite(left_confidence) or not 0.0 <= left_confidence <= 1.0:
            raise ValueError("left_confidence must be within [0, 1]")
        self.left_confirm_frames = int(left_confirm_frames)
        self.left_confidence = left_confidence
        self.route_intent = ROUTE_MAIN
        self.left_streak = 0

    def observe(self, signals: Mapping[str, float] | Iterable[str]) -> str:
        if isinstance(signals, Mapping):
            confidence = signals.get("LEFT", signals.get("left", -1.0))
            try:
                left_seen = float(confidence) >= self.left_confidence
            except (TypeError, ValueError):
                left_seen = False
        else:
            left_seen = any(str(name).strip().upper() == "LEFT" for name in signals)

        if self.route_intent == ROUTE_SHORTCUT:
            return self.route_intent
        self.left_streak = self.left_streak + 1 if left_seen else 0
        if self.left_streak >= self.left_confirm_frames:
            self.route_intent = ROUTE_SHORTCUT
        return self.route_intent

    def reset_main(self) -> str:
        self.route_intent = ROUTE_MAIN
        self.left_streak = 0
        return self.route_intent

    def reset_observation_streak(self) -> None:
        """Forget partial confirmation without changing a latched route."""

        self.left_streak = 0


@dataclass(frozen=True)
class SequencedRouteObservation:
    """Result of applying one sequence-numbered YOLO signal observation."""

    accepted: bool
    source_restarted: bool
    route_changed: bool
    route_intent: str


class SequencedRouteIntentLatch:
    """Sequence-aware wrapper around :class:`RouteIntentLatch`.

    Keeping this state machine ROS-independent prevents a duplicated DDS sample
    from counting as two LEFT confirmations.  A lower sequence denotes a YOLO
    source restart and clears only an unfinished confirmation streak; an
    already-latched shortcut remains selected until ``reset_main`` is called.
    """

    def __init__(
        self,
        *,
        left_confirm_frames: int = 2,
        left_confidence: float = 0.25,
    ) -> None:
        self._latch = RouteIntentLatch(
            left_confirm_frames=left_confirm_frames,
            left_confidence=left_confidence,
        )
        self.last_sequence = -1

    @property
    def route_intent(self) -> str:
        return self._latch.route_intent

    @property
    def left_streak(self) -> int:
        return self._latch.left_streak

    def observe(
        self,
        sequence: int,
        signals: Mapping[str, float] | Iterable[str],
    ) -> SequencedRouteObservation:
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("signal sequence must be a non-negative integer")
        if sequence == self.last_sequence:
            return SequencedRouteObservation(
                accepted=False,
                source_restarted=False,
                route_changed=False,
                route_intent=self.route_intent,
            )

        source_restarted = sequence < self.last_sequence
        if source_restarted:
            self._latch.reset_observation_streak()
        self.last_sequence = sequence
        before = self.route_intent
        after = self._latch.observe(signals)
        return SequencedRouteObservation(
            accepted=True,
            source_restarted=source_restarted,
            route_changed=after != before,
            route_intent=after,
        )

    def reset_main(self) -> str:
        self.last_sequence = -1
        return self._latch.reset_main()
