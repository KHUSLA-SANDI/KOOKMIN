"""Pure helpers for same-pass YOLO signals and route-intent latching.

The production segmentation model contains the four signal classes together
with ``lane`` and ``mid``.  Keeping extraction here makes it testable without
ROS or an Ultralytics installation and guarantees that signal handling never
runs a second model inference.
"""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, Mapping

import numpy as np


SIGNAL_SCHEMA = "yolo_signal_v1_gpt"
SIGNAL_NAMES = ("GREEN", "LEFT", "RED", "YELLOW")
ROUTE_MAIN = "main"
ROUTE_SHORTCUT = "shortcut"


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


def encode_signal_payload(sequence: int, signals: Mapping[str, float]) -> str:
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
