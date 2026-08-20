"""Pure helpers for adapting CNN paths to the legacy motion controller.

CNN path coordinates are already expressed in the current ``lidar_frame`` and
have fixed forward samples.  They must therefore not be re-anchored to the
nearest path point as the legacy long-horizon planner paths were.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EgoRelativePath:
    """Validated ego-relative path and the absolute-x focus coordinate."""

    x: np.ndarray
    y: np.ndarray
    focus_x: np.ndarray


def stamp_to_ns(stamp: Any) -> int:
    """Convert a ROS-like ``sec``/``nanosec`` stamp without importing ROS."""

    try:
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("path header stamp is invalid") from exc
    if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
        raise ValueError("path header stamp is invalid")
    return sec * 1_000_000_000 + nanosec


def validate_source_stamp_ns(
    source_ns: int,
    now_ns: int,
    *,
    stale_sec: float,
    future_tolerance_sec: float = 0.05,
) -> float:
    """Return source age in seconds or reject zero, stale, and future stamps."""

    try:
        source = int(source_ns)
        now = int(now_ns)
        stale = float(stale_sec)
        future_tolerance = float(future_tolerance_sec)
    except (TypeError, ValueError) as exc:
        raise ValueError("path stamp policy is invalid") from exc
    if source <= 0:
        raise ValueError("path source stamp is zero")
    if stale <= 0.0 or future_tolerance < 0.0:
        raise ValueError("path stamp policy is invalid")
    age = (now - source) * 1e-9
    if age < -future_tolerance:
        raise ValueError("path source stamp is in the future")
    if age > stale:
        raise ValueError("path source stamp is stale")
    return age


def prepare_ego_relative_path(
    x_values: Any,
    y_values: Any,
    *,
    x_min: float = 0.3,
    x_max: float = 3.0,
    minimum_points: int = 2,
    epsilon: float = 1e-4,
) -> EgoRelativePath:
    """Strictly validate a fixed-x CNN path without sorting or re-anchoring.

    The output ``focus_x`` is the absolute forward coordinate.  Supplying it as
    the legacy controller's third path vector makes lookahead, gain distance,
    and preview calculations use metres ahead of the vehicle instead of arc
    length measured from a dynamically selected anchor.
    """

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("path x/y must be same-length vectors")
    if x.size < int(minimum_points):
        raise ValueError("path has too few points")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("path contains NaN or infinity")

    lo = float(x_min)
    hi = float(x_max)
    eps = float(epsilon)
    if not np.isfinite([lo, hi, eps]).all() or lo < 0.0 or hi <= lo or eps <= 0.0:
        raise ValueError("ego path bounds are invalid")
    if np.any(x < lo - eps) or np.any(x > hi + eps):
        raise ValueError("path x is outside the CNN horizon")
    if np.any(np.diff(x) <= eps):
        raise ValueError("path x must be strictly increasing")

    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    return EgoRelativePath(x=x, y=y, focus_x=x.copy())


__all__ = [
    "EgoRelativePath",
    "prepare_ego_relative_path",
    "stamp_to_ns",
    "validate_source_stamp_ns",
]
