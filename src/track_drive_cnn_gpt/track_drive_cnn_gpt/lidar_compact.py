"""Strict obstacle-only LiDAR compaction for the path CNN.

The path training set does *not* contain a raw 360 degree LaserScan.  Clean
samples have an empty LiDAR channel and obstacle samples contain one sparse
set of visible returns from a 0.50 x 0.30 m box or a 0.60 x 0.30 m vehicle.
Consequently this module only accepts one compact cluster close to a
camera-only reference path.  Ambiguous input is rejected instead of being
silently erased or passed to the network out of distribution.

This file is ROS-free so recorded scans and the runtime node use exactly the
same policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import numpy as np

from .lidar_geometry import X_MAX, X_MIN, Y_MAX, Y_MIN


@dataclass(frozen=True)
class CompactLidarResult:
    """Result of matching one raw scan to one camera-only reference path."""

    usable: bool
    reason: str
    points: np.ndarray
    raw_fov_points: int
    corridor_points: int
    corridor_clusters: int
    selected_raw_points: int
    selected_extent_m: float

    @property
    def obstacle_present(self) -> bool:
        return bool(self.points.shape[0])


def _empty_result(
    *,
    usable: bool,
    reason: str,
    raw_fov_points: int = 0,
    corridor_points: int = 0,
    corridor_clusters: int = 0,
    selected_raw_points: int = 0,
    selected_extent_m: float = 0.0,
) -> CompactLidarResult:
    return CompactLidarResult(
        usable=usable,
        reason=reason,
        points=np.empty((0, 2), dtype=np.float32),
        raw_fov_points=int(raw_fov_points),
        corridor_points=int(corridor_points),
        corridor_clusters=int(corridor_clusters),
        selected_raw_points=int(selected_raw_points),
        selected_extent_m=float(selected_extent_m),
    )


def _finite_positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(parsed) or parsed < 0.0 or (parsed == 0.0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return parsed


def _positive_integer(value: int, name: str) -> int:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(parsed) or parsed < 1.0 or not parsed.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(parsed)


def _points_array(points_xy: Sequence[Sequence[float]]) -> np.ndarray:
    try:
        points = np.asarray(points_xy, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("points_xy must have shape (N, 2)") from exc
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2)")
    if not np.isfinite(points).all():
        raise ValueError("points_xy must contain only finite values")
    return points


def _reference_arrays(
    reference_x: Sequence[float], reference_y: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        x = np.asarray(reference_x, dtype=np.float64).reshape(-1)
        y = np.asarray(reference_y, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("reference path must be numeric") from exc
    if x.size < 2 or x.size != y.size:
        raise ValueError("reference path must contain matching x/y arrays of length >= 2")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("reference path must contain only finite values")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("reference path x must be strictly increasing")
    return x, y


def _cluster_ordered_points(
    points: np.ndarray, *, gap_base_m: float, gap_per_m: float
) -> list[np.ndarray]:
    """Split scan-ordered points at physical discontinuities.

    ``scan_to_points`` preserves LaserScan beam order.  A range-dependent gap
    handles the increasing arc distance of far beams without importing the
    very permissive legacy planner clustering threshold.
    """

    if points.shape[0] == 0:
        return []
    if points.shape[0] == 1:
        return [points]
    radius = np.linalg.norm(points, axis=1)
    distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
    allowed = gap_base_m + gap_per_m * np.minimum(radius[:-1], radius[1:])
    split_at = np.flatnonzero(distance > allowed) + 1
    return [group for group in np.split(points, split_at) if group.shape[0]]


def _diameter(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    delta = points[:, None, :] - points[None, :, :]
    return float(np.linalg.norm(delta, axis=2).max())


def _training_count_cap(nearest_m: float) -> int:
    """Upper hit count observed from the obstacle synthesizer by range."""

    if nearest_m < 0.75:
        return 14
    if nearest_m < 1.00:
        return 8
    if nearest_m < 1.80:
        return 6
    return 5


def _training_count_floor(nearest_m: float) -> int:
    if nearest_m < 0.75:
        return 8
    if nearest_m < 1.00:
        return 4
    return 3


def _stratified_points(points: np.ndarray, target: int) -> np.ndarray:
    if points.shape[0] <= target:
        return np.ascontiguousarray(points, dtype=np.float32)
    positions = np.linspace(0, points.shape[0] - 1, target)
    indices = np.unique(np.rint(positions).astype(np.int32))
    return np.ascontiguousarray(points[indices], dtype=np.float32)


def compact_obstacle_points(
    points_xy: Sequence[Sequence[float]],
    reference_x: Sequence[float],
    reference_y: Sequence[float],
    *,
    fov_deg: float = 110.0,
    corridor_half_width_m: float = 0.70,
    cluster_gap_base_m: float = 0.12,
    cluster_gap_per_m: float = 0.03,
    min_cluster_points: int = 3,
    max_cluster_extent_m: float = 0.72,
    max_clusters: int = 1,
) -> CompactLidarResult:
    """Extract at most one training-compatible obstacle cluster.

    A camera-only CNN prediction supplies ``reference_x/reference_y``.  Raw
    points outside its planning corridor are background for this model.  Any
    cluster touching the corridor that is too sparse, too large, or one of
    multiple clusters makes the result unusable (fail closed).
    """

    points = _points_array(points_xy)
    ref_x, ref_y = _reference_arrays(reference_x, reference_y)
    fov = _finite_positive(fov_deg, "fov_deg")
    corridor = _finite_positive(corridor_half_width_m, "corridor_half_width_m")
    gap_base = _finite_positive(cluster_gap_base_m, "cluster_gap_base_m")
    gap_per_m = _finite_positive(
        cluster_gap_per_m, "cluster_gap_per_m", allow_zero=True
    )
    min_points = _positive_integer(min_cluster_points, "min_cluster_points")
    max_extent = _finite_positive(max_cluster_extent_m, "max_cluster_extent_m")
    cluster_limit = _positive_integer(max_clusters, "max_clusters")
    if fov > 180.0:
        raise ValueError("fov_deg must not exceed 180")

    if points.shape[0] == 0:
        return _empty_result(usable=True, reason="no_lidar_returns")

    angles_deg = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    inside = (
        (np.abs(angles_deg) <= fov)
        & (points[:, 0] >= X_MIN)
        & (points[:, 0] <= X_MAX)
        & (points[:, 1] >= Y_MIN)
        & (points[:, 1] <= Y_MAX)
    )
    fov_points = points[inside]
    raw_fov_count = int(fov_points.shape[0])
    if raw_fov_count == 0:
        return _empty_result(usable=True, reason="no_lidar_returns_in_bev_fov")

    groups = _cluster_ordered_points(
        fov_points, gap_base_m=gap_base, gap_per_m=gap_per_m
    )
    corridor_groups: list[np.ndarray] = []
    corridor_point_count = 0
    for group in groups:
        expected_y = np.interp(
            group[:, 0], ref_x, ref_y, left=ref_y[0], right=ref_y[-1]
        )
        touches = np.abs(group[:, 1] - expected_y) <= corridor
        if np.any(touches):
            corridor_groups.append(group)
            corridor_point_count += int(np.count_nonzero(touches))

    group_count = len(corridor_groups)
    if group_count == 0:
        return _empty_result(
            usable=True,
            reason="no_corridor_obstacle",
            raw_fov_points=raw_fov_count,
        )
    if group_count > cluster_limit:
        return _empty_result(
            usable=False,
            reason="multiple_corridor_clusters",
            raw_fov_points=raw_fov_count,
            corridor_points=corridor_point_count,
            corridor_clusters=group_count,
        )

    selected = corridor_groups[0]
    extent = _diameter(selected)
    nearest = float(np.linalg.norm(selected, axis=1).min())
    floor = max(min_points, _training_count_floor(nearest))
    if selected.shape[0] < floor:
        return _empty_result(
            usable=False,
            reason="sparse_corridor_cluster",
            raw_fov_points=raw_fov_count,
            corridor_points=corridor_point_count,
            corridor_clusters=group_count,
            selected_raw_points=int(selected.shape[0]),
            selected_extent_m=extent,
        )
    if extent > max_extent:
        return _empty_result(
            usable=False,
            reason="long_corridor_surface",
            raw_fov_points=raw_fov_count,
            corridor_points=corridor_point_count,
            corridor_clusters=group_count,
            selected_raw_points=int(selected.shape[0]),
            selected_extent_m=extent,
        )

    compact = _stratified_points(selected, _training_count_cap(nearest))
    return CompactLidarResult(
        usable=True,
        reason="compact_corridor_obstacle",
        points=compact,
        raw_fov_points=raw_fov_count,
        corridor_points=corridor_point_count,
        corridor_clusters=group_count,
        selected_raw_points=int(selected.shape[0]),
        selected_extent_m=extent,
    )


__all__ = ["CompactLidarResult", "compact_obstacle_points"]
