"""White-boundary LiDAR filtering and conservative in-road trigger evidence.

The model channel and the trigger deliberately use two different masks:

* model input follows the training-time outside-white removal policy;
* OVERTAKE evidence must be at least 10 cm inside both white boundaries and
  in the 0 < x <= 1.4 m forward window.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .lidar_geometry import GRID_H, GRID_W, RESOLUTION, X_MAX, Y_MAX


MAX_GROUP_GAP_M = 0.038
MIN_PATH_TO_BOUNDARY_M = 0.04
CONTINUITY_RADIUS_ROWS = 2
MIN_CONTINUITY_SUPPORT = 3
MAX_LOCAL_EDGE_SPREAD_M = 0.18
MAX_INTERPOLATION_GAP_ROWS = 20
MAX_EDGE_EXTRAPOLATION_ROWS = 12


@dataclass(frozen=True)
class RoadLidarResult:
    filtered_points: np.ndarray
    trigger_detected: bool
    trigger_cluster_points: int
    trigger_inside_points: int
    trigger_cluster_centroid_x_m: float | None
    trigger_cluster_centroid_y_m: float | None
    trigger_reference_y_m: float | None
    trigger_lateral_offset_m: float | None
    trigger_side_ambiguous: bool
    reliable_rows: int
    removed_points: int


def _grid_xy(row: int, col: int) -> tuple[float, float]:
    return (
        X_MAX - (int(row) + 0.5) * RESOLUTION,
        Y_MAX - (int(col) + 0.5) * RESOLUTION,
    )


def _path_reference_by_row(x_values: Any, y_values: Any) -> np.ndarray:
    x = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y = np.asarray(y_values, dtype=np.float64).reshape(-1)
    reference = np.full(GRID_H, np.nan, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return reference
    order = np.argsort(x)
    x, y = x[order], y[order]
    x, indices = np.unique(x, return_index=True)
    y = y[indices]
    if x.size < 2:
        return reference
    row_x = X_MAX - (np.arange(GRID_H, dtype=np.float64) + 0.5) * RESOLUTION
    inside = (row_x >= x[0] - 1e-9) & (row_x <= x[-1] + 1e-9)
    reference[inside] = np.interp(row_x[inside], x, y)
    return reference


def reference_from_mid_grid(mid_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fallback road reference from the yellow occupancy channel."""

    grid = np.asarray(mid_grid)
    if grid.shape != (GRID_H, GRID_W):
        raise ValueError("mid grid shape mismatch")
    points: list[tuple[float, float]] = []
    for row in range(GRID_H):
        cols = np.flatnonzero(grid[row])
        if cols.size:
            x, _ = _grid_xy(row, 0)
            y_values = Y_MAX - (cols.astype(np.float64) + 0.5) * RESOLUTION
            points.append((x, float(np.median(y_values))))
    if len(points) < 2:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    points.sort()
    return (
        np.asarray([item[0] for item in points], dtype=np.float32),
        np.asarray([item[1] for item in points], dtype=np.float32),
    )


def _groups(values: np.ndarray) -> list[tuple[float, float]]:
    if values.size == 0:
        return []
    ordered = np.unique(np.round(values.astype(np.float64), 6))
    groups: list[list[float]] = [[float(ordered[0])]]
    for value in ordered[1:]:
        item = float(value)
        if item - groups[-1][-1] <= MAX_GROUP_GAP_M:
            groups[-1].append(item)
        else:
            groups.append([item])
    return [(group[0], group[-1]) for group in groups]


def _continuity_filter(raw: np.ndarray) -> np.ndarray:
    reliable = np.full_like(raw, np.nan)
    for row, value in enumerate(raw):
        if not np.isfinite(value):
            continue
        start = max(0, row - CONTINUITY_RADIUS_ROWS)
        stop = min(GRID_H, row + CONTINUITY_RADIUS_ROWS + 1)
        neighbors = raw[start:stop]
        neighbors = neighbors[np.isfinite(neighbors)]
        if neighbors.size < MIN_CONTINUITY_SUPPORT:
            continue
        median = float(np.median(neighbors))
        if float(np.ptp(neighbors)) > MAX_LOCAL_EDGE_SPREAD_M:
            continue
        if abs(float(value) - median) > MAX_LOCAL_EDGE_SPREAD_M:
            continue
        reliable[row] = median
    return reliable


def _extend(reliable: np.ndarray) -> np.ndarray:
    output = reliable.copy()
    indices = np.flatnonzero(np.isfinite(reliable))
    if indices.size == 0:
        return output
    for first, second in zip(indices[:-1], indices[1:]):
        gap = int(second - first - 1)
        if 0 < gap <= MAX_INTERPOLATION_GAP_ROWS:
            output[first + 1 : second] = np.linspace(
                reliable[first], reliable[second], gap + 2
            )[1:-1]
    first, last = int(indices[0]), int(indices[-1])
    output[max(0, first - MAX_EDGE_EXTRAPOLATION_ROWS) : first] = reliable[first]
    output[last + 1 : min(GRID_H, last + MAX_EDGE_EXTRAPOLATION_ROWS + 1)] = reliable[last]
    return output


def lane_edges(
    white_grid: np.ndarray,
    reference_x: Any,
    reference_y: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return continuous left/right outer white edges for every BEV row."""

    grid = np.asarray(white_grid)
    if grid.shape != (GRID_H, GRID_W):
        raise ValueError("white grid shape mismatch")
    reference = _path_reference_by_row(reference_x, reference_y)
    raw_left = np.full(GRID_H, np.nan, dtype=np.float64)
    raw_right = np.full(GRID_H, np.nan, dtype=np.float64)
    for row in range(GRID_H):
        center = reference[row]
        cols = np.flatnonzero(grid[row])
        if not np.isfinite(center) or cols.size == 0:
            continue
        ys = Y_MAX - (cols.astype(np.float64) + 0.5) * RESOLUTION
        groups = _groups(ys)
        left = [group for group in groups if group[0] > center + MIN_PATH_TO_BOUNDARY_M]
        right = [group for group in groups if group[1] < center - MIN_PATH_TO_BOUNDARY_M]
        if left:
            raw_left[row] = min(left, key=lambda group: group[0] - center)[1]
        if right:
            raw_right[row] = min(right, key=lambda group: center - group[1])[0]
    return _extend(_continuity_filter(raw_left)), _extend(
        _continuity_filter(raw_right)
    )


def _point_rows(points: np.ndarray) -> np.ndarray:
    return np.floor((X_MAX - points[:, 0]) / RESOLUTION).astype(np.int32)


def _clusters(points: np.ndarray, *, gap_base_m: float, gap_per_m: float):
    if points.size == 0:
        return []
    clusters: list[list[np.ndarray]] = [[points[0]]]
    for point in points[1:]:
        previous = clusters[-1][-1]
        radius = max(float(np.linalg.norm(point)), float(np.linalg.norm(previous)))
        gap = float(gap_base_m) + float(gap_per_m) * radius
        if float(np.linalg.norm(point - previous)) <= gap:
            clusters[-1].append(point)
        else:
            clusters.append([point])
    return [np.asarray(cluster, dtype=np.float32) for cluster in clusters]


def filter_and_detect_in_road(
    raw_points: Any,
    white_grid: np.ndarray,
    reference_x: Any,
    reference_y: Any,
    *,
    model_safety_margin_m: float = 0.025,
    model_white_guard_m: float = 0.025,
    trigger_inward_margin_m: float = 0.10,
    trigger_white_guard_m: float = 0.05,
    trigger_x_min_m: float = 0.0,
    trigger_x_max_m: float = 1.40,
    trigger_min_points: int = 3,
    trigger_min_inside_ratio: float = 0.50,
    trigger_min_road_width_m: float = 0.35,
    trigger_max_road_width_m: float = 1.50,
    trigger_lane_deadband_m: float = 0.10,
    cluster_gap_base_m: float = 0.15,
    cluster_gap_per_m: float = 0.03,
    max_cluster_extent_m: float = 0.90,
) -> RoadLidarResult:
    points = np.asarray(raw_points, dtype=np.float32)
    if points.size == 0:
        points = np.empty((0, 2), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("raw_points must have shape (N, 2)")
    scalars = (
        model_safety_margin_m,
        model_white_guard_m,
        trigger_inward_margin_m,
        trigger_white_guard_m,
        trigger_x_min_m,
        trigger_x_max_m,
        trigger_min_inside_ratio,
        trigger_min_road_width_m,
        trigger_max_road_width_m,
        trigger_lane_deadband_m,
        cluster_gap_base_m,
        cluster_gap_per_m,
        max_cluster_extent_m,
    )
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ValueError("road LiDAR parameters must be finite")
    if not 0.0 <= trigger_min_inside_ratio <= 1.0:
        raise ValueError("trigger_min_inside_ratio must be within [0, 1]")
    if trigger_x_min_m < 0.0 or trigger_x_max_m <= trigger_x_min_m:
        raise ValueError("trigger x window must satisfy 0 <= min < max")
    if trigger_lane_deadband_m <= 0.0:
        raise ValueError("trigger_lane_deadband_m must be positive")
    if int(trigger_min_points) < 1:
        raise ValueError("trigger_min_points must be positive")

    left, right = lane_edges(white_grid, reference_x, reference_y)
    rows = _point_rows(points) if points.size else np.empty(0, np.int32)
    keep = np.ones(points.shape[0], dtype=bool)
    for index, (point, row) in enumerate(zip(points, rows)):
        if row < 0 or row >= GRID_H:
            continue
        y = float(point[1])
        left_value, right_value = left[row], right[row]
        touches_white = (
            (np.isfinite(left_value) and abs(y - left_value) <= model_white_guard_m)
            or (np.isfinite(right_value) and abs(y - right_value) <= model_white_guard_m)
        )
        if touches_white:
            continue
        if np.isfinite(left_value) and y > left_value + model_safety_margin_m:
            keep[index] = False
        elif np.isfinite(right_value) and y < right_value - model_safety_margin_m:
            keep[index] = False
    filtered = np.ascontiguousarray(points[keep], dtype=np.float32)

    front: list[np.ndarray] = []
    for point in filtered:
        x, y = float(point[0]), float(point[1])
        if not trigger_x_min_m < x <= trigger_x_max_m:
            continue
        row = int(math.floor((X_MAX - x) / RESOLUTION))
        if not 0 <= row < GRID_H or not (
            np.isfinite(left[row]) and np.isfinite(right[row])
        ):
            continue
        width = float(left[row] - right[row])
        if not trigger_min_road_width_m <= width <= trigger_max_road_width_m:
            continue
        # Keep line-overlapping points in the broad component.  They increase
        # the denominator but cannot by themselves satisfy strict evidence.
        if right[row] - model_safety_margin_m <= y <= left[row] + model_safety_margin_m:
            front.append(point)
    front_values = (
        np.asarray(front, dtype=np.float32).reshape(-1, 2)
        if front
        else np.empty((0, 2), dtype=np.float32)
    )

    best_cluster = 0
    best_inside = 0
    detected = False
    selected_centroid_x: float | None = None
    selected_centroid_y: float | None = None
    selected_reference_y: float | None = None
    selected_lateral_offset: float | None = None
    selected_cluster_points = 0
    selected_inside_points = 0
    qualifying_offsets: list[float] = []
    reference_by_row = _path_reference_by_row(reference_x, reference_y)
    # Choose one deterministic qualifying cluster.  More strict in-road
    # points win; ties prefer the closer cluster.  This is also the cluster
    # used to classify lane 1/2 for the production hardcoded block.
    selected_rank: tuple[int, float, float] | None = None
    for cluster in _clusters(
        front_values,
        gap_base_m=cluster_gap_base_m,
        gap_per_m=cluster_gap_per_m,
    ):
        if cluster.shape[0] < int(trigger_min_points):
            continue
        extent = float(np.linalg.norm(np.ptp(cluster, axis=0)))
        if extent > max_cluster_extent_m:
            continue
        strict = 0
        strict_offsets: list[float] = []
        for x, y in cluster:
            row = int(math.floor((X_MAX - float(x)) / RESOLUTION))
            if not 0 <= row < GRID_H:
                continue
            if min(abs(float(y) - left[row]), abs(float(y) - right[row])) <= trigger_white_guard_m:
                continue
            if (
                float(y) <= left[row] - trigger_inward_margin_m
                and float(y) >= right[row] + trigger_inward_margin_m
            ):
                strict += 1
                center_y = reference_by_row[row]
                if np.isfinite(center_y):
                    strict_offsets.append(float(y) - float(center_y))
        best_cluster = max(best_cluster, int(cluster.shape[0]))
        best_inside = max(best_inside, strict)
        ratio = strict / float(cluster.shape[0])
        if strict >= int(trigger_min_points) and ratio >= trigger_min_inside_ratio:
            centroid = np.median(cluster, axis=0)
            offset = (
                float(np.median(np.asarray(strict_offsets, dtype=np.float64)))
                if strict_offsets
                else None
            )
            reference_y = (
                float(centroid[1]) - offset if offset is not None else None
            )
            rank = (strict, ratio, -float(centroid[0]))
            if selected_rank is None or rank > selected_rank:
                selected_rank = rank
                selected_centroid_x = float(centroid[0])
                selected_centroid_y = float(centroid[1])
                selected_reference_y = reference_y
                selected_lateral_offset = offset
                selected_cluster_points = int(cluster.shape[0])
                selected_inside_points = int(strict)
            if offset is not None:
                qualifying_offsets.append(offset)
            detected = True

    side_ambiguous = bool(
        any(value >= trigger_lane_deadband_m for value in qualifying_offsets)
        and any(value <= -trigger_lane_deadband_m for value in qualifying_offsets)
    )
    if side_ambiguous:
        # Detection remains true for the legacy CNN fallback, but no
        # directional hardcoded block may be selected from conflicting sides.
        selected_centroid_x = None
        selected_centroid_y = None
        selected_reference_y = None
        selected_lateral_offset = None

    reliable = int(np.count_nonzero(np.isfinite(left) & np.isfinite(right)))
    return RoadLidarResult(
        filtered_points=filtered,
        trigger_detected=detected,
        trigger_cluster_points=(selected_cluster_points if detected else best_cluster),
        trigger_inside_points=(selected_inside_points if detected else best_inside),
        trigger_cluster_centroid_x_m=selected_centroid_x,
        trigger_cluster_centroid_y_m=selected_centroid_y,
        trigger_reference_y_m=selected_reference_y,
        trigger_lateral_offset_m=selected_lateral_offset,
        trigger_side_ambiguous=side_ambiguous,
        reliable_rows=reliable,
        removed_points=int(points.shape[0] - filtered.shape[0]),
    )


__all__ = [
    "RoadLidarResult",
    "reference_from_mid_grid",
    "lane_edges",
    "filter_and_detect_in_road",
]
