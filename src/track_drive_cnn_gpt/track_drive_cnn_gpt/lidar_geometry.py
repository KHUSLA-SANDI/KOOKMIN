"""LiDAR scan to CNN occupancy conversion.

This file is deliberately ROS-free so the exact same conversion can be used
by offline dataset checks and by the ROS node.  Coordinates are metres in
``lidar_frame``: x forward, y left.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from .path_contract import (
    GRID_RESOLUTION_M,
    GRID_X_BOUNDS_M,
    GRID_Y_BOUNDS_M,
)

X_MIN, X_MAX = GRID_X_BOUNDS_M
Y_MIN, Y_MAX = GRID_Y_BOUNDS_M
RESOLUTION = GRID_RESOLUTION_M
GRID_H = int(round((X_MAX - X_MIN) / RESOLUTION))
GRID_W = int(round((Y_MAX - Y_MIN) / RESOLUTION))


def _finite_float(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def occupied_cell_count(grid: np.ndarray) -> int:
    """Validate one LiDAR occupancy grid and return its occupied-cell count."""

    values = np.asarray(grid)
    if values.shape != (GRID_H, GRID_W):
        raise ValueError(f"lidar grid shape {values.shape} != {(GRID_H, GRID_W)}")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("lidar grid must be binary 0/1")
    return int(np.count_nonzero(values))


def scan_to_points(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    *,
    range_scale: float = 1.1476,
    min_range_m: float = 0.05,
    max_range_m: float = 8.0,
    self_x_abs_m: float = 0.25,
    self_y_abs_m: float = 0.15,
) -> np.ndarray:
    """Convert one LaserScan payload into filtered calibrated xy points."""

    angle_start = _finite_float(angle_min, "angle_min")
    angle_step = _finite_float(angle_increment, "angle_increment")
    scale = _finite_float(range_scale, "range_scale")
    range_low = _finite_float(min_range_m, "min_range_m")
    range_high = _finite_float(max_range_m, "max_range_m")
    body_x = _finite_float(self_x_abs_m, "self_x_abs_m")
    body_y = _finite_float(self_y_abs_m, "self_y_abs_m")
    if angle_step == 0.0:
        raise ValueError("angle_increment must be non-zero")
    if scale <= 0.0:
        raise ValueError("range_scale must be positive")
    if range_low < 0.0 or range_high <= range_low:
        raise ValueError("range limits must satisfy 0 <= min < max")
    if body_x < 0.0 or body_y < 0.0:
        raise ValueError("self mask dimensions must be non-negative")

    try:
        raw = np.asarray(ranges, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("ranges must be a one-dimensional numeric sequence") from exc
    if raw.ndim != 1:
        raise ValueError("ranges must be a one-dimensional numeric sequence")
    angles = angle_start + np.arange(raw.size, dtype=np.float64) * angle_step
    calibrated = raw * scale
    finite = np.isfinite(calibrated)
    valid = finite & (calibrated >= range_low) & (calibrated <= range_high)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float32)

    radius = calibrated[valid]
    theta = angles[valid]
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Returns from the lidar mount/body are not obstacles.  The mask only
    # removes only the configured central vehicle/self-return box. The training
    # ego footprint is 0.50 x 0.30 m, hence half extents 0.25 x 0.15 m.
    outside_body = ~(
        (np.abs(x) < body_x) & (np.abs(y) < body_y)
    )
    points = np.stack((x[outside_body], y[outside_body]), axis=1)
    return np.ascontiguousarray(points, dtype=np.float32)


def points_to_grid(
    points: Iterable[Sequence[float]], *, radius_cells: int = 1
) -> np.ndarray:
    """Rasterize xy hits into the fixed 128x120 binary CNN channel.

    ``radius_cells=1`` exactly matches the generated obstacle dataset: each
    physical hit becomes a clipped 3x3 square.
    """

    try:
        radius_value = _finite_float(radius_cells, "radius_cells")
    except ValueError as exc:
        raise ValueError("radius_cells must be a non-negative integer") from exc
    if radius_value < 0.0 or not radius_value.is_integer():
        raise ValueError("radius_cells must be a non-negative integer")
    radius = int(radius_value)

    grid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
    source = points if isinstance(points, np.ndarray) else list(points)
    try:
        point_array = np.asarray(source, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("points must contain numeric (x, y) pairs") from exc
    if point_array.size == 0:
        return grid
    if point_array.ndim == 1 and point_array.shape == (2,):
        point_array = point_array.reshape(1, 2)
    elif point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    finite = np.isfinite(point_array).all(axis=1)
    point_array = point_array[finite]
    if point_array.size == 0:
        return grid

    rows = np.floor((X_MAX - point_array[:, 0]) / RESOLUTION).astype(np.int32)
    cols = np.floor((Y_MAX - point_array[:, 1]) / RESOLUTION).astype(np.int32)
    keep = (
        (rows >= 0)
        & (rows < GRID_H)
        & (cols >= 0)
        & (cols < GRID_W)
    )
    for row, col in zip(rows[keep], cols[keep]):
        r0, r1 = max(0, row - radius), min(GRID_H, row + radius + 1)
        c0, c1 = max(0, col - radius), min(GRID_W, col + radius + 1)
        grid[r0:r1, c0:c1] = 1
    return grid


def scan_to_grid(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    *,
    radius_cells: int = 1,
    **point_options,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(binary_grid, filtered_xy_points)`` for diagnostics."""

    points = scan_to_points(
        ranges, angle_min, angle_increment, **point_options
    )
    return points_to_grid(points, radius_cells=radius_cells), points
