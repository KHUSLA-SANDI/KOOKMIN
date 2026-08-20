import math

import numpy as np
import pytest

from track_drive_cnn_gpt.lidar_geometry import (
    GRID_H,
    GRID_W,
    RESOLUTION,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    occupied_cell_count,
    points_to_grid,
    scan_to_grid,
    scan_to_points,
)


def test_grid_axes_and_three_by_three_hit():
    grid = points_to_grid(np.array([[1.0, 0.0]], np.float32), radius_cells=1)
    assert grid.shape == (GRID_H, GRID_W)
    # row=(3-1)/.025=80, col=(1.5-0)/.025=60
    assert int(grid[79:82, 59:62].sum()) == 9
    assert int(grid.sum()) == 9
    assert occupied_cell_count(grid) == 9


def test_scan_scale_and_body_mask():
    # beam 0: forward 1 m -> calibrated 1.1476 m and retained
    # beam 1: 90 degrees, raw .1 m -> inside body and removed
    points = scan_to_points(
        [1.0, 0.1],
        0.0,
        math.pi / 2,
        range_scale=1.1476,
    )
    assert points.shape == (1, 2)
    assert np.allclose(points[0], [1.1476, 0.0], atol=1e-4)


def test_out_of_grid_points_do_not_wrap():
    grid = points_to_grid([[-0.21, 0.0], [3.01, 0.0], [1.0, 1.51]])
    assert not grid.any()


def test_grid_contract_boundaries_and_axis_directions():
    assert (GRID_H, GRID_W) == (128, 120)
    near_right = points_to_grid(
        [[X_MIN + RESOLUTION / 2, Y_MIN + RESOLUTION / 2]],
        radius_cells=0,
    )
    far_left = points_to_grid([[X_MAX, Y_MAX]], radius_cells=0)
    assert near_right[GRID_H - 1, GRID_W - 1] == 1
    assert far_left[0, 0] == 1

    positive_y = np.argwhere(points_to_grid([[1.0, 0.5]], radius_cells=0))[0]
    negative_y = np.argwhere(points_to_grid([[1.0, -0.5]], radius_cells=0))[0]
    assert positive_y[1] < negative_y[1]


def test_nonfinite_ranges_are_dropped_without_creating_hits():
    points = scan_to_points(
        [1.0, np.nan, np.inf, -np.inf],
        0.0,
        0.1,
        range_scale=1.0,
    )
    assert points.shape == (1, 2)
    grid, raster_points = scan_to_grid(
        [1.0, np.nan, np.inf],
        0.0,
        0.1,
        range_scale=1.0,
        radius_cells=0,
    )
    assert raster_points.shape == (1, 2)
    assert int(grid.sum()) == 1


def test_side_return_outside_configured_body_mask_is_retained():
    points = scan_to_points(
        [0.25],
        math.pi / 2,
        0.1,
        range_scale=1.0,
        self_x_abs_m=0.25,
    )
    assert points.shape == (1, 2)
    assert points[0, 1] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "positional,options",
    [
        (([1.0], 0.0, 0.0), {}),
        (([1.0], np.nan, 0.1), {}),
        (([1.0], 0.0, 0.1), {"range_scale": 0.0}),
        (([1.0], 0.0, 0.1), {"min_range_m": -0.1}),
        (([1.0], 0.0, 0.1), {"min_range_m": 1.0, "max_range_m": 1.0}),
        (([1.0], 0.0, 0.1), {"self_x_abs_m": -0.1}),
        (([[1.0]], 0.0, 0.1), {}),
    ],
)
def test_invalid_scan_contract_fails_closed(positional, options):
    with pytest.raises(ValueError):
        scan_to_points(*positional, **options)


@pytest.mark.parametrize("radius", [-1, 0.5, np.nan])
def test_invalid_raster_radius_fails_closed(radius):
    with pytest.raises(ValueError):
        points_to_grid([[1.0, 0.0]], radius_cells=radius)


@pytest.mark.parametrize("points", [[[1.0, 0.0, 2.0]], [1.0], [[1.0], [2.0]]])
def test_malformed_point_shapes_fail_closed(points):
    with pytest.raises(ValueError):
        points_to_grid(points)


def test_occupied_cell_count_rejects_non_binary_or_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        occupied_cell_count(np.zeros((10, 10), dtype=np.uint8))
    grid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
    grid[0, 0] = 2
    with pytest.raises(ValueError, match="binary"):
        occupied_cell_count(grid)
