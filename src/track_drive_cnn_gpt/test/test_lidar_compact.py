import numpy as np
import pytest

from track_drive_cnn_gpt.lidar_compact import compact_obstacle_points


REF_X = np.arange(0.3, 3.01, 0.1, dtype=np.float32)
REF_Y = np.zeros_like(REF_X)


def surface(x, y0, y1, count):
    return np.column_stack(
        [np.full(count, x, dtype=np.float32), np.linspace(y0, y1, count)]
    )


def test_no_returns_or_only_outside_corridor_is_clean():
    empty = compact_obstacle_points([], REF_X, REF_Y)
    assert empty.usable
    assert empty.reason == "no_lidar_returns"
    assert not empty.obstacle_present

    outside = compact_obstacle_points(surface(1.0, 1.0, 1.2, 8), REF_X, REF_Y)
    assert outside.usable
    assert outside.reason == "no_corridor_obstacle"
    assert outside.raw_fov_points == 8


def test_single_box_face_is_stratified_to_training_count_cap():
    # At 0.6 m the synthesized training domain contains 8..14 returns.
    points = surface(0.6, -0.15, 0.15, 30)
    result = compact_obstacle_points(points, REF_X, REF_Y)
    assert result.usable
    assert result.reason == "compact_corridor_obstacle"
    assert result.selected_raw_points == 30
    assert result.points.shape == (14, 2)
    assert result.selected_extent_m == pytest.approx(0.30)


def test_far_box_face_uses_far_training_count_cap():
    result = compact_obstacle_points(
        surface(2.0, -0.15, 0.15, 20), REF_X, REF_Y
    )
    assert result.usable
    assert result.points.shape == (5, 2)


def test_multiple_obstacles_fail_closed():
    first = surface(0.8, -0.25, -0.05, 8)
    second = surface(1.6, 0.05, 0.25, 8)
    result = compact_obstacle_points(np.vstack([first, second]), REF_X, REF_Y)
    assert not result.usable
    assert result.reason == "multiple_corridor_clusters"
    assert result.corridor_clusters == 2
    assert not result.obstacle_present


def test_long_wall_touching_corridor_fails_closed():
    wall = np.column_stack(
        [np.full(60, 1.0), np.linspace(-0.6, 0.6, 60)]
    )
    result = compact_obstacle_points(wall, REF_X, REF_Y)
    assert not result.usable
    assert result.reason == "long_corridor_surface"
    assert result.selected_extent_m == pytest.approx(1.2)


def test_sparse_near_return_fails_closed_instead_of_disappearing():
    result = compact_obstacle_points(
        surface(0.6, -0.04, 0.04, 3), REF_X, REF_Y
    )
    assert not result.usable
    assert result.reason == "sparse_corridor_cluster"


def test_curve_reference_selects_obstacle_relative_to_curve():
    curved_y = 0.4 * REF_X
    obstacle = surface(1.0, 0.30, 0.50, 8)
    result = compact_obstacle_points(
        obstacle, REF_X, curved_y, corridor_half_width_m=0.25
    )
    assert result.usable
    assert result.obstacle_present


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fov_deg": 181.0},
        {"corridor_half_width_m": 0.0},
        {"cluster_gap_base_m": -0.1},
        {"cluster_gap_per_m": -0.1},
        {"min_cluster_points": 0},
        {"max_cluster_extent_m": 0.0},
        {"max_clusters": 0},
    ],
)
def test_bad_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        compact_obstacle_points(surface(1.0, -0.1, 0.1, 8), REF_X, REF_Y, **kwargs)


def test_bad_reference_or_points_are_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        compact_obstacle_points([[1.0, 0.0]], [0.3, 0.2], [0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        compact_obstacle_points([[np.nan, 0.0]], REF_X, REF_Y)
