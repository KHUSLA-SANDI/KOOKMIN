import numpy as np
import pytest

from track_drive_cnn_gpt.lidar_geometry import GRID_H, GRID_W, RESOLUTION, X_MAX, Y_MAX
from track_drive_cnn_gpt.road_lidar_filter import filter_and_detect_in_road


def _straight_white_grid(half_width_m=0.50):
    grid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
    for y in (half_width_m, -half_width_m):
        col = int(np.floor((Y_MAX - y) / RESOLUTION))
        grid[:, max(0, col - 1) : min(GRID_W, col + 2)] = 1
    return grid


def _straight_reference():
    return np.asarray([0.3, 3.0], np.float32), np.asarray([0.0, 0.0], np.float32)


def _result(points):
    x, y = _straight_reference()
    return filter_and_detect_in_road(
        np.asarray(points, np.float32),
        _straight_white_grid(),
        x,
        y,
        trigger_inward_margin_m=0.10,
        trigger_x_max_m=1.20,
        trigger_min_points=3,
    )


def test_center_cluster_within_1p2m_triggers_overtake():
    result = _result([(0.60, -0.02), (0.62, 0.00), (0.64, 0.02)])
    assert result.trigger_detected
    assert result.trigger_inside_points == 3


def test_lateral_offset_is_relative_to_mid_reference():
    result = _result([(0.60, 0.20), (0.62, 0.22), (0.64, 0.24)])
    assert result.trigger_detected
    assert result.trigger_lateral_offset_m == pytest.approx(0.22, abs=0.02)
    assert not result.trigger_side_ambiguous


def test_opposite_side_qualifying_clusters_are_directionally_ambiguous():
    result = _result(
        [
            (0.55, 0.20),
            (0.57, 0.22),
            (0.59, 0.24),
            (0.90, -0.20),
            (0.92, -0.22),
            (0.94, -0.24),
        ]
    )
    assert result.trigger_detected
    assert result.trigger_side_ambiguous
    assert result.trigger_lateral_offset_m is None


def test_white_line_overlap_is_not_strict_trigger_evidence():
    result = _result([(0.60, 0.45), (0.62, 0.46), (0.64, 0.44)])
    assert not result.trigger_detected


def test_cluster_beyond_1p2m_does_not_trigger():
    result = _result([(1.30, -0.02), (1.32, 0.00), (1.34, 0.02)])
    assert not result.trigger_detected


def test_outside_white_point_is_removed_from_model_input():
    result = _result([(0.60, 0.70), (0.62, 0.72), (0.64, 0.74)])
    assert result.filtered_points.shape == (0, 2)
    assert result.removed_points == 3
    assert not result.trigger_detected
