import numpy as np
import pytest

from track_drive_cnn_gpt.simple_motion_node import (
    simple_steering_command,
    smooth_steering,
)


def test_straight_path_commands_zero_steering():
    x = np.arange(0.3, 3.01, 0.1)
    y = np.zeros_like(x)

    angle, target_x, target_y = simple_steering_command(
        x, y, lookahead_m=1.0, steer_gain=0.45
    )

    assert angle == pytest.approx(0.0)
    assert target_x == pytest.approx(1.0)
    assert target_y == pytest.approx(0.0)


def test_left_positive_path_uses_vehicle_left_steering_sign():
    x = np.arange(0.3, 3.01, 0.1)
    y = np.full_like(x, 0.2)

    angle, _, _ = simple_steering_command(
        x, y, lookahead_m=1.0, steer_gain=0.45
    )

    # Existing vehicle logical convention is left-negative.
    assert angle < 0.0


def test_short_path_clamps_lookahead_without_extrapolation():
    angle, target_x, target_y = simple_steering_command(
        [0.3, 0.5, 0.7],
        [0.0, 0.1, 0.2],
        lookahead_m=1.0,
        steer_gain=0.45,
    )

    assert target_x == pytest.approx(0.7)
    assert target_y == pytest.approx(0.2)
    assert angle < 0.0


def test_smoothing_uses_only_one_alpha():
    assert smooth_steering(0.0, 20.0, 0.4) == pytest.approx(8.0)
    assert smooth_steering(8.0, -2.0, 0.4) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="alpha"):
        smooth_steering(0.0, 1.0, 0.0)
