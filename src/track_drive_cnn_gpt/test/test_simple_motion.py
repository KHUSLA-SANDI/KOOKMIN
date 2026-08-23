import numpy as np
import pytest

from track_drive_cnn_gpt.simple_motion_node import (
    TwoStageSpeedController,
    motion_drive_allowed,
    select_requested_speed,
    simple_steering_command,
    smooth_steering,
)


def _speed_controller():
    return TwoStageSpeedController(
        startup_speed_cmd=0.0,
        startup_hold_sec=0.0,
        slew_speed_up_per_tick=0.25,
        slew_speed_down_per_tick=2.0,
    )


def test_speed_starts_at_zero_and_ramps_each_tick():
    controller = _speed_controller()

    assert controller.update(drive=True, target_speed=12.0, now_sec=10.0) == 0.0
    assert controller.phase == "RUNNING"
    assert controller.update(drive=True, target_speed=12.0, now_sec=10.05) == 0.25
    assert controller.update(drive=True, target_speed=12.0, now_sec=10.10) == 0.50


def test_mode_speed_change_does_not_restart_startup_and_stop_is_immediate():
    controller = _speed_controller()
    controller.update(drive=True, target_speed=12.0, now_sec=10.0)
    for tick in range(40):
        controller.update(
            drive=True,
            target_speed=12.0,
            now_sec=10.05 + tick * 0.05,
        )

    assert controller.update(drive=True, target_speed=8.0, now_sec=12.05) == 8.0
    assert controller.phase == "RUNNING"
    assert controller.update(drive=False, target_speed=0.0, now_sec=11.1) == 0.0
    assert controller.phase == "STOPPED"
    assert controller.update(drive=True, target_speed=6.0, now_sec=12.0) == 0.0


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


def test_drive_authorization_does_not_depend_on_path_availability():
    assert motion_drive_allowed(
        cmd_fresh=True,
        cmd_valid=True,
        owner_lane=True,
        speed_cap=5.0,
    )
    assert not motion_drive_allowed(
        cmd_fresh=False,
        cmd_valid=True,
        owner_lane=True,
        speed_cap=5.0,
    )
    assert not motion_drive_allowed(
        cmd_fresh=True,
        cmd_valid=True,
        owner_lane=True,
        speed_cap=0.0,
    )


def test_cone_approach_limits_only_general_profile_speed():
    speed, limited = select_requested_speed(
        50.0,
        cnn_mode="GENERAL",
        cone_approach_active=True,
        cone_approach_speed=8.0,
    )
    assert speed == pytest.approx(8.0)
    assert limited

    speed, limited = select_requested_speed(
        6.0,
        cnn_mode="CONE",
        cone_approach_active=True,
        cone_approach_speed=8.0,
    )
    assert speed == pytest.approx(6.0)
    assert not limited

    speed, limited = select_requested_speed(
        50.0,
        cnn_mode="GENERAL",
        cone_approach_active=False,
        cone_approach_speed=8.0,
    )
    assert speed == pytest.approx(50.0)
    assert not limited
