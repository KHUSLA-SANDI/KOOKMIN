from track_drive_cnn_gpt.cnn_modes import (
    MODE_CONE,
    MODE_GENERAL,
    MODE_OVERTAKE,
    MODE_SHORTCUT,
    CnnModeController,
    ConfirmedOneShot,
)


def test_fixed_windows_expire_without_detection_extension():
    modes = CnnModeController(
        shortcut_hold_sec=10.0,
        overtake_hold_sec=7.0,
        cone_hold_sec=20.0,
    )
    assert modes.trigger(MODE_SHORTCUT, now_sec=1.0, reason="LEFT").mode == MODE_SHORTCUT
    repeated = modes.trigger(MODE_SHORTCUT, now_sec=5.0, reason="LEFT_again")
    assert repeated.mode == MODE_SHORTCUT
    assert repeated.remaining_sec == 6.0
    assert modes.snapshot(10.99).mode == MODE_SHORTCUT
    expired = modes.snapshot(11.0)
    assert expired.mode == MODE_GENERAL
    assert expired.changed


def test_higher_priority_mode_preempts_but_lower_priority_noise_does_not():
    modes = CnnModeController()
    modes.trigger(MODE_SHORTCUT, now_sec=0.0, reason="LEFT")
    assert modes.trigger(MODE_OVERTAKE, now_sec=1.0, reason="road_obstacle").mode == MODE_OVERTAKE
    assert modes.trigger(MODE_SHORTCUT, now_sec=2.0, reason="LEFT_noise").mode == MODE_OVERTAKE
    cone = modes.trigger(MODE_CONE, now_sec=3.0, reason="START_R")
    assert cone.mode == MODE_CONE
    assert modes.trigger(MODE_OVERTAKE, now_sec=4.0, reason="lidar_noise").mode == MODE_CONE
    assert modes.snapshot(23.0).mode == MODE_GENERAL


def test_confirmed_trigger_requires_clear_frames_before_rearming():
    trigger = ConfirmedOneShot(confirm_frames=2, rearm_clear_frames=3)
    assert trigger.armed
    assert trigger.positive_count == 0
    assert not trigger.observe(True)
    assert trigger.positive_count == 1
    assert trigger.observe(True)
    assert not trigger.armed
    assert not trigger.observe(True)
    assert not trigger.observe(False)
    assert not trigger.observe(False)
    assert not trigger.observe(False)
    assert not trigger.observe(True)
    assert trigger.observe(True)
