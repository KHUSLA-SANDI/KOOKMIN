import pytest

from track_drive_cnn_gpt.hardcoded_overtake import (
    OBSTACLE_LANE_1,
    OBSTACLE_LANE_2,
    OBSTACLE_UNKNOWN,
    DirectionalConfirmedOneShot,
    HardcodedOvertakeRequest,
    PostConeHardcodeWindow,
    avoidance_direction,
    classify_obstacle_lane,
)
from track_drive_cnn_gpt.overtake_block import (
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
    PHASE_COUNTER,
    PHASE_IDLE,
    PHASE_PASS,
    PHASE_SHIFT,
    HardcodedOvertakeBlock,
    OvertakePulseProfile,
)


def _profile(direction: str) -> OvertakePulseProfile:
    if direction == DIRECTION_RIGHT:
        return OvertakePulseProfile(
            direction=direction,
            shift_angle_cmd=60.0,
            shift_ticks=6,
            counter_angle_cmd=-60.0,
            counter_ticks=10,
            lane_change_speed_cmd=16.0,
            pass_speed_cmd=16.0,
            pass_ticks=20,
        )
    return OvertakePulseProfile(
        direction=direction,
        shift_angle_cmd=-60.0,
        shift_ticks=7,
        counter_angle_cmd=60.0,
        counter_ticks=10,
        lane_change_speed_cmd=16.0,
        pass_speed_cmd=16.0,
        pass_ticks=20,
    )


@pytest.mark.parametrize(
    ("direction", "shift_ticks"),
    ((DIRECTION_RIGHT, 6), (DIRECTION_LEFT, 7)),
)
def test_field_profiles_run_exact_ticks_then_next_cycle_is_idle(direction, shift_ticks):
    block = HardcodedOvertakeBlock()
    block.start(_profile(direction))
    commands = [block.step() for _ in range(shift_ticks + 10 + 20)]

    assert [item.phase for item in commands[:shift_ticks]] == [PHASE_SHIFT] * shift_ticks
    assert [item.phase for item in commands[shift_ticks : shift_ticks + 10]] == [
        PHASE_COUNTER
    ] * 10
    assert [item.phase for item in commands[-20:]] == [PHASE_PASS] * 20
    assert commands[-1].phase_tick == 20
    assert commands[-1].phase_ticks == 20
    assert commands[-1].completes_after_publish
    assert not block.active
    # simple_motion maps the first following path-control cycle to IDLE.
    following_phase = PHASE_IDLE if not block.active else block.step().phase
    assert following_phase == PHASE_IDLE


def test_lane_sign_and_deadband_map_away_from_obstacle():
    assert classify_obstacle_lane(0.176, deadband_m=0.10) == OBSTACLE_LANE_1
    assert classify_obstacle_lane(-0.174, deadband_m=0.10) == OBSTACLE_LANE_2
    assert classify_obstacle_lane(0.099, deadband_m=0.10) == OBSTACLE_UNKNOWN
    assert avoidance_direction(OBSTACLE_LANE_1) == DIRECTION_RIGHT
    assert avoidance_direction(OBSTACLE_LANE_2) == DIRECTION_LEFT


def test_confirmation_requires_two_frames_from_the_same_side():
    latch = DirectionalConfirmedOneShot(confirm_frames=2, rearm_clear_frames=2)
    assert latch.observe(OBSTACLE_LANE_1) is None
    assert latch.observe(OBSTACLE_LANE_2) is None
    assert latch.positive_count == 1
    assert latch.observe(OBSTACLE_LANE_2) == OBSTACLE_LANE_2
    assert not latch.armed
    assert latch.observe(None) is None
    assert latch.observe(None) is None
    assert latch.armed


def test_unknown_is_rejected_and_does_not_rearm_a_fired_obstacle():
    latch = DirectionalConfirmedOneShot(confirm_frames=2, rearm_clear_frames=1)
    assert latch.observe(OBSTACLE_LANE_1) is None
    assert latch.observe(OBSTACLE_UNKNOWN) is None
    assert latch.observe(OBSTACLE_LANE_1) is None
    assert latch.observe(OBSTACLE_LANE_1) == OBSTACLE_LANE_1
    assert latch.observe(OBSTACLE_UNKNOWN) is None
    assert not latch.armed


def test_request_round_trip_rejects_direction_toward_obstacle():
    request = HardcodedOvertakeRequest(
        event_id="session-1",
        direction=DIRECTION_RIGHT,
        obstacle_lane=OBSTACLE_LANE_1,
        lateral_offset_m=0.22,
    )
    assert HardcodedOvertakeRequest.from_json(request.to_json()) == request
    with pytest.raises(ValueError, match="does not avoid"):
        HardcodedOvertakeRequest(
            event_id="bad",
            direction=DIRECTION_LEFT,
            obstacle_lane=OBSTACLE_LANE_1,
            lateral_offset_m=0.22,
        )


def test_post_cone_window_uses_original_deadline_and_reset_never_arms():
    window = PostConeHardcodeWindow(window_sec=5.0)
    window.observe_transition(
        previous_mode="GENERAL",
        mode="CONE",
        changed=True,
        reason="start_r_confirmed",
        remaining_sec=10.0,
        now_sec=100.0,
    )
    # The expiry is observed late at 111, but the window still ends at 115.
    window.observe_transition(
        previous_mode="CONE",
        mode="GENERAL",
        changed=True,
        reason="timer_expired",
        remaining_sec=0.0,
        now_sec=111.0,
    )
    assert window.remaining_sec(111.0) == pytest.approx(4.0)
    assert not window.active(115.0)

    reset = PostConeHardcodeWindow(window_sec=5.0)
    reset.observe_transition(
        previous_mode="GENERAL",
        mode="CONE",
        changed=True,
        reason="start_r_confirmed",
        remaining_sec=10.0,
        now_sec=100.0,
    )
    reset.observe_transition(
        previous_mode="CONE",
        mode="GENERAL",
        changed=True,
        reason="route_reset",
        remaining_sec=0.0,
        now_sec=102.0,
    )
    assert not reset.active(102.0)
