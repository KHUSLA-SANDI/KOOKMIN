from types import SimpleNamespace

import numpy as np
import pytest

from track_drive_cnn_gpt.signal_mission import (
    ROUTE_MAIN,
    ROUTE_SHORTCUT,
    RACE_FINISHED,
    RACE_RUNNING,
    RACE_WAIT_GREEN,
    RaceLapController,
    RouteIntentLatch,
    SequencedRouteIntentLatch,
    TimedTriggerLatch,
    TrafficMissionController,
    classifier_detections_from_traffic_light,
    crop_normalized_detection,
    decode_signal_payload,
    decode_signal_payload_full,
    encode_signal_payload,
    extract_classifier_probabilities,
    extract_signal_confidences,
    extract_signal_detections,
    optional_class_id,
    select_route_value,
    signal_class_ids,
)


def test_classifier_probabilities_and_detector_geometry_share_one_box():
    ids = {"GREEN": 0, "LEFT": 1, "RED": 2, "YELLOW": 3}
    result = SimpleNamespace(
        probs=SimpleNamespace(data=np.asarray([0.05, 0.10, 0.80, 0.05]))
    )
    signals = extract_classifier_probabilities(result, ids)
    traffic_box = {
        "confidence": 0.91,
        "x1_norm": 0.20,
        "y1_norm": 0.10,
        "x2_norm": 0.50,
        "y2_norm": 0.30,
        "center_x_norm": 0.35,
        "center_y_norm": 0.20,
        "width_norm": 0.30,
        "height_norm": 0.20,
    }
    detections = classifier_detections_from_traffic_light(signals, traffic_box)

    assert signals == pytest.approx(
        {"GREEN": 0.05, "LEFT": 0.10, "RED": 0.80, "YELLOW": 0.05}
    )
    assert set(detections) == set(ids)
    assert detections["RED"]["confidence"] == pytest.approx(0.80)
    assert detections["RED"]["center_y_norm"] == pytest.approx(0.20)
    assert detections["GREEN"]["width_norm"] == pytest.approx(0.30)


def test_traffic_crop_padding_clamps_to_source_edges():
    image = np.zeros((100, 200, 3), np.uint8)
    crop, bounds = crop_normalized_detection(
        image,
        {
            "x1_norm": 0.00,
            "y1_norm": 0.10,
            "x2_norm": 0.20,
            "y2_norm": 0.30,
        },
        padding_ratio=0.25,
        min_size_px=8,
    )
    assert bounds == (0, 5, 50, 35)
    assert crop.shape == (30, 50, 3)


def test_signal_class_ids_accept_the_single_v3_model_with_start_r():
    names = {
        0: "GREEN",
        1: "LEFT",
        2: "RED",
        3: "START_R",
        4: "YELLOW",
        5: "lane",
        6: "mid",
    }
    assert signal_class_ids(names) == {
        "GREEN": 0,
        "LEFT": 1,
        "RED": 2,
        "YELLOW": 4,
    }
    with pytest.raises(ValueError, match="LEFT"):
        signal_class_ids({0: "GREEN", 1: "RED", 2: "YELLOW", 3: "lane", 4: "mid"})


def test_optional_trigger_class_resolution_is_case_insensitive():
    names = {0: "GREEN", 3: "START_R"}
    assert optional_class_id(names, "start_r") == 3
    assert optional_class_id(names, "cone_zone") is None
    assert optional_class_id(names, "") is None


def test_extracts_every_signal_from_the_existing_result_without_masks():
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            cls=np.asarray([0, 1, 1, 2, 3, 4], np.float32),
            conf=np.asarray([0.81, 0.30, 0.77, 0.19, 0.66, 0.99], np.float32),
        )
    )
    ids = {"GREEN": 0, "LEFT": 1, "RED": 2, "YELLOW": 3}
    signals = extract_signal_confidences(result, ids, min_confidence=0.25)
    assert signals == pytest.approx({"GREEN": 0.81, "LEFT": 0.77, "YELLOW": 0.66})


def test_signal_payload_round_trip_and_validation():
    encoded = encode_signal_payload(17, {"left": 0.8, "GREEN": 0.7})
    sequence, signals = decode_signal_payload(encoded)
    assert sequence == 17
    assert signals == {"GREEN": 0.7, "LEFT": 0.8}
    with pytest.raises(ValueError):
        decode_signal_payload('{"schema_version":"wrong","sequence":1,"signals":{}}')
    with pytest.raises(ValueError):
        encode_signal_payload(1, {"STOP": 0.9})


def test_signal_boxes_are_normalized_and_round_trip_with_confidences():
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            cls=np.asarray([2, 2, 4], np.float32),
            conf=np.asarray([0.61, 0.91, 0.99], np.float32),
            xyxy=np.asarray(
                [[100, 100, 300, 200], [200, 80, 500, 240], [0, 0, 10, 10]],
                np.float32,
            ),
        )
    )
    ids = {"GREEN": 0, "LEFT": 1, "RED": 2, "YELLOW": 3}
    detections = extract_signal_detections(
        result, ids, image_hw=(1080, 1920), min_confidence=0.25
    )
    assert set(detections) == {"RED"}
    assert detections["RED"]["confidence"] == pytest.approx(0.91)
    assert detections["RED"]["center_y_norm"] == pytest.approx(160 / 1080)
    assert detections["RED"]["width_norm"] == pytest.approx(300 / 1920)

    encoded = encode_signal_payload(3, {"RED": 0.91}, detections)
    sequence, signals, decoded = decode_signal_payload_full(encoded)
    assert sequence == 3
    assert signals == pytest.approx({"RED": 0.91})
    assert decoded["RED"]["center_y_norm"] == pytest.approx(160 / 1080)


def test_route_latch_requires_consecutive_left_and_explicit_reset():
    latch = RouteIntentLatch(left_confirm_frames=2, left_confidence=0.4)
    assert latch.route_intent == ROUTE_MAIN
    assert latch.observe({"LEFT": 0.9}) == ROUTE_MAIN
    assert latch.observe({}) == ROUTE_MAIN
    assert latch.left_streak == 0
    assert latch.observe({"LEFT": 0.8}) == ROUTE_MAIN
    assert latch.observe({"LEFT": 0.7}) == ROUTE_SHORTCUT
    assert latch.observe({}) == ROUTE_SHORTCUT
    assert latch.reset_main() == ROUTE_MAIN
    assert latch.left_streak == 0


def test_source_restart_clears_only_partial_confirmation():
    latch = RouteIntentLatch(left_confirm_frames=2, left_confidence=0.5)
    assert latch.observe({"LEFT": 0.9}) == ROUTE_MAIN
    latch.reset_observation_streak()
    assert latch.observe({"LEFT": 0.9}) == ROUTE_MAIN
    assert latch.observe({"LEFT": 0.9}) == ROUTE_SHORTCUT
    latch.reset_observation_streak()
    assert latch.route_intent == ROUTE_SHORTCUT


def test_sequence_duplicate_cannot_create_a_false_two_hit_confirmation():
    latch = SequencedRouteIntentLatch(
        left_confirm_frames=2,
        left_confidence=0.5,
    )

    first = latch.observe(10, {"LEFT": 0.9})
    duplicate = latch.observe(10, {"LEFT": 0.9})

    assert first.accepted
    assert not duplicate.accepted
    assert not duplicate.route_changed
    assert latch.left_streak == 1
    assert latch.route_intent == ROUTE_MAIN
    assert latch.observe(11, {"LEFT": 0.9}).route_intent == ROUTE_SHORTCUT


def test_sequence_restart_clears_partial_streak_but_not_latched_shortcut():
    partial = SequencedRouteIntentLatch(
        left_confirm_frames=2,
        left_confidence=0.5,
    )
    partial.observe(100, {"LEFT": 0.9})

    restarted = partial.observe(1, {"LEFT": 0.9})
    assert restarted.accepted
    assert restarted.source_restarted
    assert not restarted.route_changed
    assert partial.left_streak == 1
    assert partial.route_intent == ROUTE_MAIN
    assert partial.observe(2, {"LEFT": 0.9}).route_intent == ROUTE_SHORTCUT

    still_shortcut = partial.observe(1, {})
    assert still_shortcut.source_restarted
    assert partial.route_intent == ROUTE_SHORTCUT


def test_sequence_reset_is_explicit_and_clears_sequence_epoch():
    latch = SequencedRouteIntentLatch(left_confirm_frames=2)
    latch.observe(8, {"LEFT": 0.9})
    latch.observe(9, {"LEFT": 0.9})
    assert latch.route_intent == ROUTE_SHORTCUT

    assert latch.reset_main() == ROUTE_MAIN
    assert latch.last_sequence == -1
    assert latch.left_streak == 0
    assert latch.observe(1, {"LEFT": 0.9}).route_intent == ROUTE_MAIN


@pytest.mark.parametrize("sequence", [True, -1, 1.5, "1"])
def test_sequence_wrapper_rejects_invalid_sequence(sequence):
    latch = SequencedRouteIntentLatch()
    with pytest.raises(ValueError, match="sequence"):
        latch.observe(sequence, {})


def test_route_selection_never_falls_back_from_unavailable_shortcut():
    main = object()
    unavailable_shortcut = object()

    assert select_route_value(ROUTE_MAIN, main, unavailable_shortcut) is main
    assert (
        select_route_value(ROUTE_SHORTCUT, main, unavailable_shortcut)
        is unavailable_shortcut
    )
    with pytest.raises(ValueError, match="unsupported route"):
        select_route_value("unknown", main, unavailable_shortcut)


def _box(*, center_y=0.15, width=0.12, confidence=0.9):
    return {
        "confidence": confidence,
        "x1_norm": 0.5 - width / 2,
        "y1_norm": center_y - 0.03,
        "x2_norm": 0.5 + width / 2,
        "y2_norm": center_y + 0.03,
        "center_x_norm": 0.5,
        "center_y_norm": center_y,
        "width_norm": width,
        "height_norm": 0.06,
    }


def test_left_selects_shortcut_for_ten_seconds_without_refreshing_each_frame():
    mission = TrafficMissionController(
        confirm_frames=2, confidence=0.25, shortcut_hold_sec=10.0
    )
    first = mission.observe(1, {"LEFT": 0.9}, {"LEFT": _box()}, now_sec=100.0)
    assert first.route_changed
    assert mission.route_intent(109.99) == ROUTE_SHORTCUT

    # Continuous LEFT observations must not keep moving the ten-second end.
    mission.observe(3, {"LEFT": 0.9}, {"LEFT": _box()}, now_sec=105.0)
    assert mission.route_intent(109.99) == ROUTE_SHORTCUT
    assert mission.route_intent(110.01) == ROUTE_MAIN


def test_far_left_waits_until_the_stop_line_decision_zone():
    mission = TrafficMissionController(
        confirm_frames=2, confidence=0.25, shortcut_hold_sec=10.0
    )
    far_left = {"LEFT": _box(center_y=0.30, width=0.05)}
    first = mission.observe(1, {"LEFT": 0.9}, far_left, now_sec=1.0)
    second = mission.observe(2, {"LEFT": 0.9}, far_left, now_sec=1.1)
    assert first.dominant_signal == "LEFT"
    assert not first.decision_zone
    assert not second.route_changed
    assert mission.left_streak == 0
    assert mission.route_intent(1.1) == ROUTE_MAIN

    close_left = {"LEFT": _box(center_y=0.18, width=0.12)}
    entered = mission.observe(3, {"LEFT": 0.9}, close_left, now_sec=1.2)
    assert entered.decision_zone
    assert entered.route_changed
    assert mission.route_intent(1.2) == ROUTE_SHORTCUT


def test_one_left_in_decision_zone_latches_immediately():
    mission = TrafficMissionController(
        confirm_frames=2,
        shortcut_hold_sec=10.0,
        left_confirm_window_sec=1.5,
    )
    update = mission.observe(1, {"LEFT": 0.9}, {"LEFT": _box()}, now_sec=1.0)
    assert update.route_changed
    assert mission.route_intent(1.0) == ROUTE_SHORTCUT


@pytest.mark.parametrize("stop_name", ["RED", "YELLOW"])
def test_red_yellow_only_stop_in_position_and_near_green_releases(stop_name):
    mission = TrafficMissionController(confirm_frames=2)
    far_stop = {stop_name: _box(center_y=0.30, width=0.04)}
    mission.observe(1, {stop_name: 0.9}, far_stop, now_sec=1.0)
    mission.observe(2, {stop_name: 0.9}, far_stop, now_sec=1.1)
    assert not mission.traffic_stop

    close_stop = {stop_name: _box(center_y=0.18, width=0.12)}
    mission.observe(3, {stop_name: 0.9}, close_stop, now_sec=1.2)
    stopped = mission.observe(4, {stop_name: 0.9}, close_stop, now_sec=1.3)
    assert stopped.stop_changed
    assert mission.traffic_stop

    # Loss of detection must never release an already stopped vehicle.
    mission.observe(5, {}, {}, now_sec=1.4)
    assert mission.traffic_stop

    # A visible but far GREEN is also only a candidate, not a GO decision.
    far_green = {"GREEN": _box(center_y=0.30, width=0.04)}
    mission.observe(6, {"GREEN": 0.9}, far_green, now_sec=1.5)
    mission.observe(7, {"GREEN": 0.9}, far_green, now_sec=1.6)
    assert mission.traffic_stop

    close_green = {"GREEN": _box(center_y=0.18, width=0.12)}
    mission.observe(8, {"GREEN": 0.9}, close_green, now_sec=1.7)
    released = mission.observe(9, {"GREEN": 0.9}, close_green, now_sec=1.8)
    assert released.stop_changed
    assert not mission.traffic_stop


def test_duplicate_sequence_does_not_refresh_one_hit_left_latch():
    mission = TrafficMissionController(confirm_frames=2)
    left = {"LEFT": _box()}
    first = mission.observe(8, {"LEFT": 0.9}, left, now_sec=10.0)
    assert first.route_changed
    duplicate = mission.observe(8, {"LEFT": 0.9}, left, now_sec=10.1)
    assert not duplicate.accepted
    assert mission.route_intent(10.1) == ROUTE_SHORTCUT


def test_timed_trigger_confirms_entry_and_holds_exit_for_one_second():
    trigger = TimedTriggerLatch(enter_confirm_frames=2, exit_hold_sec=1.0)
    assert not trigger.observe(True, now_sec=1.0).active
    entered = trigger.observe(True, now_sec=1.1)
    assert entered.active and entered.changed
    assert trigger.observe(False, now_sec=2.0).active
    exited = trigger.observe(False, now_sec=2.11)
    assert not exited.active and exited.changed


def test_confirmed_green_does_not_cancel_one_hit_left_route():
    mission = TrafficMissionController(confirm_frames=2, shortcut_hold_sec=10.0)
    left = {"LEFT": _box(center_y=0.18, width=0.12)}
    mission.observe(1, {"LEFT": 0.9}, left, now_sec=1.0)
    assert mission.route_intent(1.1) == ROUTE_SHORTCUT

    green = {"GREEN": _box(center_y=0.18, width=0.12)}
    mission.observe(3, {"GREEN": 0.9}, green, now_sec=1.3)
    update = mission.observe(4, {"GREEN": 0.9}, green, now_sec=1.4)
    assert not update.route_changed
    assert mission.route_intent(1.4) == ROUTE_SHORTCUT
    assert not mission.traffic_stop


def _race_controller(*, finish_enabled=True):
    return RaceLapController(
        confirm_frames=2,
        confidence=0.25,
        decision_center_y_max=0.22,
        decision_min_width=0.08,
        target_laps=3,
        cooldown_sec=10.0,
        clear_frames=3,
        finish_enabled=finish_enabled,
    )


def _observe_race(controller, sequence, now_sec, signal=None, in_zone=False):
    signals = {} if signal is None else {signal: 0.9}
    detections = (
        {signal or "traffic_light": _box(center_y=0.18, width=0.12)}
        if in_zone
        else {}
    )
    return controller.observe(
        sequence,
        signals,
        detections,
        now_sec=now_sec,
    )


def _clear_race_zone(controller, sequence, now_sec):
    update = None
    for offset in range(3):
        update = _observe_race(
            controller,
            sequence + offset,
            now_sec + offset * 0.1,
        )
    return update


def _confirm_race_encounter(controller, sequence, now_sec, signal="RED"):
    _observe_race(controller, sequence, now_sec, signal, True)
    return _observe_race(controller, sequence + 1, now_sec + 0.1, signal, True)


def test_race_starts_on_two_consecutive_green_frames_without_box_geometry():
    race = _race_controller()
    assert race.state_name == RACE_WAIT_GREEN

    far = {"GREEN": _box(center_y=0.30, width=0.04)}
    first = race.observe(1, {"GREEN": 0.9}, far, now_sec=1.0)
    assert not first.traffic_zone
    assert first.green_confirm_count == 1
    started = race.observe(2, {"GREEN": 0.9}, far, now_sec=1.1)
    assert not started.traffic_zone
    assert started.start_changed
    assert started.race_started
    assert started.race_go
    assert started.lap_count == 0
    assert started.state_name == RACE_RUNNING
    assert not started.encounter_armed


def test_race_requires_clear_rearm_cooldown_and_two_generic_zone_frames():
    race = _race_controller()
    _observe_race(race, 1, 0.0, "GREEN", True)
    _observe_race(race, 2, 0.1, "GREEN", True)

    # Remaining in the starting zone, including a class change, never counts.
    for sequence in range(3, 8):
        update = _observe_race(race, sequence, float(sequence), "RED", True)
    assert update.lap_count == 0

    # Two OUT frames are insufficient; the third rearms the next encounter.
    _observe_race(race, 8, 8.0)
    still_disarmed = _observe_race(race, 9, 8.1)
    assert not still_disarmed.encounter_armed
    rearmed = _observe_race(race, 10, 8.2)
    assert rearmed.encounter_armed

    # IN evidence before ten seconds does not leak into the confirmation.
    too_early = _observe_race(race, 11, 9.9, "YELLOW", True)
    assert too_early.zone_confirm_count == 0
    # Lap counting is detector-geometry based; classification may be absent.
    first = _observe_race(race, 12, 10.2, None, True)
    assert first.zone_confirm_count == 1
    lap_one = _observe_race(race, 13, 10.3, None, True)
    assert lap_one.lap_changed
    assert lap_one.lap_count == 1
    assert not lap_one.encounter_armed


def test_race_sequence_duplicate_and_restart_cannot_duplicate_same_encounter():
    race = _race_controller()
    _observe_race(race, 1, 0.0, "GREEN", True)
    _observe_race(race, 2, 0.1, "GREEN", True)
    _clear_race_zone(race, 3, 1.0)
    _observe_race(race, 6, 10.2, "RED", True)
    lap_one = _observe_race(race, 7, 10.3, "RED", True)
    assert lap_one.lap_count == 1

    duplicate = _observe_race(race, 7, 10.4, "RED", True)
    assert not duplicate.accepted
    assert duplicate.lap_count == 1

    restarted = _observe_race(race, 0, 20.5, "RED", True)
    assert restarted.source_restarted
    assert restarted.lap_count == 1
    _observe_race(race, 1, 20.6, "RED", True)
    assert race.lap_count == 1


def test_race_finishes_and_latches_on_third_return():
    race = _race_controller()
    _observe_race(race, 1, 0.0, "GREEN", True)
    _observe_race(race, 2, 0.1, "GREEN", True)
    sequence = 3
    now = 1.0
    for expected_lap in range(1, 4):
        _clear_race_zone(race, sequence, now)
        sequence += 3
        now += 10.0
        update = _confirm_race_encounter(race, sequence, now)
        sequence += 2
        now += 0.2
        assert update.lap_count == expected_lap

    assert update.finish_changed
    assert update.race_finished
    assert not update.race_go
    assert update.state_name == RACE_FINISHED
    _clear_race_zone(race, sequence, now)
    later = _confirm_race_encounter(race, sequence + 3, now + 20.0, "GREEN")
    assert later.race_finished
    assert later.lap_count == 3


def test_race_finish_disabled_keeps_running_and_counting_past_target():
    race = _race_controller(finish_enabled=False)
    _observe_race(race, 1, 0.0, "GREEN", True)
    _observe_race(race, 2, 0.1, "GREEN", True)
    sequence = 3
    now = 1.0
    for expected_lap in range(1, 5):
        _clear_race_zone(race, sequence, now)
        sequence += 3
        now += 10.0
        update = _confirm_race_encounter(race, sequence, now)
        sequence += 2
        now += 0.2
        assert update.lap_count == expected_lap
        assert not update.race_finished
        assert update.race_go
        assert update.state_name == RACE_RUNNING
