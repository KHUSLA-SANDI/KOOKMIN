from types import SimpleNamespace

import numpy as np
import pytest

from track_drive_cnn_gpt.signal_mission import (
    ROUTE_MAIN,
    ROUTE_SHORTCUT,
    RouteIntentLatch,
    SequencedRouteIntentLatch,
    TimedTriggerLatch,
    TrafficMissionController,
    decode_signal_payload,
    decode_signal_payload_full,
    encode_signal_payload,
    extract_signal_confidences,
    extract_signal_detections,
    optional_class_id,
    select_route_value,
    signal_class_ids,
)


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
    mission.observe(1, {"LEFT": 0.9}, {"LEFT": _box()}, now_sec=100.0)
    second = mission.observe(
        2, {"LEFT": 0.9}, {"LEFT": _box()}, now_sec=100.1
    )
    assert second.route_changed
    assert mission.route_intent(109.99) == ROUTE_SHORTCUT

    # Continuous LEFT observations must not keep moving the ten-second end.
    mission.observe(3, {"LEFT": 0.9}, {"LEFT": _box()}, now_sec=105.0)
    assert mission.route_intent(110.09) == ROUTE_SHORTCUT
    assert mission.route_intent(110.11) == ROUTE_MAIN


def test_left_confirmation_allows_one_short_gap_inside_time_window():
    mission = TrafficMissionController(
        confirm_frames=2,
        shortcut_hold_sec=10.0,
        left_confirm_window_sec=1.5,
    )
    mission.observe(1, {"LEFT": 0.9}, {"LEFT": _box()}, now_sec=1.0)
    mission.observe(2, {"GREEN": 0.9}, {"GREEN": _box()}, now_sec=1.4)
    update = mission.observe(3, {"LEFT": 0.9}, {"LEFT": _box()}, now_sec=2.0)
    assert update.route_changed
    assert mission.route_intent(2.0) == ROUTE_SHORTCUT


def test_red_yellow_only_stop_in_position_and_green_releases():
    mission = TrafficMissionController(confirm_frames=2)
    far_red = {"RED": _box(center_y=0.30, width=0.04)}
    mission.observe(1, {"RED": 0.9}, far_red, now_sec=1.0)
    mission.observe(2, {"RED": 0.9}, far_red, now_sec=1.1)
    assert not mission.traffic_stop

    close_red = {"RED": _box(center_y=0.18, width=0.12)}
    mission.observe(3, {"RED": 0.9}, close_red, now_sec=1.2)
    stopped = mission.observe(4, {"RED": 0.9}, close_red, now_sec=1.3)
    assert stopped.stop_changed
    assert mission.traffic_stop

    # Loss of detection must never release an already stopped vehicle.
    mission.observe(5, {}, {}, now_sec=1.4)
    assert mission.traffic_stop
    close_green = {"GREEN": _box(center_y=0.18, width=0.12)}
    mission.observe(6, {"GREEN": 0.9}, close_green, now_sec=1.5)
    released = mission.observe(7, {"GREEN": 0.9}, close_green, now_sec=1.6)
    assert released.stop_changed
    assert not mission.traffic_stop


def test_duplicate_sequence_does_not_confirm_stop_or_left():
    mission = TrafficMissionController(confirm_frames=2)
    left = {"LEFT": _box()}
    mission.observe(8, {"LEFT": 0.9}, left, now_sec=10.0)
    duplicate = mission.observe(8, {"LEFT": 0.9}, left, now_sec=10.1)
    assert not duplicate.accepted
    assert mission.route_intent(10.1) == ROUTE_MAIN


def test_timed_trigger_confirms_entry_and_holds_exit_for_one_second():
    trigger = TimedTriggerLatch(enter_confirm_frames=2, exit_hold_sec=1.0)
    assert not trigger.observe(True, now_sec=1.0).active
    entered = trigger.observe(True, now_sec=1.1)
    assert entered.active and entered.changed
    assert trigger.observe(False, now_sec=2.0).active
    exited = trigger.observe(False, now_sec=2.11)
    assert not exited.active and exited.changed


def test_confirmed_green_at_decision_line_forces_main_and_go():
    mission = TrafficMissionController(confirm_frames=2, shortcut_hold_sec=10.0)
    left = {"LEFT": _box(center_y=0.30, width=0.05)}
    mission.observe(1, {"LEFT": 0.9}, left, now_sec=1.0)
    mission.observe(2, {"LEFT": 0.9}, left, now_sec=1.1)
    assert mission.route_intent(1.2) == ROUTE_SHORTCUT

    green = {"GREEN": _box(center_y=0.18, width=0.12)}
    mission.observe(3, {"GREEN": 0.9}, green, now_sec=1.3)
    update = mission.observe(4, {"GREEN": 0.9}, green, now_sec=1.4)
    assert update.route_changed
    assert mission.route_intent(1.4) == ROUTE_MAIN
    assert not mission.traffic_stop
