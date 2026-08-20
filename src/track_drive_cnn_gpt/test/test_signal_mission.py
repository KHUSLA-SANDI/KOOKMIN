from types import SimpleNamespace

import numpy as np
import pytest

from track_drive_cnn_gpt.signal_mission import (
    ROUTE_MAIN,
    ROUTE_SHORTCUT,
    RouteIntentLatch,
    decode_signal_payload,
    encode_signal_payload,
    extract_signal_confidences,
    signal_class_ids,
)


def test_signal_class_ids_require_the_single_six_class_model():
    names = {0: "GREEN", 1: "LEFT", 2: "RED", 3: "YELLOW", 4: "lane", 5: "mid"}
    assert signal_class_ids(names) == {
        "GREEN": 0,
        "LEFT": 1,
        "RED": 2,
        "YELLOW": 3,
    }
    with pytest.raises(ValueError, match="LEFT"):
        signal_class_ids({0: "GREEN", 1: "RED", 2: "YELLOW", 3: "lane", 4: "mid"})


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
