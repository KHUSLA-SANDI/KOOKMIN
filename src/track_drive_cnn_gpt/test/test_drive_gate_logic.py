import pytest

from track_drive_cnn_gpt.drive_gate_node import (
    DriveGateLogic,
    encode_drive_command,
)


def test_drive_gate_is_disabled_and_unarmed_by_default():
    logic = DriveGateLogic(enable_drive=False, path_stale_sec=0.25)
    logic.update_path(10.0)

    decision = logic.decide(10.0)

    assert not decision.drive_allowed
    assert decision.reason == "drive_disabled"


def test_drive_gate_waits_for_green_but_not_a_selected_path():
    logic = DriveGateLogic(enable_drive=True, path_stale_sec=0.25)
    assert logic.decide(10.0).reason == "waiting_for_green"

    logic.set_race_go(True)
    assert logic.decide(10.0).drive_allowed

    logic.update_path(9.90)
    decision = logic.decide(10.0)
    assert decision.drive_allowed
    assert decision.reason == "ok_race"


def test_manual_command_overrides_automatic_race_arming_both_ways():
    logic = DriveGateLogic(enable_drive=True, path_stale_sec=0.25)
    logic.set_race_go(True)
    assert logic.decide(10.0).drive_allowed

    logic.set_manual_go(False)
    assert logic.decide(10.0).reason == "manual_stop"

    logic.set_manual_go(True)
    logic.set_race_go(False)
    decision = logic.decide(10.0)
    assert decision.drive_allowed
    assert decision.reason == "ok_manual_override"


def test_drive_gate_ignores_stale_future_and_invalidated_paths():
    logic = DriveGateLogic(enable_drive=True, path_stale_sec=0.25)
    logic.set_manual_go(True)

    logic.update_path(9.70)
    assert logic.decide(10.0).drive_allowed

    logic.update_path(10.01)
    assert logic.decide(10.0).drive_allowed

    logic.update_path(None)
    assert logic.decide(10.0).drive_allowed


def test_emergency_stop_has_priority_over_manual_go_and_path():
    logic = DriveGateLogic(enable_drive=True, path_stale_sec=0.25)
    logic.set_manual_go(True)
    logic.update_path(10.0)
    logic.set_emergency_stop(True)

    decision = logic.decide(10.0)

    assert not decision.drive_allowed
    assert decision.reason == "emergency_stop"


def test_traffic_signal_stop_blocks_and_releases_a_fresh_path():
    logic = DriveGateLogic(enable_drive=True, path_stale_sec=0.25)
    logic.set_manual_go(True)
    logic.update_path(10.0)
    logic.set_traffic_stop(True)
    assert logic.decide(10.0).reason == "traffic_signal_stop"

    logic.set_traffic_stop(False)
    assert logic.decide(10.0).drive_allowed


@pytest.mark.parametrize("stale_sec", [0.0, -1.0, float("nan")])
def test_drive_gate_rejects_invalid_stale_policy(stale_sec):
    with pytest.raises(ValueError, match="path_stale_sec"):
        DriveGateLogic(path_stale_sec=stale_sec)


def test_drive_command_encoding_stays_in_the_legacy_eight_float_contract():
    assert encode_drive_command(False, 6.0) == [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert encode_drive_command(True, 6.0) == [0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert encode_drive_command(True, float("nan"))[0] == 3.0
    assert encode_drive_command(True, 0.0)[0] == 3.0
