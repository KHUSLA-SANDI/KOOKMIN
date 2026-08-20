from pathlib import Path

from track_drive_cnn_gpt.supervisor_node import (
    OWNER_LANE,
    OWNER_SAFETY_STOP,
    PathPairAssembler,
    ROUTE_MAIN,
    ROUTE_SHORTCUT,
    SupervisorLogic,
    encode_drive_command,
)


def test_route_topics_commit_only_as_one_same_stamp_pair():
    pairs = PathPairAssembler()

    assert pairs.add(ROUTE_MAIN, 100, "main-100", usable=True) is None
    assert pairs.add(ROUTE_SHORTCUT, 99, "shortcut-99", usable=True) is None

    committed = pairs.add(
        ROUTE_SHORTCUT,
        100,
        "shortcut-100",
        usable=True,
    )
    assert committed is not None
    assert committed.stamp_ns == 100
    assert committed.main == "main-100"
    assert committed.shortcut == "shortcut-100"


def test_empty_route_is_part_of_pair_and_cannot_leave_old_route_available():
    pairs = PathPairAssembler()
    assert pairs.add(ROUTE_MAIN, 200, "main-200", usable=True) is None

    committed = pairs.add(
        ROUTE_SHORTCUT,
        200,
        "empty-shortcut-message-200",
        usable=False,
    )

    assert committed is not None
    assert committed.main_usable
    assert not committed.shortcut_usable
    # Keep the actual empty ROS message so the supervisor can forward it to
    # motion as an explicit cache invalidation when shortcut is requested.
    assert committed.shortcut == "empty-shortcut-message-200"


def test_delayed_pair_cannot_replace_a_newer_committed_pair():
    pairs = PathPairAssembler()
    pairs.add(ROUTE_MAIN, 300, "main-300", usable=True)
    committed = pairs.add(ROUTE_SHORTCUT, 300, "shortcut-300", usable=True)
    assert committed is not None

    assert pairs.add(ROUTE_MAIN, 299, "old-main", usable=True) is None
    assert pairs.add(ROUTE_SHORTCUT, 299, "old-shortcut", usable=True) is None
    assert pairs.last_committed_stamp_ns == 300


def test_pair_pending_clear_does_not_allow_replay_of_committed_stamp():
    pairs = PathPairAssembler()
    pairs.add(ROUTE_MAIN, 400, "main-400", usable=True)
    assert pairs.add(ROUTE_SHORTCUT, 400, "shortcut-400", usable=True)

    pairs.add(ROUTE_MAIN, 401, "partial", usable=True)
    pairs.clear()
    assert pairs.add(ROUTE_SHORTCUT, 400, "replay", usable=True) is None

    assert pairs.add(ROUTE_MAIN, 402, "main-402", usable=True) is None
    assert pairs.add(ROUTE_SHORTCUT, 402, "shortcut-402", usable=True)


def test_drive_is_disabled_by_default():
    logic = SupervisorLogic()
    logic.set_manual_go(True)
    logic.update_path(ROUTE_MAIN, 10.0)

    decision = logic.decide(10.0)

    assert not decision.drive_allowed
    assert decision.reason == "drive_disabled"


def test_manual_go_and_fresh_main_path_are_both_required():
    logic = SupervisorLogic(enable_drive=True, path_stale_sec=0.25)
    logic.update_path(ROUTE_MAIN, 10.0)

    assert logic.decide(10.1).reason == "manual_go_required"

    logic.set_manual_go(True)
    decision = logic.decide(10.2)
    assert decision.drive_allowed
    assert decision.selected_route == ROUTE_MAIN

    stale = logic.decide(10.251)
    assert not stale.drive_allowed
    assert stale.reason == "main_path_stale"


def test_shortcut_does_not_silently_fallback_by_default():
    logic = SupervisorLogic(enable_drive=True)
    logic.set_manual_go(True)
    logic.set_route_intent("LEFT")
    logic.update_path(ROUTE_MAIN, 5.0)

    decision = logic.decide(5.0)

    assert not decision.drive_allowed
    assert decision.reason == "shortcut_path_stale"


def test_shortcut_can_fallback_only_when_explicitly_enabled():
    logic = SupervisorLogic(
        enable_drive=True,
        shortcut_fallback_to_main=True,
    )
    logic.set_manual_go(True)
    logic.set_route_intent("shortcut")
    logic.update_path(ROUTE_MAIN, 5.0)

    fallback = logic.decide(5.1)
    assert fallback.drive_allowed
    assert fallback.selected_route == ROUTE_MAIN
    assert fallback.reason == "shortcut_fallback_main"

    logic.update_path(ROUTE_SHORTCUT, 5.1)
    shortcut = logic.decide(5.2)
    assert shortcut.drive_allowed
    assert shortcut.selected_route == ROUTE_SHORTCUT


def test_emergency_stop_and_invalid_route_fail_closed():
    logic = SupervisorLogic(enable_drive=True)
    logic.set_manual_go(True)
    logic.update_path(ROUTE_MAIN, 2.0)
    logic.set_emergency_stop(True)

    assert logic.decide(2.0).reason == "emergency_stop"

    logic.set_emergency_stop(False)
    logic.set_route_intent("unknown-route")
    assert logic.decide(2.0).reason == "invalid_route_intent"


def test_future_source_timestamp_is_not_treated_as_fresh():
    logic = SupervisorLogic(enable_drive=True)
    logic.set_manual_go(True)
    logic.update_path(ROUTE_MAIN, 10.1)

    decision = logic.decide(10.0)

    assert not decision.drive_allowed
    assert decision.reason == "main_path_stale"


def test_drive_command_encoding_matches_legacy_motion_contract():
    stop = encode_drive_command(False, 6.0)
    assert len(stop) == 8
    assert stop[0] == OWNER_SAFETY_STOP
    assert stop[2] == 0.0

    drive = encode_drive_command(True, 6.0)
    assert len(drive) == 8
    assert drive[0] == OWNER_LANE
    assert drive[2] == 6.0

    invalid_cap = encode_drive_command(True, 0.0)
    assert invalid_cap[0] == OWNER_SAFETY_STOP


def test_perception_launch_has_no_control_publishers():
    launch_dir = Path(__file__).resolve().parents[1] / "launch"
    source = (launch_dir / "perception_only.launch.py").read_text(encoding="utf-8")

    assert 'executable="cnn_supervisor"' not in source
    assert 'executable="motion"' not in source
    assert 'executable="path_planner"' not in source
    assert '"/drive_cmd"' not in source
    assert '"/xycar_motor"' not in source


def test_perception_launch_can_run_yolo_without_cnn():
    launch_dir = Path(__file__).resolve().parents[1] / "launch"
    source = (launch_dir / "perception_only.launch.py").read_text(encoding="utf-8")

    assert 'DeclareLaunchArgument("enable_yolo", default_value="true")' in source
    assert 'DeclareLaunchArgument("enable_cnn", default_value="false")' in source
    assert 'condition=IfCondition(enable_yolo)' in source
    assert 'condition=IfCondition(enable_cnn)' in source
    assert 'executable="mission_route"' in source


def test_low_speed_launch_requires_explicit_supervisor_opt_in():
    launch_dir = Path(__file__).resolve().parents[1] / "launch"
    source = (launch_dir / "drive_low_speed.launch.py").read_text(encoding="utf-8")

    assert 'DeclareLaunchArgument("enable_supervisor", default_value="false")' in source
    assert 'condition=IfCondition(enable_supervisor)' in source
    assert 'DeclareLaunchArgument("enable_motor", default_value="false")' in source
    assert 'DeclareLaunchArgument("enable_drive", default_value="false")' in source
    assert 'executable="path_planner"' not in source
