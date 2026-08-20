import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from track_drive_cnn_gpt.replay_io import (
    discover_image_files,
    load_frame_records,
    load_frame_records_with_stats,
    load_scan_records,
    load_scan_records_with_stats,
    percentile,
)
from track_drive_cnn_gpt.replay_source_node import _spin_until_exit
from track_drive_cnn_gpt.replay_preview_node import ReplayPreviewNode
from track_drive_cnn_gpt.replay_recorder_node import ReplayRecorderNode


ROOT = Path(__file__).resolve().parents[1]


def test_image_discovery_is_recursive_filtered_and_natural(tmp_path):
    (tmp_path / "nested").mkdir()
    for relative in ("10.jpg", "2.JPG", "nested/001.png"):
        (tmp_path / relative).write_bytes(b"fixture")
    (tmp_path / "ignore.json").write_text("{}", encoding="utf-8")
    found = discover_image_files(tmp_path)
    assert [path.name for path in found] == ["2.JPG", "10.jpg", "001.png"]


def test_image_discovery_rejects_empty_and_non_image(tmp_path):
    with pytest.raises(ValueError, match="contains no supported images"):
        discover_image_files(tmp_path)
    text = tmp_path / "frames.txt"
    text.write_text("none", encoding="utf-8")
    with pytest.raises(ValueError, match="not a supported image"):
        discover_image_files(text)


def test_percentile_is_interpolated_and_validated():
    assert percentile([], 0.5) is None
    assert percentile([7], 0.95) == 7
    assert percentile([0, 10], 0.5) == 5
    with pytest.raises(ValueError, match="fraction"):
        percentile([1], 1.1)


def test_recorded_wall_timeline_and_compact_scan_conversion(tmp_path):
    (tmp_path / "000000.jpg").write_bytes(b"frame")
    (tmp_path / "000001.jpg").write_bytes(b"frame")
    (tmp_path / "frames.csv").write_text(
        "idx,file,stamp_sec,wall_sec\n"
        "0,000000.jpg,90.0,100.0\n"
        "1,000001.jpg,90.1,100.1\n",
        encoding="utf-8",
    )
    (tmp_path / "scan.jsonl").write_text(
        '{"i":0,"wall":100.02,"a0":-3.14,"da":0.01,'
        '"rmin":0.1,"rmax":16.0,"cm":[0,125]}\n',
        encoding="utf-8",
    )
    frames, mode = load_frame_records(tmp_path, 30.0)
    scans = load_scan_records(tmp_path / "scan.jsonl")
    assert mode == "recorded_wall"
    assert [row.wall_sec for row in frames] == [100.0, 100.1]
    assert scans[0].ranges_m[0] == float("inf")
    assert scans[0].ranges_m[1] == pytest.approx(1.25)


def test_scan_loader_skips_and_reports_malformed_complete_lines(tmp_path):
    path = tmp_path / "scan.jsonl"
    path.write_text(
        '{{"i":0,"wall":1}\n'
        '{"i":1,"wall":1.1,"a0":-3.14,"da":0.01,'
        '"rmin":0.1,"rmax":16.0,"cm":[0,100]}\n',
        encoding="utf-8",
    )
    records, stats = load_scan_records_with_stats(path)
    assert [record.source_index for record in records] == [1]
    assert stats.total_nonblank_lines == 2
    assert stats.loaded_records == 1
    assert stats.skipped_lines == (1,)


def test_frame_loader_skips_bad_row_and_clamps_backward_wall_time(tmp_path):
    (tmp_path / "000000.jpg").write_bytes(b"frame")
    (tmp_path / "000001.jpg").write_bytes(b"frame")
    (tmp_path / "frames.csv").write_text(
        "idx,file,wall_sec\n"
        "0,000000.jpg,10.0\n"
        "broken\n"
        "1,000001.jpg,9.99\n",
        encoding="utf-8",
    )
    frames, mode, stats = load_frame_records_with_stats(tmp_path, 30.0)
    assert mode == "recorded_wall"
    assert [record.source_index for record in frames] == [0, 1]
    assert [record.wall_sec for record in frames] == [10.0, 10.0]
    assert stats.skipped_lines == (3,)
    assert stats.clamped_wall_lines == (4,)


def test_replay_config_matches_live_sensor_contract():
    config = yaml.safe_load(
        (ROOT / "config" / "replay_gpt.yaml").read_text(encoding="utf-8")
    )
    image = config["replay_image_node"]["ros__parameters"]
    scan = config["replay_clean_scan_node"]["ros__parameters"]
    assert image["image_topic"] == "/image_raw"
    assert image["expected_width"] == 1920
    assert image["expected_height"] == 1080
    assert image["fail_if_other_publisher"] is True
    assert image["scan_mode"] == "auto"
    assert scan["scan_topic"] == "/scan"
    assert scan["fail_if_other_publisher"] is True
    assert scan["rate_hz"] == 10.0


def test_replay_launch_cannot_start_drive_nodes():
    launch = (ROOT / "launch" / "replay_perception_cnn.launch.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "cnn_supervisor",
        "cnn_drive_gate",
        "mission_route",
        "cnn_motion",
        "xycar_motor",
        "drive_cmd",
    ):
        assert forbidden not in launch
    assert 'executable="replay_images"' in launch
    assert 'executable="yolo_bev"' in launch
    assert 'executable="cnn_path"' in launch


def test_replay_executables_are_installed():
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    for executable in (
        "replay_images",
        "replay_clean_scan",
        "replay_recorder",
        "replay_preview",
    ):
        assert f'"{executable} = ' in setup


def test_spin_loop_returns_when_replay_requests_exit():
    class FakeNode:
        exit_requested = False
        spins = 0

    node = FakeNode()

    def spin_once(fake, *, timeout_sec):
        assert timeout_sec == 0.01
        fake.spins += 1
        if fake.spins == 3:
            fake.exit_requested = True

    exited = _spin_until_exit(
        node,
        spin_once=spin_once,
        context_ok=lambda: True,
        timeout_sec=0.01,
    )
    assert exited is True
    assert node.spins == 3


def test_replay_completion_never_shuts_down_rclpy_inside_callback():
    source = (
        ROOT / "track_drive_cnn_gpt" / "replay_source_node.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    finish = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_finish"
    )
    called_attributes = {
        node.func.attr
        for node in ast.walk(finish)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "shutdown" not in called_attributes
    assert "_exit_deadline_mono" in ast.unparse(finish)


def test_replay_summary_reports_selected_path_and_route_intent():
    fake = SimpleNamespace(
        _output_path=Path("/tmp/result.jsonl"),
        _started_wall=1.0,
        _started_mono=0.0,
        _counts={"path_main": 2, "path_selected": 2},
        _main_stamps=[1_000_000_000, 2_000_000_000],
        _shortcut_stamps=[],
        _selected_stamps=[1_000_000_000, 2_000_000_000],
        _main_valid_stamps=[1_000_000_000, 2_000_000_000],
        _shortcut_valid_stamps=[],
        _selected_valid_stamps=[1_000_000_000, 2_000_000_000],
        _route_intent="shortcut",
        _route_intent_messages=2,
        _route_intent_transitions=1,
        _route_intent_counts={"main": 1, "shortcut": 1},
        _yolo_inference_ms=[],
        _yolo_total_ms=[],
        _cnn_total_ms=[],
        _cnn_ok=2,
        _cnn_reasons={},
        _rate=ReplayRecorderNode._rate,
        _latency=ReplayRecorderNode._latency,
    )
    summary = ReplayRecorderNode._summary(fake)
    assert summary["schema"] == "track_drive_replay_summary_v2_gpt"
    assert summary["valid_path_counts"]["selected"] == 2
    assert summary["selected_path_hz"] == pytest.approx(1.0)
    assert summary["selected_valid_path_hz"] == pytest.approx(1.0)
    assert summary["route_intent"] == {
        "last": "shortcut",
        "messages": 2,
        "transitions": 1,
        "counts": {"main": 1, "shortcut": 1},
    }


def test_preview_draws_selected_path_over_candidate_paths():
    fake = SimpleNamespace(_bev_scale=1)
    bev = np.zeros((128, 120, 3), dtype=np.uint8)
    empty = np.empty((0, 2), dtype=np.float32)
    selected = np.asarray([[1.0, 0.0]], dtype=np.float32)
    panel = ReplayPreviewNode._bev_panel(fake, bev, empty, empty, selected)
    row = int(np.floor((3.0 - 1.0) / 0.025))
    col = int(np.floor((1.5 - 0.0) / 0.025))
    assert tuple(panel[row, col]) == (255, 255, 0)


def test_replay_outputs_subscribe_to_direct_selected_path_and_route():
    recorder = (
        ROOT / "track_drive_cnn_gpt" / "replay_recorder_node.py"
    ).read_text(encoding="utf-8")
    preview = (
        ROOT / "track_drive_cnn_gpt" / "replay_preview_node.py"
    ).read_text(encoding="utf-8")
    for source in (recorder, preview):
        assert '"/center_path"' in source
        assert '"/route_intent"' in source
    assert '"path_selected"' in recorder
