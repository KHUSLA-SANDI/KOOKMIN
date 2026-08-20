from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_ROOT = ROOT.parents[1]
AUDITED_MOTION_SHA256 = (
    "1f2d3e6c2ea58073f0c6d917d587109d4055f4f6ec20153fb511c94d6cad089e"
)


def test_package_xml_and_installed_config_patterns():
    root = ET.parse(ROOT / "package.xml").getroot()
    assert root.findtext("name") == "track_drive_cnn_gpt"
    assert "v4l-utils" in {node.text for node in root.findall("exec_depend")}
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert 'glob("config/*.xml")' in setup_text
    assert 'glob("config/*.yaml")' in setup_text
    assert 'glob("tools/*.py")' in setup_text
    assert '"requirements_models_gpt.txt"' in setup_text
    assert (
        '"cnn_motion = track_drive_cnn_gpt.motion_cnn_node:main"'
        in setup_text
    )


def test_runtime_grid_config_is_the_frozen_contract():
    config = yaml.safe_load(
        (ROOT / "config" / "perception_cnn.yaml").read_text(encoding="utf-8")
    )
    params = config["cnn_path_node"]["ros__parameters"]
    yolo_params = config["yolo_bev_node"]["ros__parameters"]
    assert yolo_params["imgsz"] == [384, 640]
    assert params["lidar_range_scale"] == 1.1476
    assert params["lidar_self_x_abs"] == 0.25
    assert params["lidar_self_y_abs"] == 0.15
    assert params["allow_unverified_legacy"] is False
    assert params["require_fresh_scan"] is True
    assert params["max_bev_age_sec"] == 0.35
    assert params["shortcut_threshold"] == 0.50
    assert params["min_camera_cells"] == 1
    assert params["expected_model_sha256"] == (
        "439937eb506fe0a25c8d9f225abc2f10b4de96d388099fe4b2570598a5c03a67"
    )
    assert params["lidar_preprocess_mode"] == "compact_reference"
    assert params["min_lidar_cells"] == 27
    assert params["max_lidar_cells"] == 67
    assert params["lidar_max_clusters"] == 1
    assert params["max_abs_y_m"] >= 1.5
    assert params["max_abs_slope"] == 3.1
    assert params["max_abs_curvature"] == 12.0


def test_shm_segment_can_hold_multiple_full_hd_rgb_frames():
    root = ET.parse(ROOT / "config" / "dds_shm_lan_gpt.xml").getroot()
    ns = {"f": "http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles"}
    segment = root.find(".//f:transport_descriptor[f:type='SHM']/f:segment_size", ns)
    assert segment is not None
    # One 1920x1080 rgb8 frame is 6,220,800 bytes.  Keep ample room for
    # writer/reader histories instead of relying on Fast DDS's small default.
    assert int(segment.text) >= 8 * 1920 * 1080 * 3


def test_launches_never_start_legacy_path_planner():
    for launch_file in (ROOT / "launch").glob("*.launch.py"):
        text = launch_file.read_text(encoding="utf-8")
        assert 'executable="path_planner"' not in text


def test_live_path_only_launch_has_sensors_and_perception_but_no_drive_chain():
    text = (ROOT / "launch" / "live_path_only.launch.py").read_text(
        encoding="utf-8"
    )
    assert 'default_value="true"' in text
    assert 'executable="yolo_bev"' in text
    assert 'executable="cnn_path"' in text
    assert 'DeclareLaunchArgument("enable_imu", default_value="false")' in text
    for forbidden in (
        'executable="cnn_supervisor"',
        'executable="cnn_motion"',
        'executable="motion"',
        "dynamic_bridge",
        "motor_up",
    ):
        assert forbidden not in text


def test_low_speed_launch_uses_isolated_cnn_motion_overlay():
    text = (ROOT / "launch" / "drive_low_speed.launch.py").read_text(
        encoding="utf-8"
    )
    assert 'package="track_drive_cnn_gpt"' in text
    assert 'executable="cnn_motion"' in text
    assert 'executable="motion"' not in text


def test_motion_overlay_is_pinned_to_the_audited_vehicle_source():
    candidates = (
        INTEGRATION_ROOT / "vehicle_snapshot_20260819_gpt" / "motion_node.py",
        Path("/home/xytron/xycar_ws/src/track_drive/track_drive/motion_node.py"),
    )
    snapshot = next((path for path in candidates if path.is_file()), None)
    if snapshot is None:
        pytest.skip("audited vehicle MotionNode source is not available here")
    actual = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert actual == AUDITED_MOTION_SHA256

    config = yaml.safe_load(
        (ROOT / "config" / "motion_cnn.yaml").read_text(encoding="utf-8")
    )["motion_node"]["ros__parameters"]
    assert config["expected_vehicle_motion_sha256"] == AUDITED_MOTION_SHA256
    assert config["verify_vehicle_motion_source"] is True
    assert config["cnn_path_frame_id"] == "lidar_frame"
    assert config["lookahead_x_min"] == 0.3
    assert config["lookahead_x_max"] == 3.0


def test_supervisor_requires_source_frame_and_timestamp_policy():
    config = yaml.safe_load(
        (ROOT / "config" / "perception_cnn.yaml").read_text(encoding="utf-8")
    )["cnn_supervisor_node"]["ros__parameters"]
    assert config["path_frame_id"] == "lidar_frame"
    assert config["path_stale_sec"] == 0.25
    assert config["path_future_tolerance_sec"] == 0.05


def test_camera_does_not_respawn_forever_on_device_busy():
    text = (ROOT / "launch" / "sensors_gpt.launch.py").read_text(encoding="utf-8")
    assert "respawn=False" in text


def test_rectangular_openvino_and_signal_mission_contract_are_configured():
    config = yaml.safe_load(
        (ROOT / "config" / "perception_cnn.yaml").read_text(encoding="utf-8")
    )
    yolo = config["yolo_bev_node"]["ros__parameters"]
    mission = config["mission_route_node"]["ros__parameters"]
    assert yolo["imgsz"] == [384, 640]
    assert yolo["signal_topic"] == "/perception/signals"
    assert mission["signal_topic"] == yolo["signal_topic"]
    assert mission["route_intent_topic"] == "/route_intent"
    assert mission["left_confirm_frames"] >= 2

    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "mission_route = track_drive_cnn_gpt.mission_route_node:main" in setup_text


def test_all_yolo_inference_entrypoints_force_static_batch_one():
    paths = (
        ROOT / "track_drive_cnn_gpt" / "yolo_bev_node.py",
        ROOT / "tools" / "benchmark_yolo_backends_gpt.py",
        ROOT / "tools" / "smoke_yolo_bev_gpt.py",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "batch=1" in source, path


def test_all_ros_nodes_guard_shutdown_after_launch_sigint():
    paths = (
        "yolo_bev_node.py",
        "mission_route_node.py",
        "cnn_path_node.py",
        "supervisor_node.py",
        "motion_cnn_node.py",
    )
    for name in paths:
        source = (ROOT / "track_drive_cnn_gpt" / name).read_text(encoding="utf-8")
        assert "if rclpy.ok():" in source, name
