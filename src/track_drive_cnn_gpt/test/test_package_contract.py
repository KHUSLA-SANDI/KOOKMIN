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
AUDITED_CAR_INTERFACE_SHA256 = (
    "6554c716957cc93eb77b626bca3554653708b130640a6860da55d8811b889091"
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
    assert (
        '"simple_motion = track_drive_cnn_gpt.simple_motion_node:main"'
        in setup_text
    )
    assert (
        '"live_pipeline_viewer = track_drive_cnn_gpt.live_pipeline_viewer_node:main"'
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
        'executable="cnn_drive_gate"',
        'executable="mission_route"',
        'executable="cnn_motion"',
        'executable="motion"',
        "dynamic_bridge",
        "motor_up",
    ):
        assert forbidden not in text


def test_low_speed_launch_uses_minimal_motion_with_audited_car_yaml():
    text = (ROOT / "launch" / "drive_low_speed.launch.py").read_text(
        encoding="utf-8"
    )
    assert 'package="track_drive_cnn_gpt"' in text
    assert 'executable="simple_motion"' in text
    assert '"config", "simple_motion.yaml"' in text
    assert "legacy_car_config" in text
    assert 'DeclareLaunchArgument("speed_cap", default_value="5.0")' in text
    assert 'executable="cnn_drive_gate"' in text
    assert 'executable="cnn_supervisor"' not in text
    assert 'executable="mission_route"' not in text
    assert 'executable="motion"' not in text


def test_simple_motion_has_only_three_control_tuning_values():
    config = yaml.safe_load(
        (ROOT / "config" / "simple_motion.yaml").read_text(encoding="utf-8")
    )["motion_node"]["ros__parameters"]
    assert config["lookahead_m"] == 1.0
    assert config["steer_gain"] == 0.45
    assert config["steer_smooth_alpha"] == 0.40
    assert config["control_hz"] == 20.0
    assert config["path_stale_sec"] == 0.25
    assert config["verify_car_interface_source"] is True
    assert config["expected_car_interface_sha256"] == AUDITED_CAR_INTERFACE_SHA256


def test_simple_motion_uses_the_audited_vehicle_steering_calibration():
    source = (
        INTEGRATION_ROOT / "vehicle_snapshot_20260819_gpt" / "car_interface.py"
    )
    config_path = INTEGRATION_ROOT / "vehicle_snapshot_20260819_gpt" / "car.yaml"
    if not source.is_file() or not config_path.is_file():
        pytest.skip("audited vehicle CarInterface snapshot is not available")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == AUDITED_CAR_INTERFACE_SHA256

    car = yaml.safe_load(config_path.read_text(encoding="utf-8"))["motion_node"][
        "ros__parameters"
    ]
    assert car["steer_trim"] == 0.0
    assert car["steer_scale_left"] == pytest.approx(0.625933146)
    assert car["steer_scale_right"] == pytest.approx(0.585923661)
    assert car["steer_limit_left"] == pytest.approx(-62.593314622)
    assert car["steer_limit_right"] == pytest.approx(58.592366078)


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


def test_drive_gate_requires_source_frame_and_timestamp_policy():
    config = yaml.safe_load(
        (ROOT / "config" / "perception_cnn.yaml").read_text(encoding="utf-8")
    )["cnn_drive_gate_node"]["ros__parameters"]
    assert config["path_frame_id"] == "lidar_frame"
    assert config["path_stale_sec"] == 0.25
    assert config["path_future_tolerance_sec"] == 0.05
    assert config["enable_drive"] is False
    assert config["initial_manual_go"] is False


def test_camera_does_not_respawn_forever_on_device_busy():
    text = (ROOT / "launch" / "sensors_gpt.launch.py").read_text(encoding="utf-8")
    assert "respawn=False" in text


def test_rectangular_openvino_and_direct_signal_selection_are_configured():
    config = yaml.safe_load(
        (ROOT / "config" / "perception_cnn.yaml").read_text(encoding="utf-8")
    )
    yolo = config["yolo_bev_node"]["ros__parameters"]
    cnn = config["cnn_path_node"]["ros__parameters"]
    gate = config["cnn_drive_gate_node"]["ros__parameters"]
    assert yolo["imgsz"] == [384, 640]
    assert yolo["signal_topic"] == "/perception/signals"
    assert cnn["signal_topic"] == yolo["signal_topic"]
    assert cnn["center_path_topic"] == "/center_path"
    assert cnn["left_confirm_frames"] >= 2
    assert cnn["enable_center_path"] is True
    assert gate["center_path_topic"] == cnn["center_path_topic"]
    assert gate["enable_drive"] is False

    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "cnn_drive_gate = track_drive_cnn_gpt.drive_gate_node:main" in setup_text
    assert "mission_route = track_drive_cnn_gpt.mission_route_node:main" not in setup_text
    assert "cnn_supervisor = track_drive_cnn_gpt.supervisor_node:main" not in setup_text
    assert "signal_preview = track_drive_cnn_gpt.signal_preview_node:main" not in setup_text


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
        "cnn_path_node.py",
        "drive_gate_node.py",
        "motion_cnn_node.py",
        "simple_motion_node.py",
        "live_pipeline_viewer_node.py",
    )
    for name in paths:
        source = (ROOT / "track_drive_cnn_gpt" / name).read_text(encoding="utf-8")
        assert "if rclpy.ok():" in source, name


def test_obsolete_route_nodes_and_signal_only_viewer_are_removed():
    runtime = ROOT / "track_drive_cnn_gpt"
    tests = ROOT / "test"
    for obsolete in (
        runtime / "mission_route_node.py",
        runtime / "supervisor_node.py",
        runtime / "signal_preview_node.py",
        tests / "test_supervisor_logic.py",
        tests / "test_signal_preview.py",
    ):
        assert not obsolete.exists(), obsolete
    gate_source = (runtime / "drive_gate_node.py").read_text(encoding="utf-8")
    assert "from .supervisor_node" not in gate_source
    assert "def encode_drive_command" in gate_source


def test_live_dry_run_uses_minimal_motion_with_hard_motor_topic_isolation():
    source = (ROOT / "launch" / "live_pipeline_dry_run.launch.py").read_text(
        encoding="utf-8"
    )
    assert 'executable="yolo_bev"' in source
    assert 'executable="cnn_path"' in source
    assert 'executable="cnn_drive_gate"' in source
    assert 'executable="simple_motion"' in source
    assert '"config", "simple_motion.yaml"' in source
    assert 'DeclareLaunchArgument("speed_cap", default_value="5.0")' in source
    assert 'executable="live_pipeline_viewer"' in source
    assert '("/drive_cmd", "/debug/drive_cmd_dryrun")' in source
    assert '("/xycar_motor", "/debug/xycar_motor_dryrun")' in source
    assert '"initial_manual_go": True' in source
    assert 'executable="car_state"' not in source
    assert "dynamic_bridge" not in source
    assert "motor_up" not in source

    viewer = (
        ROOT / "track_drive_cnn_gpt" / "live_pipeline_viewer_node.py"
    ).read_text(encoding="utf-8")
    assert '"/debug/xycar_motor_dryrun"' in viewer
    assert "create_publisher(\n            CompressedImage" in viewer
    assert "create_publisher(\n            Float32MultiArray" not in viewer
