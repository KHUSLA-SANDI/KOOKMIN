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
    "595628d4c08bed8ddc59ad7e53bd5393084256f74b80815994544f957935d9f8"
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
    assert params["general_expected_sha256"] == (
        "e90b0de551fdcdde5a5de96d754365380f49db412242535313c396a77201c1b1"
    )
    assert params["shortcut_model_path"].endswith("left_shortcut_single_best_gpt.pt")
    assert params["shortcut_expected_sha256"] == (
        "88dc9a1da262c2622e2b618f1fb6789aa5027f3fc3f9bc03c04dcdbef648ae1a"
    )
    assert params["overtake_model_path"].endswith(
        "obstacle_single_main508_best_gpt.pt"
    )
    assert params["overtake_expected_sha256"] == (
        "87fd7a0363d8544a1801b446a5ca16bcfcfd483ea52a8921255eb98562751003"
    )
    assert params["cone_model_path"].endswith("cone_single_all967_final_gpt.pt")
    assert params["cone_expected_sha256"] == (
        "4844b66bb4440580b2cf39cc6ac7e5e8cc9538d3c1a13579f5267094abb18d13"
    )
    assert params["shortcut_hold_sec"] == 13.0
    assert params["overtake_hold_sec"] == 3.0
    assert params["cone_hold_sec"] == 10.0
    assert params["overtake_inward_margin_m"] == 0.10
    assert params["overtake_x_min_m"] == 0.0
    assert params["overtake_x_max_m"] == 1.40
    assert params["overtake_strategy"] == "hardcoded_all"
    assert params["overtake_block_request_topic"] == (
        "/motion/overtake_block_request"
    )
    assert params["obstacle_lane_deadband_m"] == 0.10
    assert params["post_cone_hardcode_window_sec"] == 5.0
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
    assert 'DeclareLaunchArgument("speed_cap", default_value="50.0")' in text
    assert 'executable="cnn_drive_gate"' in text
    assert 'DeclareLaunchArgument("enable_viewer", default_value="false")' in text
    assert 'executable="live_pipeline_viewer"' in text
    assert '"motion_topic": "/xycar_motor"' in text
    assert 'viewer_publish_compressed, value_type=bool' in text
    assert 'executable="cnn_supervisor"' not in text
    assert 'executable="mission_route"' not in text
    assert 'executable="motion"' not in text


def test_simple_motion_has_four_independent_profiles():
    config = yaml.safe_load(
        (ROOT / "config" / "simple_motion.yaml").read_text(encoding="utf-8")
    )["motion_node"]["ros__parameters"]
    prefixes = ("", "shortcut_", "overtake_", "cone_")
    for prefix in prefixes:
        lookahead = float(config[f"{prefix}lookahead_m"])
        gain = float(config[f"{prefix}steer_gain"])
        alpha = float(config[f"{prefix}steer_smooth_alpha"])
        speed = float(config[f"{prefix}speed_cmd"])
        assert 0.3 <= lookahead <= 3.0
        assert gain > 0.0
        assert 0.0 < alpha <= 1.0
        assert 0.0 < speed <= 50.0
    assert config["cnn_mode_topic"] == "/cnn_mode"
    assert config["cone_approach_topic"] == "/perception/cone_approach"
    assert config["cone_approach_speed_cmd"] == 12.0
    assert config["control_hz"] == 20.0
    assert config["debug_hz"] == 10.0
    assert config["overtake_strategy"] == "hardcoded_all"
    assert config["overtake_block_request_topic"] == (
        "/motion/overtake_block_request"
    )
    assert config["hardcoded_block_lane_change_speed_cmd"] == 16.0
    assert config["hardcoded_block_pass_speed_cmd"] == 16.0
    assert config["hardcoded_block_pass_ticks"] == 20
    assert config["hardcoded_block_right_shift_angle_cmd"] == 60.0
    assert config["hardcoded_block_right_shift_ticks"] == 6
    assert config["hardcoded_block_right_counter_angle_cmd"] == -60.0
    assert config["hardcoded_block_right_counter_ticks"] == 10
    assert config["hardcoded_block_left_shift_angle_cmd"] == -60.0
    assert config["hardcoded_block_left_shift_ticks"] == 7
    assert config["hardcoded_block_left_counter_angle_cmd"] == 60.0
    assert config["hardcoded_block_left_counter_ticks"] == 10
    assert config["path_stale_sec"] == 1.0
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
    assert config["path_stale_sec"] == 1.0
    assert config["path_future_tolerance_sec"] == 0.05
    assert config["enable_drive"] is False
    assert config["initial_manual_go"] is False
    assert config["race_go_topic"] == "/race_go"

    cnn = yaml.safe_load(
        (ROOT / "config" / "perception_cnn.yaml").read_text(encoding="utf-8")
    )["cnn_path_node"]["ros__parameters"]
    assert cnn["race_go_topic"] == "/race_go"
    assert cnn["race_finish_enabled"] is False
    assert cnn["race_target_laps"] == 3
    assert cnn["race_lap_cooldown_sec"] == pytest.approx(10.0)
    assert cnn["race_lap_clear_frames"] == 5


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
    assert yolo["model_path"] == (
        "/home/xytron/xycar_ws/models/traffic_detector_seg_best_openvino_model"
    )
    assert yolo["expected_model_sha256"] == (
        "6ead43178acb34a1006c2c9d5b21a73f9d2500a9dcfc8f8ee8229c0c4e7acc59"
    )
    assert yolo["classifier_model_path"].endswith(
        "/yolo11n_cls_best_openvino_model"
    )
    assert yolo["classifier_expected_model_sha256"] == (
        "2c16877f7fe43231ae08ccbfce6c8a9388d17aeb50a01dc21f6359f925f8e959"
    )
    assert yolo["classifier_imgsz"] == [320, 320]
    assert yolo["traffic_light_class_name"] == "traffic_light"
    assert yolo["signal_topic"] == "/perception/signals"
    assert yolo["yolo_state_topic"] == "/debug/yolo_state"
    assert yolo["cone_mode_topic"] == "/perception/cone_trigger"
    assert yolo["start_r_detected_topic"] == "/perception/start_r_detected"
    assert yolo["cone_approach_topic"] == "/perception/cone_approach"
    assert yolo["cone_approach_bottom_y_min"] == 0.70
    assert yolo["cone_approach_bottom_y_min"] < yolo["cone_enter_bottom_y_min"]
    assert yolo["cone_trigger_class_name"] == "START_R"
    assert cnn["signal_topic"] == yolo["signal_topic"]
    assert cnn["cone_trigger_topic"] == yolo["cone_mode_topic"]
    assert cnn["start_r_detected_topic"] == yolo["start_r_detected_topic"]
    assert cnn["cnn_mode_topic"] == "/cnn_mode"
    assert cnn["cnn_input_bev_topic"] == "/debug/cnn_input_bev"
    assert cnn["center_path_topic"] == "/center_path"
    assert cnn["left_confirm_frames"] >= 2
    assert cnn["enable_center_path"] is True
    assert gate["center_path_topic"] == cnn["center_path_topic"]
    assert gate["enable_drive"] is False

    viewer = config["live_pipeline_viewer_node"]["ros__parameters"]
    assert viewer["cnn_input_bev_topic"] == cnn["cnn_input_bev_topic"]
    assert viewer["cone_trigger_topic"] == cnn["cone_trigger_topic"]
    assert viewer["yolo_state_topic"] == yolo["yolo_state_topic"]
    assert viewer["cone_enter_bottom_y_min"] == yolo["cone_enter_bottom_y_min"]
    assert viewer["cone_approach_bottom_y_min"] == yolo["cone_approach_bottom_y_min"]
    assert viewer["signal_display_floor"] == yolo["signal_conf"]
    assert viewer["overtake_min_raw_points"] == cnn["overtake_min_raw_points"]

    cnn_source = (ROOT / "track_drive_cnn_gpt" / "cnn_path_node.py").read_text(
        encoding="utf-8"
    )
    assert 'if mode_state.mode == MODE_CONE:' in cnn_source
    assert 'lidar = raw_lidar_grid' in cnn_source
    assert 'lidar_preprocess = "outside_white_removed"' in cnn_source

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
    assert 'DeclareLaunchArgument("enable_viewer", default_value="true")' in source
    assert 'condition=IfCondition(enable_viewer)' in source
    assert '"viewer_publish_compressed", default_value="false"' in source
    assert '("/drive_cmd", "/debug/drive_cmd_dryrun")' in source
    assert '("/xycar_motor", "/debug/xycar_motor_dryrun")' in source
    assert '"initial_manual_go": False' in source
    assert 'executable="car_state"' not in source
    assert "dynamic_bridge" not in source
    assert "motor_up" not in source

    viewer = (
        ROOT / "track_drive_cnn_gpt" / "live_pipeline_viewer_node.py"
    ).read_text(encoding="utf-8")
    assert '"/debug/xycar_motor_dryrun"' in viewer
    assert 'p("cone_mode_topic", "/cone_mode")' in viewer
    assert "self._on_cone_mode" in viewer
    assert 'p("traffic_stop_topic", "/traffic_stop")' in viewer
    assert "self._on_traffic_stop" in viewer
    assert 'p("yolo_state_topic", "/debug/yolo_state")' in viewer
    assert "self._on_yolo_state" in viewer
    assert '"GENERAL": (40, 255, 40)' in viewer
    assert '"CONE": (255, 255, 0)' in viewer
    assert '"OVERTAKE": (30, 30, 255)' in viewer
    assert '"SHORTCUT": (255, 80, 255)' in viewer
    assert 'signal_label = "<- LEFT"' in viewer
    assert 'f"{signal_label}={confidence:.2f}"' in viewer
    assert 'f"{signal_label}<{self._signal_display_floor:.2f}"' in viewer
    assert 'f"MODE={cnn_mode}"' in viewer
    assert 'f"Y2={cone_y2:.3f} ({cone_y2_px:.0f}px)"' in viewer
    assert 'f"SLOW={int(cone_approach_latch_active)} "' in viewer
    assert 'f"CONE={int(cone_trigger_raw or cone_state_latch_active)} "' in viewer
    assert 'f"OVERTAKE={overtake_steps}/5"' in viewer
    assert "create_publisher(\n            CompressedImage" in viewer
    assert "create_publisher(\n            Float32MultiArray" not in viewer

    config = yaml.safe_load(
        (ROOT / "config" / "perception_cnn.yaml").read_text(encoding="utf-8")
    )
    assert (
        config["live_pipeline_viewer_node"]["ros__parameters"][
            "publish_compressed"
        ]
        is False
    )


def test_signal_test_uses_read_only_live_viewer_without_drive_nodes():
    source = (ROOT / "launch" / "signal_test.launch.py").read_text(
        encoding="utf-8"
    )
    assert 'executable="yolo_bev"' in source
    assert 'executable="live_pipeline_viewer"' in source
    assert 'executable="signal_preview"' not in source
    assert '"publish_signal_preview": True' in source
    assert '"publish_compressed": False' in source
    assert '"motion_topic": "/debug/xycar_motor_signal_test_unused"' in source
    for forbidden in (
        'executable="cnn_path"',
        'executable="cnn_drive_gate"',
        'executable="simple_motion"',
        'executable="cnn_motion"',
        '"/xycar_motor"',
        "dynamic_bridge",
        "motor_up",
    ):
        assert forbidden not in source
