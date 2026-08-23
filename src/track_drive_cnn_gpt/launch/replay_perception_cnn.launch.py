#!/usr/bin/env python3
"""Recorded camera -> YOLO/BEV -> LiDAR/CNN replay, with no drive nodes."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory("track_drive_cnn_gpt")
    perception_config = os.path.join(package_share, "config", "perception_cnn.yaml")
    replay_config = os.path.join(package_share, "config", "replay_gpt.yaml")
    dds_profile = os.path.join(package_share, "config", "dds_shm_lan_gpt.xml")

    source_path = LaunchConfiguration("source_path")
    replay_fps = LaunchConfiguration("replay_fps")
    playback_rate = LaunchConfiguration("playback_rate")
    scan_mode = LaunchConfiguration("scan_mode")
    loop = LaunchConfiguration("loop")
    start_index = LaunchConfiguration("start_index")
    max_frames = LaunchConfiguration("max_frames")
    exit_when_done = LaunchConfiguration("exit_when_done")
    enable_yolo = LaunchConfiguration("enable_yolo")
    enable_cnn = LaunchConfiguration("enable_cnn")
    enable_viewer = LaunchConfiguration("enable_viewer")
    viewer_show_window = LaunchConfiguration("viewer_show_window")
    viewer_publish_compressed = LaunchConfiguration("viewer_publish_compressed")
    record_output = LaunchConfiguration("record_output")
    record_preview = LaunchConfiguration("record_preview")
    output_jsonl = LaunchConfiguration("output_jsonl")
    output_preview = LaunchConfiguration("output_preview")
    yolo_model_path = LaunchConfiguration("yolo_model_path")
    cnn_model_path = LaunchConfiguration("cnn_model_path")
    general_model_path = LaunchConfiguration("general_model_path")
    shortcut_model_path = LaunchConfiguration("shortcut_model_path")
    overtake_model_path = LaunchConfiguration("overtake_model_path")
    cone_model_path = LaunchConfiguration("cone_model_path")
    camera_yaml = LaunchConfiguration("camera_yaml")
    ros_domain_id = LaunchConfiguration("ros_domain_id")

    replay_node = Node(
        package="track_drive_cnn_gpt",
        executable="replay_images",
        output="screen",
        parameters=[replay_config, {
            "source_path": source_path,
            "fps": replay_fps,
            "playback_rate": playback_rate,
            "scan_mode": scan_mode,
            "loop": loop,
            "start_index": start_index,
            "max_frames": max_frames,
            "exit_when_done": exit_when_done,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "source_path",
            default_value="/home/xytron/recordings/20260818_211400",
        ),
        DeclareLaunchArgument("replay_fps", default_value="30.0"),
        DeclareLaunchArgument("playback_rate", default_value="1.0"),
        DeclareLaunchArgument("scan_mode", default_value="auto"),
        DeclareLaunchArgument("loop", default_value="true"),
        DeclareLaunchArgument("start_index", default_value="0"),
        DeclareLaunchArgument("max_frames", default_value="0"),
        DeclareLaunchArgument("exit_when_done", default_value="false"),
        DeclareLaunchArgument("enable_yolo", default_value="true"),
        DeclareLaunchArgument("enable_cnn", default_value="true"),
        DeclareLaunchArgument("enable_viewer", default_value="false"),
        DeclareLaunchArgument("viewer_show_window", default_value="false"),
        DeclareLaunchArgument("viewer_publish_compressed", default_value="false"),
        DeclareLaunchArgument("record_output", default_value="true"),
        # MP4 encoding consumes CPU, so leave it off for latency benchmarks.
        DeclareLaunchArgument("record_preview", default_value="false"),
        DeclareLaunchArgument(
            "output_jsonl", default_value="/tmp/track_drive_replay_gpt.jsonl"
        ),
        DeclareLaunchArgument(
            "output_preview",
            default_value="/tmp/track_drive_replay_preview_gpt.mp4",
        ),
        DeclareLaunchArgument(
            "yolo_model_path",
            default_value="/home/xytron/xycar_ws/models/best_v3_gpt_openvino_model",
        ),
        DeclareLaunchArgument(
            "cnn_model_path",
            default_value="/home/xytron/xycar_ws/models/path_cnn.pt",
        ),
        DeclareLaunchArgument(
            "general_model_path",
            default_value="/home/xytron/xycar_ws/models/general_single_best_gpt.pt",
        ),
        DeclareLaunchArgument(
            "shortcut_model_path",
            default_value="/home/xytron/xycar_ws/models/left_shortcut_single_best_gpt.pt",
        ),
        DeclareLaunchArgument(
            "overtake_model_path",
            default_value="/home/xytron/xycar_ws/models/obstacle_single_best_gpt.pt",
        ),
        DeclareLaunchArgument(
            "cone_model_path",
            default_value="/home/xytron/xycar_ws/models/cone_single_all967_final_gpt.pt",
        ),
        DeclareLaunchArgument(
            "camera_yaml",
            default_value=os.path.join(package_share, "config", "camera.yaml"),
        ),
        # Isolate replay from normal live domain 7 unless explicitly overridden.
        DeclareLaunchArgument("ros_domain_id", default_value="17"),
        SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", dds_profile),
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
        replay_node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=replay_node,
                on_exit=[EmitEvent(event=Shutdown(reason="replay completed"))],
            ),
            condition=IfCondition(exit_when_done),
        ),
        Node(
            package="track_drive_cnn_gpt",
            executable="yolo_bev",
            output="screen",
            condition=IfCondition(enable_yolo),
            parameters=[perception_config, {
                "model_path": yolo_model_path,
                "camera_yaml": camera_yaml,
                # More windows make the short replay latency summary useful.
                "diag_period_sec": 0.5,
            }],
        ),
        Node(
            package="track_drive_cnn_gpt",
            executable="cnn_path",
            output="screen",
            condition=IfCondition(enable_cnn),
            parameters=[perception_config, {
                # Keep the old single-model override for backward-compatible
                # checkpoints, while exposing the four production models that
                # take precedence in the current node.
                "model_path": cnn_model_path,
                "general_model_path": general_model_path,
                "shortcut_model_path": shortcut_model_path,
                "overtake_model_path": overtake_model_path,
                "cone_model_path": cone_model_path,
            }],
        ),
        Node(
            package="track_drive_cnn_gpt",
            executable="live_pipeline_viewer",
            output="screen",
            condition=IfCondition(enable_viewer),
            parameters=[perception_config, {
                "show_window": ParameterValue(viewer_show_window, value_type=bool),
                "publish_compressed": ParameterValue(
                    viewer_publish_compressed, value_type=bool
                ),
            }],
        ),
        Node(
            package="track_drive_cnn_gpt",
            executable="replay_recorder",
            output="screen",
            condition=IfCondition(record_output),
            parameters=[replay_config, {"output_path": output_jsonl}],
        ),
        Node(
            package="track_drive_cnn_gpt",
            executable="replay_preview",
            output="screen",
            condition=IfCondition(record_preview),
            parameters=[replay_config, {"output_path": output_preview}],
        ),
    ])
