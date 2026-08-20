#!/usr/bin/env python3
"""Live camera + LiDAR -> YOLO/BEV -> CNN paths, with no drive nodes.

This launch deliberately contains no supervisor, motion, ROS1 bridge, or
motor process.  It is the hardware-in-the-loop perception test entry point.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("track_drive_cnn_gpt")
    config_path = os.path.join(package_share, "config", "perception_cnn.yaml")
    sensors_launch = os.path.join(package_share, "launch", "sensors_gpt.launch.py")
    dds_profile = os.path.join(package_share, "config", "dds_shm_lan_gpt.xml")

    enable_camera = LaunchConfiguration("enable_camera")
    enable_lidar = LaunchConfiguration("enable_lidar")
    enable_imu = LaunchConfiguration("enable_imu")
    enable_yolo = LaunchConfiguration("enable_yolo")
    enable_cnn = LaunchConfiguration("enable_cnn")
    record_output = LaunchConfiguration("record_output")
    record_preview = LaunchConfiguration("record_preview")
    output_jsonl = LaunchConfiguration("output_jsonl")
    output_preview = LaunchConfiguration("output_preview")
    ros_domain_id = LaunchConfiguration("ros_domain_id")

    return LaunchDescription([
        DeclareLaunchArgument("enable_camera", default_value="true"),
        DeclareLaunchArgument("enable_lidar", default_value="true"),
        DeclareLaunchArgument("enable_imu", default_value="false"),
        DeclareLaunchArgument("enable_yolo", default_value="true"),
        DeclareLaunchArgument("enable_cnn", default_value="true"),
        DeclareLaunchArgument("record_output", default_value="true"),
        # Video encoding is useful for visual QA but changes the CPU benchmark.
        DeclareLaunchArgument("record_preview", default_value="false"),
        DeclareLaunchArgument(
            "output_jsonl", default_value="/tmp/track_drive_live_gpt.jsonl"
        ),
        DeclareLaunchArgument(
            "output_preview", default_value="/tmp/track_drive_live_preview_gpt.mp4"
        ),
        DeclareLaunchArgument("ros_domain_id", default_value="77"),
        SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", dds_profile),
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(sensors_launch),
            launch_arguments={
                "enable_camera": enable_camera,
                "enable_lidar": enable_lidar,
                "enable_imu": enable_imu,
                "ros_domain_id": ros_domain_id,
            }.items(),
        ),
        Node(
            package="track_drive_cnn_gpt",
            executable="yolo_bev",
            output="screen",
            condition=IfCondition(enable_yolo),
            parameters=[config_path],
        ),
        Node(
            package="track_drive_cnn_gpt",
            executable="cnn_path",
            output="screen",
            condition=IfCondition(enable_cnn),
            parameters=[config_path],
        ),
        Node(
            package="track_drive_cnn_gpt",
            executable="replay_recorder",
            output="screen",
            condition=IfCondition(record_output),
            parameters=[{"output_path": output_jsonl}],
        ),
        Node(
            package="track_drive_cnn_gpt",
            executable="replay_preview",
            output="screen",
            condition=IfCondition(record_preview),
            parameters=[{"output_path": output_preview}],
        ),
    ])
