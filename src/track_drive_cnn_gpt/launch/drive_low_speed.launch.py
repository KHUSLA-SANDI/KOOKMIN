#!/usr/bin/env python3
"""Low-speed CNN drive launch with independent hardware and software gates."""

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
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = get_package_share_directory("track_drive_cnn_gpt")
    perception_config = os.path.join(
        package_share, "config", "perception_cnn.yaml"
    )
    motion_config = os.path.join(package_share, "config", "simple_motion.yaml")
    sensors_launch = os.path.join(package_share, "launch", "sensors_gpt.launch.py")
    dds_profile = os.path.join(package_share, "config", "dds_shm_lan_gpt.xml")
    legacy_car_config = PathJoinSubstitution([
        FindPackageShare("track_drive"), "config", "car.yaml"
    ])

    enable_sensors = LaunchConfiguration("enable_sensors")
    enable_lidar = LaunchConfiguration("enable_lidar")
    enable_motor = LaunchConfiguration("enable_motor")
    enable_drive_gate = LaunchConfiguration("enable_drive_gate")
    enable_drive = LaunchConfiguration("enable_drive")
    speed_cap = LaunchConfiguration("speed_cap")
    enable_viewer = LaunchConfiguration("enable_viewer")
    viewer_show_window = LaunchConfiguration("viewer_show_window")
    viewer_publish_compressed = LaunchConfiguration("viewer_publish_compressed")
    ros_domain_id = LaunchConfiguration("ros_domain_id")

    return LaunchDescription([
        DeclareLaunchArgument("enable_sensors", default_value="false"),
        DeclareLaunchArgument("enable_lidar", default_value=enable_sensors),
        DeclareLaunchArgument("enable_motor", default_value="false"),
        # The selected path goes directly from cnn_path to simple_motion.  This
        # independent gate refreshes only the legacy /drive_cmd watchdog and
        # remains explicitly disabled by default.
        DeclareLaunchArgument("enable_drive_gate", default_value="false"),
        DeclareLaunchArgument("enable_drive", default_value="false"),
        # Actual NORMAL/CONE speeds live in simple_motion.yaml.  This launch
        # value is only a global upper bound carried by /drive_cmd.
        DeclareLaunchArgument("speed_cap", default_value="50.0"),
        # Viewer is an optional read-only diagnostic consumer.  Keeping the
        # process absent (not merely hiding its window) removes all render cost.
        DeclareLaunchArgument("enable_viewer", default_value="false"),
        DeclareLaunchArgument("viewer_show_window", default_value="true"),
        DeclareLaunchArgument(
            "viewer_publish_compressed", default_value="false"
        ),
        DeclareLaunchArgument("ros_domain_id", default_value="7"),
        SetEnvironmentVariable(
            "FASTRTPS_DEFAULT_PROFILES_FILE", dds_profile
        ),
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(sensors_launch),
            launch_arguments={
                "enable_camera": enable_sensors,
                "enable_lidar": enable_lidar,
                "enable_imu": "false",
                "ros_domain_id": ros_domain_id,
            }.items(),
        ),
        Node(
            package="track_drive_cnn_gpt",
            executable="yolo_bev",
            output="screen",
            parameters=[perception_config],
        ),
        Node(
            package="track_drive_cnn_gpt",
            executable="cnn_path",
            output="screen",
            parameters=[
                perception_config,
                {
                    "require_fresh_scan": ParameterValue(
                        enable_lidar, value_type=bool
                    ),
                },
            ],
        ),
        Node(
            package="track_drive_cnn_gpt",
            executable="cnn_drive_gate",
            output="screen",
            condition=IfCondition(enable_drive_gate),
            parameters=[
                perception_config,
                {
                    "enable_drive": ParameterValue(enable_drive, value_type=bool),
                    "speed_cap": ParameterValue(speed_cap, value_type=float),
                },
            ],
        ),
        Node(
            package="track_drive",
            executable="car_state",
            output="screen",
            condition=IfCondition(enable_motor),
            parameters=[legacy_car_config],
        ),
        Node(
            # Minimal low-speed follower.  It imports only the installed
            # CarInterface and receives the audited car.yaml below, so the
            # measured left/right steering balance remains unchanged.
            package="track_drive_cnn_gpt",
            executable="simple_motion",
            output="screen",
            condition=IfCondition(enable_motor),
            parameters=[motion_config, legacy_car_config],
        ),
        Node(
            package="track_drive_cnn_gpt",
            executable="live_pipeline_viewer",
            output="screen",
            condition=IfCondition(enable_viewer),
            parameters=[
                perception_config,
                {
                    "motion_topic": "/xycar_motor",
                    "show_window": ParameterValue(
                        viewer_show_window, value_type=bool
                    ),
                    "publish_compressed": ParameterValue(
                        viewer_publish_compressed, value_type=bool
                    ),
                },
            ],
        ),
    ])
