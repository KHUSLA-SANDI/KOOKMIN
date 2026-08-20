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
    motion_config = os.path.join(package_share, "config", "motion_cnn.yaml")
    sensors_launch = os.path.join(package_share, "launch", "sensors_gpt.launch.py")
    dds_profile = os.path.join(package_share, "config", "dds_shm_lan_gpt.xml")
    legacy_car_config = PathJoinSubstitution([
        FindPackageShare("track_drive"), "config", "car.yaml"
    ])

    enable_sensors = LaunchConfiguration("enable_sensors")
    enable_motor = LaunchConfiguration("enable_motor")
    enable_supervisor = LaunchConfiguration("enable_supervisor")
    enable_drive = LaunchConfiguration("enable_drive")
    ros_domain_id = LaunchConfiguration("ros_domain_id")

    return LaunchDescription([
        DeclareLaunchArgument("enable_sensors", default_value="false"),
        DeclareLaunchArgument("enable_motor", default_value="false"),
        DeclareLaunchArgument("enable_supervisor", default_value="false"),
        DeclareLaunchArgument("enable_drive", default_value="false"),
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
                "enable_lidar": enable_sensors,
                "enable_imu": enable_sensors,
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
            parameters=[perception_config],
        ),
        Node(
            package="track_drive_cnn_gpt",
            executable="mission_route",
            output="screen",
            parameters=[perception_config],
        ),
        Node(
            package="track_drive_cnn_gpt",
            executable="cnn_supervisor",
            output="screen",
            condition=IfCondition(enable_supervisor),
            parameters=[
                perception_config,
                {"enable_drive": ParameterValue(enable_drive, value_type=bool)},
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
            # Isolated overlay: imports the installed vehicle MotionNode so its
            # steering calibration remains untouched, but replaces legacy
            # re-anchoring and receive-time path freshness for CNN paths.
            package="track_drive_cnn_gpt",
            executable="cnn_motion",
            output="screen",
            condition=IfCondition(enable_motor),
            parameters=[motion_config, legacy_car_config],
        ),
    ])
