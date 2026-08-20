#!/usr/bin/env python3
"""Run gated perception without motion; CNN stays off until its checkpoint exists."""

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

    enable_sensors = LaunchConfiguration("enable_sensors")
    enable_yolo = LaunchConfiguration("enable_yolo")
    enable_cnn = LaunchConfiguration("enable_cnn")
    ros_domain_id = LaunchConfiguration("ros_domain_id")

    return LaunchDescription([
        DeclareLaunchArgument("enable_sensors", default_value="false"),
        DeclareLaunchArgument("enable_yolo", default_value="true"),
        DeclareLaunchArgument("enable_cnn", default_value="false"),
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
    ])
