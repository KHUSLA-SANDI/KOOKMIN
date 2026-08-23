#!/usr/bin/env python3
"""One-command race drive launch without the diagnostic viewer."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    package_share = get_package_share_directory("track_drive_cnn_gpt")
    base_launch = os.path.join(
        package_share, "launch", "drive_low_speed.launch.py"
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base_launch),
            launch_arguments={
                "enable_sensors": "true",
                "enable_lidar": "true",
                "enable_motor": "true",
                "enable_drive_gate": "true",
                "enable_drive": "true",
                "enable_viewer": "false",
                "speed_cap": "50.0",
                "ros_domain_id": "7",
            }.items(),
        ),
    ])
