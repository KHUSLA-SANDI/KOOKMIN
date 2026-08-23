#!/usr/bin/env python3
"""Launch only the real-car hardcoded-overtake pulse tuner."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("track_drive_cnn_gpt")
    track_drive_share = get_package_share_directory("track_drive")
    tuner_config = os.path.join(
        package_share, "config", "overtake_block_tuner.yaml"
    )
    car_config = os.path.join(track_drive_share, "config", "car.yaml")

    return LaunchDescription([
        Node(
            package="track_drive_cnn_gpt",
            executable="overtake_block_tuner",
            name="motion_node",
            output="screen",
            parameters=[tuner_config, car_config],
        ),
    ])
