import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([
        Node(
            package='app_cam_exposure',
            executable='app_cam_exposure',
            name='app_cam_exposure',
        ),
    ])
