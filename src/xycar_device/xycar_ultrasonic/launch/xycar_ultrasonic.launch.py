import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription

def generate_launch_description():

    return LaunchDescription([
        Node(
            package='xycar_ultrasonic',
            executable='xycar_ultrasonic',
            name='xycar_ultrasonic'
        ),
    ])
