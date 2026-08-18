from launch import LaunchDescription
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    imu_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('xycar_imu'),
                'launch/xycar_imu.launch.py'))
    )
    
    return LaunchDescription([
        imu_include,
        Node(
            package='my_imu',
            executable='roll_pitch_yaw',
            name='driver',
        ),
    ])
