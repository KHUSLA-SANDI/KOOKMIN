from launch import LaunchDescription
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    '''
    motor_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('xycar_motor'),
                'launch/xycar_motor.launch.py'))
    )
    '''
    return LaunchDescription([
        #motor_include,
        Node(
            package='app_8_drive',
            executable='app_8_drive',
            name='motor_drive',
            parameters=[{'speed': 12}],
        ),
    ])
