import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('xycar_ultrasonic'),
                'launch/xycar_ultrasonic.launch.py'))
    )
    return LaunchDescription([
        launch_include,
        Node(
            package='app_ultra_viewer',
            executable='app_ultra_viewer',
            name='ultra_viewer'
        ),
    ])
