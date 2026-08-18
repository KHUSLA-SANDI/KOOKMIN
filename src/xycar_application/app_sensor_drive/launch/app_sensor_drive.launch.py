import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution
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
    
    lidar_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('xycar_lidar'),
                'launch/xycar_lidar.launch.py'))
    )
            
    return LaunchDescription([
        #motor_include,
        lidar_include,
        Node(
            package='app_sensor_drive',
            executable='sensor_driver',
            name='sensor_driver',
            output='screen',  
            parameters=[
                {'front_speed': 10},
                {'back_speed': -12},
                {'distance': 25}
            ]
        ),
    ])
