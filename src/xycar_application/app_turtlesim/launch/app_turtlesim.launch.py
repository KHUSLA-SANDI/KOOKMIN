import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
    
        # turtlesim_node - Draw turtle
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            output='screen'
        ),
        
        # turtle_teleop_key - Get user input
        Node(
            package='turtlesim',
            executable='turtle_teleop_key',
            output='screen'
        ),
    ])
        

