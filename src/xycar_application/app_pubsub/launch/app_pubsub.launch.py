import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
    
        # Publisher - talker
        Node(
            package='demo_nodes_cpp',
            executable='talker',
            #name='talker'
            output='screen',
        ),
        
        # Subscriber - listener
        Node(
            package='demo_nodes_cpp',
            executable='listener',
            #name='listener',
            output='screen'
        ),
        
        # Rqt_graph
        Node(
            package='rqt_graph',
            executable='rqt_graph',
            #name='rviz',
            output='screen',            
        ),
    ])
        

