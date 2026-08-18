import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    usb_cam_dir = get_package_share_directory('usb_cam')

    # get path to params file
    params_path = os.path.join(
        usb_cam_dir,
        'config',
        'params.yaml'
    )

    print(params_path)
    
    ld.add_action(Node(
        package='usb_cam', 
        executable='usb_cam_node_exe', 
        name='xycar_cam',
        arguments=['--ros-args', '--log-level', 'error'],
        parameters=[params_path]
        ))
        
    ld.add_action(Node(
        package='usb_cam', executable='show_image.py', output='screen',
        ))

    return ld
