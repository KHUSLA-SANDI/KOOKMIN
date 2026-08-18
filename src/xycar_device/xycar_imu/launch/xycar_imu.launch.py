import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    config_path = os.path.join(
        get_package_share_directory("xycar_imu"), "config", "imu.yaml"
    )
        
    imu_node = Node(
        package="xycar_imu", executable="imu_node", output="screen",
        parameters=[config_path]
    )

    return LaunchDescription([imu_node])

