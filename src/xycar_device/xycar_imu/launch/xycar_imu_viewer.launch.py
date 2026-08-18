import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

tf_node = Node(
    package="tf2_ros",
    executable="static_transform_publisher",
    arguments=[
        "0", "0", "0",
        "0", "0", "0",
        "map",
        "base_imu_link"
    ]
)

def generate_launch_description():

    config_path = os.path.join(
        get_package_share_directory("xycar_imu"), "config", "imu.yaml"
    )
        
    rviz_path = os.path.join(os.getenv("HOME"), "xycar_ws", "src", "xycar_device", "xycar_imu", "rviz", "view_imu.rviz")

    imu_node = Node(
        package="xycar_imu", executable="imu_node", output="screen",
        parameters=[config_path]
    )

    rviz_node = Node(
        package="rviz2", executable="rviz2", name="rviz_imu", output="screen",
        arguments=["-d", str(rviz_path)]
    )

    return LaunchDescription([imu_node, tf_node, rviz_node])

