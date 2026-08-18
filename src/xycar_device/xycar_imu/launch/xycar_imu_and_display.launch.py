import sys
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchService, LaunchIntrospector
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for publishing and visualizing Razor IMU data.
    """
    config_path = os.path.join(
        get_package_share_directory("xycar_imu"),
        "config",
        "xycar_imu.yaml"
    )

    print(f"Config path: {config_path}")

    imu_node = Node(
        package="xycar_imu",
        executable="imu_node",
        output="screen",
        parameters=[config_path],
    )

    display_3D_visualization_node = Node(
        package="xycar_imu",
        executable="display_3D_visualization_node",
        output="screen",
    )

    return LaunchDescription([imu_node, display_3D_visualization_node])


def main(args=None):
    ld = generate_launch_description()

    print("Starting introspection of launch description...\n")
    print(LaunchIntrospector().format_launch_description(ld))
    print("\nStarting launch of launch description...\n")

    ls = LaunchService()
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == "__main__":
    main(sys.argv)

