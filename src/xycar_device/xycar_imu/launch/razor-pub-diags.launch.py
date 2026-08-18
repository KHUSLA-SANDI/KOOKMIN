import sys
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchIntrospector, LaunchService
from launch_ros.actions import Node  # ✅ launch_ros.actions.Node 사용


def generate_launch_description():
    """
    Launch file for visualizing and diagnosing Razor IMU data
    """
    config_path = os.path.join(get_package_share_directory("xycar_imu"), "config",
                               "razor_diags.yaml")

    display_3D_visualization_node = Node(
        package='xycar_imu', executable='display_3D_visualization_node', output='screen')

    diagnostic_aggregator = Node(
        package='diagnostic_aggregator', executable='aggregator_node', output='screen',
        parameters=[config_path])  # ✅ parameter → parameters 수정

    rqt_robot_monitor = Node(
        package='rqt_robot_monitor', executable='rqt_robot_monitor', output='screen')

    return LaunchDescription(
        [display_3D_visualization_node, diagnostic_aggregator, rqt_robot_monitor])


def main(argv):
    ld = generate_launch_description()

    print('Starting introspection of launch description...')
    print(LaunchIntrospector().format_launch_description(ld))

    print('Starting launch of launch description...')
    ls = LaunchService()
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == '__main__':
    main(sys.argv)

