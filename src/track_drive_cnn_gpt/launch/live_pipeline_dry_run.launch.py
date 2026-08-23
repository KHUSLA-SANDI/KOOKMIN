#!/usr/bin/env python3
"""Run the real live pipeline while hard-remapping every motor-side topic."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = get_package_share_directory("track_drive_cnn_gpt")
    perception_config = os.path.join(package_share, "config", "perception_cnn.yaml")
    motion_config = os.path.join(package_share, "config", "simple_motion.yaml")
    sensors_launch = os.path.join(package_share, "launch", "sensors_gpt.launch.py")
    dds_profile = os.path.join(package_share, "config", "dds_shm_lan_gpt.xml")
    legacy_car_config = PathJoinSubstitution(
        [FindPackageShare("track_drive"), "config", "car.yaml"]
    )

    enable_sensors = LaunchConfiguration("enable_sensors")
    enable_lidar = LaunchConfiguration("enable_lidar")
    enable_motion = LaunchConfiguration("enable_motion")
    enable_viewer = LaunchConfiguration("enable_viewer")
    show_viewer = LaunchConfiguration("show_viewer")
    viewer_publish_compressed = LaunchConfiguration("viewer_publish_compressed")
    ros_domain_id = LaunchConfiguration("ros_domain_id")
    speed_cap = LaunchConfiguration("speed_cap")

    return LaunchDescription(
        [
            DeclareLaunchArgument("enable_sensors", default_value="true"),
            DeclareLaunchArgument("enable_lidar", default_value=enable_sensors),
            DeclareLaunchArgument("enable_motion", default_value="true"),
            DeclareLaunchArgument("enable_viewer", default_value="true"),
            DeclareLaunchArgument("show_viewer", default_value="true"),
            DeclareLaunchArgument(
                "viewer_publish_compressed", default_value="false"
            ),
            DeclareLaunchArgument("ros_domain_id", default_value="7"),
            DeclareLaunchArgument("speed_cap", default_value="5.0"),
            SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", dds_profile),
            SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
            SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(sensors_launch),
                launch_arguments={
                    "enable_camera": enable_sensors,
                    "enable_lidar": enable_lidar,
                    "enable_imu": "false",
                    "ros_domain_id": ros_domain_id,
                }.items(),
            ),
            Node(
                package="track_drive_cnn_gpt",
                executable="yolo_bev",
                output="screen",
                parameters=[perception_config],
            ),
            Node(
                package="track_drive_cnn_gpt",
                executable="cnn_path",
                output="screen",
                parameters=[
                    perception_config,
                    {
                        # When LiDAR is disabled, run the exact clean training
                        # contract: channel 2 stays all-zero instead of failing
                        # closed on a missing scan.
                        "require_fresh_scan": ParameterValue(
                            enable_lidar, value_type=bool
                        ),
                    },
                ],
            ),
            Node(
                package="track_drive_cnn_gpt",
                executable="cnn_drive_gate",
                output="screen",
                condition=IfCondition(enable_motion),
                parameters=[
                    perception_config,
                    {
                        "enable_drive": True,
                        # Exercise the same WAIT_GREEN -> RUNNING lifecycle as
                        # the real drive launches.  /manual_go remains an
                        # explicit operator override when needed.
                        "initial_manual_go": False,
                        "speed_cap": ParameterValue(speed_cap, value_type=float),
                    },
                ],
                # Isolate the dry-run gate from any real motion node that may
                # accidentally exist in the same ROS domain.
                remappings=[("/drive_cmd", "/debug/drive_cmd_dryrun")],
            ),
            Node(
                package="track_drive_cnn_gpt",
                executable="simple_motion",
                output="screen",
                condition=IfCondition(enable_motion),
                parameters=[motion_config, legacy_car_config],
                # The minimal controller still uses the audited vehicle
                # CarInterface and car.yaml, but its output cannot reach the
                # ROS1 motor bridge in this dry run.
                remappings=[
                    ("/drive_cmd", "/debug/drive_cmd_dryrun"),
                    ("/xycar_motor", "/debug/xycar_motor_dryrun"),
                ],
            ),
            Node(
                package="track_drive_cnn_gpt",
                executable="live_pipeline_viewer",
                output="screen",
                condition=IfCondition(enable_viewer),
                parameters=[
                    perception_config,
                    {
                        "show_window": ParameterValue(show_viewer, value_type=bool),
                        "publish_compressed": ParameterValue(
                            viewer_publish_compressed, value_type=bool
                        ),
                    },
                ],
            ),
        ]
    )
