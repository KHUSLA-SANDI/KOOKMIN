#!/usr/bin/env python3
"""Run camera + configured YOLO signal preview; never start CNN or drive nodes."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("track_drive_cnn_gpt")
    config_path = os.path.join(package_share, "config", "perception_cnn.yaml")
    sensors_launch = os.path.join(package_share, "launch", "sensors_gpt.launch.py")
    dds_profile = os.path.join(package_share, "config", "dds_shm_lan_gpt.xml")

    enable_camera = LaunchConfiguration("enable_camera")
    enable_viewer = LaunchConfiguration("enable_viewer")
    ros_domain_id = LaunchConfiguration("ros_domain_id")

    viewer_node = Node(
        package="track_drive_cnn_gpt",
        executable="live_pipeline_viewer",
        output="screen",
        condition=IfCondition(enable_viewer),
        parameters=[
            config_path,
            {
                "window_name": "Xycar YOLO signal test",
                "screenshot_dir": "/home/xytron/signal_label_candidates_gpt",
                # yolo_bev already publishes the annotated JPEG stream below;
                # avoid a second dashboard JPEG encode in this light-weight test.
                "publish_compressed": False,
                # Keep this signal-only launch isolated from any motion topic
                # that may exist in the same ROS domain.  The viewer is
                # read-only, but showing stale steering data here is confusing.
                "motion_topic": "/debug/xycar_motor_signal_test_unused",
            },
        ],
    )

    return LaunchDescription([
        # Keep false by default so an already running usb_cam is never opened twice.
        DeclareLaunchArgument("enable_camera", default_value="false"),
        DeclareLaunchArgument("enable_viewer", default_value="true"),
        DeclareLaunchArgument("ros_domain_id", default_value="7"),
        SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", dds_profile),
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(sensors_launch),
            launch_arguments={
                "enable_camera": enable_camera,
                "enable_lidar": "false",
                "enable_imu": "false",
                "ros_domain_id": ros_domain_id,
            }.items(),
        ),
        Node(
            package="track_drive_cnn_gpt",
            executable="yolo_bev",
            output="screen",
            parameters=[config_path, {"publish_signal_preview": True}],
        ),
        viewer_node,
        # If the viewer process exits, stop camera and YOLO too.
        RegisterEventHandler(
            OnProcessExit(
                target_action=viewer_node,
                on_exit=[EmitEvent(event=Shutdown(reason="signal viewer closed"))],
            )
        ),
    ])
