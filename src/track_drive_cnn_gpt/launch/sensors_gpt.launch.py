#!/usr/bin/env python3
"""Explicit, opt-in sensor bringup for the CNN pipeline."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = get_package_share_directory("track_drive_cnn_gpt")
    dds_profile = os.path.join(package_share, "config", "dds_shm_lan_gpt.xml")

    enable_camera = LaunchConfiguration("enable_camera")
    enable_lidar = LaunchConfiguration("enable_lidar")
    enable_imu = LaunchConfiguration("enable_imu")
    ros_domain_id = LaunchConfiguration("ros_domain_id")

    # This camera exposes the current UVC control as
    # ``exposure_dynamic_framerate``.  usb_cam 0.8.x tries the legacy
    # ``exposure_auto`` name instead, so it cannot disable the FPS drop by
    # itself.  Apply the control before opening /dev/video0, then start the
    # camera only after v4l2-ctl has exited.
    camera_fps_control = ExecuteProcess(
        cmd=[
            "v4l2-ctl",
            "--device=/dev/video0",
            "--set-ctrl=exposure_dynamic_framerate=0",
        ],
        output="screen",
        condition=IfCondition(enable_camera),
    )
    camera_node = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        name="usb_cam",
        output="screen",
        # A duplicate launch must fail once instead of retrying forever
        # while another process owns /dev/video0.
        respawn=False,
        parameters=[{
            "video_device": "/dev/video0",
            "pixel_format": "mjpeg2rgb",
            "image_width": 1920,
            "image_height": 1080,
            "framerate": 30.0,
            # Production consumes raw /image_raw only.  Do not load JPEG,
            # Theora or compressedDepth publishers even if an accidental
            # all-topics bridge asks for every advertised transport.
            "image_raw.enable_pub_plugins": ["image_transport/raw"],
        }],
    )

    lidar_launch = PathJoinSubstitution([
        FindPackageShare("xycar_lidar"),
        "launch",
        "xycar_lidar.launch.py",
    ])
    imu_launch = PathJoinSubstitution([
        FindPackageShare("xycar_imu"),
        "launch",
        "xycar_imu.launch.py",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("enable_camera", default_value="false"),
        DeclareLaunchArgument("enable_lidar", default_value="false"),
        DeclareLaunchArgument("enable_imu", default_value="false"),
        DeclareLaunchArgument("ros_domain_id", default_value="7"),
        SetEnvironmentVariable(
            "FASTRTPS_DEFAULT_PROFILES_FILE", dds_profile
        ),
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
        camera_fps_control,
        RegisterEventHandler(
            OnProcessExit(
                target_action=camera_fps_control,
                on_exit=[camera_node],
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(lidar_launch),
            condition=IfCondition(enable_lidar),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(imu_launch),
            condition=IfCondition(enable_imu),
        ),
    ])
