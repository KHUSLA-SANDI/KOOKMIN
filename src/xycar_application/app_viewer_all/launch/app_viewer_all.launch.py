from launch import LaunchDescription
import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    cam_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('xycar_cam'),
                'launch/xycar_cam_viewer.launch.py'))
    )
    
    imu_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('xycar_imu'),
                'launch/xycar_imu_viewer.launch.py'))
    )
    
    ultra_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('xycar_ultrasonic'),
                'launch/xycar_ultrasonic_viewer.launch.py'))
    )
    
    lidar_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('xycar_lidar'),
                'launch/xycar_lidar_viewer.launch.py'))
    )
    
    return LaunchDescription([
        cam_include,
        imu_include,
        ultra_include,
        lidar_include,
    ])
