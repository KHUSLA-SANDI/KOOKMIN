import os
from launch import LaunchDescription
from launch_ros.actions import Node  
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    '''
    motor_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('xycar_motor'),
                'launch/xycar_motor.launch.py'))
    )
    '''
    
    return LaunchDescription([
        #motor_include,
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam_node',
            arguments=['--ros-args', 
                       '-p', 'image_width:=640', 
                       '-p', 'image_height:=480',
                       '-p', 'queue_size:=1'],
        ),
        Node(
            package='web_video_server',
            executable='web_video_server',
            name='web_video_server', 
            arguments=['--ros-args', 
                       '--remap', '_image_transport:=compressed',
                       '--param', 'queue_size:=10'], 
        ),
        
        Node(
            package='rosbridge_server',
            executable='rosbridge_websocket',
            name='ros_bridge',  
        ),
        
    ])
    

