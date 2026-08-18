from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution

def generate_launch_description():

    map_name_arg = DeclareLaunchArgument(
        "map_name", default_value=TextSubstitution(text="snake"))  # square, snake
        
    range_sensor_arg = DeclareLaunchArgument(
        "range_sensor", default_value=TextSubstitution(text="ultrasonic")) # ultrasonic, lidar
        
    drive_mode_arg = DeclareLaunchArgument( 
        "drive_mode", default_value=TextSubstitution(text="ros")) # ros, keyboard
        
    simul_node = Node(
            package='xycar_sim_driving',
            executable='simulator',
            name='simulator',
            parameters=[{
                'map_name': LaunchConfiguration('map_name'),
                'range_sensor': LaunchConfiguration('range_sensor'),
                'drive_mode': LaunchConfiguration('drive_mode'),}]
            )
    
    return LaunchDescription([
        map_name_arg,
        range_sensor_arg,
        drive_mode_arg,
        simul_node,
        Node(
            package='app_rule_drive_sim',
            executable='drive',
            name='drive'
        ),
    ])
