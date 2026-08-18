# Copyright (c) 2019, Andreas Klintberg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

from launch import LaunchDescription, LaunchIntrospector, LaunchService
from launch_ros.actions import Node  # 올바른 임포트

def generate_launch_description():
    """
    Launch file for visualizing Razor IMU data
    """
    display_3D_visualization_node = Node(  # actions.Node 대신 Node 사용
        package='xycar_imu', executable='display_3D_visualization_node', output='screen'  # node_executable → executable
    )

    return LaunchDescription([display_3D_visualization_node])

def main(args=None):
    ld = generate_launch_description()

    print('Starting introspection of launch description...\n')
    print(LaunchIntrospector().format_launch_description(ld))

    print('\nStarting launch of launch description...\n')

    ls = LaunchService()
    ls.include_launch_description(ld)  # get_default_launch_description() 제거
    return ls.run()

if __name__ == '__main__':
    main(sys.argv)

