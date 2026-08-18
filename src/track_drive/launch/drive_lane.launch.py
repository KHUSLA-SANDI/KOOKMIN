#!/usr/bin/env python3
"""노란선만 따라가기 — 연습/검증용 주행 실행.

역할: 본선용 drive.launch.py 와 같은 4노드를 띄우되, config/lane_only.yaml 을
  마지막에 얹어 "노랑 mid 하나만 보고 그 선을 따라간다" 로 좁힌다.
  콘·정지선·지름길·신호등·보행자 전부 꺼진다 (연습장 오작동 방지).

입력: tuning.yaml → car.yaml → lane_only.yaml 순서로 전달 (뒤가 우선).
출력: 없음 (노드 기동만). 센서는 sensors.launch.py 를 따로 띄운다.

★ 출발은 자동이 아니다. 신호등을 안 쓰므로 아래를 쳐야 움직인다:
    ros2 topic pub --once /manual_go std_msgs/msg/Bool "{data: true}"
  세우려면 그냥 이 런치를 Ctrl+C (또는 teleop 을 띄워 수동 개입).

사용: ros2 launch track_drive drive_lane.launch.py
      ros2 launch track_drive drive_lane.launch.py speed:=6.0    # 더 느리게
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = "track_drive"
    share = get_package_share_directory(pkg)
    cfg = os.path.join(share, "config")

    # 같은 키가 여러 파일에 있으면 뒤에 오는 것이 이긴다
    params = [
        os.path.join(cfg, "tuning.yaml"),     # 본선 기본값
        os.path.join(cfg, "car.yaml"),        # 차량 실측값
        os.path.join(cfg, "lane_only.yaml"),  # ★ 노란선 전용 오버라이드
    ]

    speed = LaunchConfiguration("speed")

    return LaunchDescription([
        DeclareLaunchArgument(
            "speed", default_value="6.0",
            description="기본 주행 속도 명령 (×0.08 = m/s). 6.0 ≈ 0.48m/s = teleop w 3번"),

        Node(package=pkg, executable="yolo", output="screen", parameters=params),
        Node(package=pkg, executable="path_planner", output="screen",
             parameters=params),
        # 속도만 런치 인자로 덮어쓸 수 있게 (첫 주행은 낮추기 쉬워야 한다).
        # ★ ParameterValue 로 float 을 명시해야 한다 — 안 그러면 launch 가
        #   "8.0" 문자열로 넘겨서 motion_node 가 타입 불일치로 죽는다.
        Node(package=pkg, executable="motion", output="screen",
             parameters=params + [{
                 "speed_default": ParameterValue(speed, value_type=float)}]),
        Node(package=pkg, executable="car_state", output="screen",
             parameters=params),
    ])
