#!/usr/bin/env python3
"""BEV 디버그 뷰어 단독 실행 — 주행 launch 와 분리 (규정: drive.launch.py 에 뷰어 금지).

역할: yolo_lane_viewer 1개 기동. 실행: ros2 launch track_drive viewer.launch.py
입력: 없음 (뷰어가 /detect/lane, /center_path, /scan 등 구독).
출력: OpenCV 창 (X 포워딩/모니터 필요).
예선 대비 변경점: drive.launch.py 에 섞여 있던 뷰어를 별도 launch 로 분리.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="track_drive", executable="yolo_lane_viewer",
             output="screen"),
    ])
