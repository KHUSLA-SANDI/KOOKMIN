#!/usr/bin/env python3
"""본선 주행 일괄 실행 — yolo + path_planner + motion + car_state 4노드.

역할: 규정 실행 명령 `ros2 launch track_drive drive.launch.py` 하나로 주행 파이프라인 기동.
입력: config/tuning.yaml(노드별 튜닝) + config/car.yaml(차량 실측 — 뒤에 전달돼 우선).
출력: 없음 (노드 기동만). 센서는 sensors.launch.py, 뷰어는 viewer.launch.py 로 분리.
예선 대비 변경점: 6노드+뷰어 → 4노드 (yolo 3노드+integration 이 yolo 1노드로 통합,
  car_state 신규). 파라미터 yaml 2종 전달 추가. viewer 는 규정상 미포함.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = "track_drive"
    share = get_package_share_directory(pkg)
    # 같은 키가 양쪽에 있으면 뒤(car.yaml=실측값)가 우선
    params = [
        os.path.join(share, "config", "tuning.yaml"),
        os.path.join(share, "config", "car.yaml"),
    ]

    def node(exe):
        return Node(package=pkg, executable=exe,
                    output="screen", parameters=params)

    return LaunchDescription([
        node("yolo"),          # 3모델 순차 추론 → /detect/lane·objects·traffic_light
        node("path_planner"),  # 상태머신 → /drive_cmd, /center_path
        node("motion"),        # 추종 + 중재 + 실차 변환 → /xycar_motor (20Hz 연속)
        node("car_state"),     # IMU + VESC 집계 → /car_state
    ])
