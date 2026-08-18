from setuptools import setup
import os
from glob import glob

package_name = 'track_drive'

setup(
    name=package_name,
    version='0.0.0',
    # lib/ = rclpy 없는 순수 파이썬 모듈 (유닛테스트/rosbag 재생용) — 명시 포함
    packages=[package_name, package_name + '.lib'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # drive/viewer/sensors launch 전부 (*.launch.py 아님 주의 없이 *.py 로 수집)
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # 파라미터 yaml + YOLO 가중치 (camera.yaml 포함)
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 본선 주행 파이프라인 (실차)
            'yolo = track_drive.yolo_node:main',
            'path_planner = track_drive.path_planner_node:main',
            'motion = track_drive.motion_node:main',
            'car_state = track_drive.car_state_node:main',
            'teleop = track_drive.teleop_node:main',
            'recorder = track_drive.recorder_node:main',
            # 디버그 뷰어
            'live_viewer = track_drive.live_viewer_node:main',   # 원본+BEV 2패널
            'yolo_lane_viewer = track_drive.yolo_lane_viewer_node:main',
            # 예선 유래 파일은 lane_planner.py(폴리핏 코어, planner가 import)와
            # yolo_lane_viewer_node.py(뷰어)만 유지 — 나머지는 예선/ 폴더 참조
        ],
    },
)
