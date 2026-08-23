from glob import glob
import os

from setuptools import find_packages, setup


package_name = "track_drive_cnn_gpt"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["requirements_models_gpt.txt"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"),
         glob("config/*.yaml") + glob("config/*.xml")),
        (os.path.join("share", package_name, "tools"), glob("tools/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="xytron",
    maintainer_email="xytron@example.com",
    description="Isolated YOLO-BEV-LiDAR-CNN path pipeline for the 2026 main race",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo_bev = track_drive_cnn_gpt.yolo_bev_node:main",
            "cnn_path = track_drive_cnn_gpt.cnn_path_node:main",
            "cnn_drive_gate = track_drive_cnn_gpt.drive_gate_node:main",
            "cnn_motion = track_drive_cnn_gpt.motion_cnn_node:main",
            "simple_motion = track_drive_cnn_gpt.simple_motion_node:main",
            "overtake_block_tuner = track_drive_cnn_gpt.overtake_block_tuner_node:main",
            "live_pipeline_viewer = track_drive_cnn_gpt.live_pipeline_viewer_node:main",
            "replay_images = track_drive_cnn_gpt.replay_source_node:main_images",
            "replay_clean_scan = track_drive_cnn_gpt.replay_source_node:main_clean_scan",
            "replay_recorder = track_drive_cnn_gpt.replay_recorder_node:main",
            "replay_preview = track_drive_cnn_gpt.replay_preview_node:main",
        ],
    },
)
