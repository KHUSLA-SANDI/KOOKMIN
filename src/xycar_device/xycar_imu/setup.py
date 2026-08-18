from glob import glob
import os
from setuptools import setup, find_packages

PACKAGE_NAME = 'xycar_imu'
SHARE_DIR = os.path.join('share', PACKAGE_NAME)
setup(
    name=PACKAGE_NAME,
    version='1.2.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + PACKAGE_NAME],
        ),
        (
            SHARE_DIR,
            ['package.xml'],
        ),
        (
            os.path.join(SHARE_DIR, 'launch'),
            glob(os.path.join('launch', '*.launch.py')),
        ),
        (
            os.path.join(SHARE_DIR, 'config'),
            glob(os.path.join('config', '*.yaml')),
        ),
    ],
    py_modules=[],
    zip_safe=True,
    install_requires=[
        'setuptools',
        'pyserial',
        'pyyaml',
        'transforms3d',
        'vpython',
        'wxPython'
    ],
    author='Kristof Robot, Tang Tiong Yew, Paul Bouchier, Peter Bartz',
    maintainer='Kristof Robot',
    keywords=['ROS2'],
    description='xycar_imu is a package that provides a ROS2 driver for the Sparkfun Razor IMU 9DOF.',
    license='BSD',
    entry_points={
        'console_scripts': [
            'imu_node = xycar_imu.imu_node:main',
            'display_3D_visualization_node = xycar_imu.display_3D_visualization:main',
        ],
    },
)
