from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'app_path_planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # .so 파일 포함
        (os.path.join('lib', package_name), glob('app_path_planning/*.so')),  
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
            'main = app_path_planning.main:run'   # run() 함수를 실행하도록 변경
        ],
    },    
)
