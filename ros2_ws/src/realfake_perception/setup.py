from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'realfake_perception'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name] if os.path.exists('resource/' + package_name) else []),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Shivam Maurya',
    maintainer_email='shivam@example.com',
    description='Real-Time Perception Pipeline & ROS 2 Vision Node for ABU Robocon',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = realfake_perception.perception_node:main',
            'mock_camera_node = realfake_perception.mock_camera_node:main',
        ],
    },
)
