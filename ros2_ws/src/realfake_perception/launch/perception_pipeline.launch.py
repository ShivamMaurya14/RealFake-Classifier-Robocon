"""
ROS 2 Launch File for Robocon Perception Pipeline
Launches the RealFake Perception Node with configurable backend and parameter overrides.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('realfake_perception')
    default_config = os.path.join(pkg_share, 'config', 'perception_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'backend',
            default_value='onnx',
            description='Inference Backend: onnx (30+ FPS CPU), yolo, or tf'
        ),
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/camera/image_raw',
            description='Input sensor_msgs/Image camera topic'
        ),
        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.5',
            description='Confidence threshold for classification'
        ),
        Node(
            package='realfake_perception',
            executable='perception_node',
            name='realfake_perception_node',
            output='screen',
            parameters=[
                default_config,
                {
                    'backend': LaunchConfiguration('backend'),
                    'camera_topic': LaunchConfiguration('camera_topic'),
                    'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                }
            ]
        )
    ])
