"""
ROS 2 Simulation Launch File for Robocon Perception Pipeline
Launches Mock Camera Streamer + Perception Node for complete end-to-end testing without hardware.
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
        # Launch Arguments
        DeclareLaunchArgument(
            'backend',
            default_value='onnx',
            description='Inference Backend: onnx (30+ FPS CPU), yolo, or tf'
        ),
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/camera/image_raw',
            description='Simulation camera image topic'
        ),
        DeclareLaunchArgument(
            'fps',
            default_value='30.0',
            description='Simulation camera framerate'
        ),
        DeclareLaunchArgument(
            'sim_mode',
            default_value='synthetic',
            description='Simulation source mode: synthetic or dataset'
        ),

        # 1. Mock Camera Publisher Node
        Node(
            package='realfake_perception',
            executable='mock_camera_node',
            name='mock_camera_node',
            output='screen',
            parameters=[
                {
                    'camera_topic': LaunchConfiguration('camera_topic'),
                    'fps': LaunchConfiguration('fps'),
                    'mode': LaunchConfiguration('sim_mode'),
                    'frame_id': 'camera_link'
                }
            ]
        ),

        # 2. Perception & Decision Node
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
                }
            ]
        )
    ])
