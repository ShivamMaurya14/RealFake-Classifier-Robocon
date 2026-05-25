"""
Robocon Autonomous Robot Decision Engine
Translates real-time perception detections into actionable robotics commands (PICK, REJECT, ALIGN, SEARCH).
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class DecisionEngine:
    """High-level autonomous robot decision-making engine for Robocon."""

    def __init__(self, frame_width: int = 640, frame_height: int = 480, deadzone_px: int = 40):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2
        self.deadzone_px = deadzone_px
        self.target_history = []
        self.history_len = 5

    def evaluate(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes perception detection and generates robot movement and gripper commands.
        
        Args:
            detection: Dict output from perception backend (classification, confidence, bbox, etc.)
            
        Returns:
            Dict containing:
                - 'action': 'PICK_REAL' | 'REJECT_FAKE' | 'ALIGN_CENTER' | 'SEARCHING'
                - 'target_locked': bool
                - 'offset_x': int (horizontal offset from optical center)
                - 'offset_y': int (vertical offset from optical center)
                - 'heading_error_deg': float (estimated yaw angle correction for robot base)
                - 'distance_proxy': float (estimated relative proximity based on box pixel area)
                - 'gripper_command': 'OPEN' | 'CLOSE' | 'BYPASS' | 'IDLE'
                - 'status_message': str
        """
        cls = detection.get('classification', 'UNKNOWN')
        conf = detection.get('confidence', 0.0)
        bbox = detection.get('bbox')

        if cls == 'UNKNOWN' or bbox is None or conf < 0.4:
            return {
                'action': 'SEARCHING',
                'target_locked': False,
                'offset_x': 0,
                'offset_y': 0,
                'heading_error_deg': 0.0,
                'distance_proxy': 0.0,
                'gripper_command': 'IDLE',
                'status_message': 'No target detected. Robot sweeping arena.'
            }

        x1, y1, x2, y2 = bbox
        box_center_x = (x1 + x2) // 2
        box_center_y = (y1 + y2) // 2
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height

        offset_x = box_center_x - self.center_x
        offset_y = box_center_y - self.center_y

        # Approximate yaw angle error: FOV ~ 60 degrees across width
        heading_error_deg = (offset_x / (self.frame_width / 2.0)) * 30.0

        # Approximate distance proxy: closer box = larger pixel area
        distance_proxy = round(float(box_area / (self.frame_width * self.frame_height)), 4)

        # Decision Logic based on ABU Robocon Rulebook
        if cls == 'FAKE':
            action = 'REJECT_FAKE'
            gripper_command = 'BYPASS'
            status_message = f'⚠️ Fake box detected ({conf*100:.1f}%). Bypassing obstacle.'
            target_locked = False
        elif cls == 'REAL':
            if abs(offset_x) > self.deadzone_px:
                action = 'ALIGN_CENTER'
                gripper_command = 'OPEN'
                status_message = f'🎯 Real box detected. Aligning robot heading ({heading_error_deg:+.1f}°).'
                target_locked = False
            else:
                action = 'PICK_REAL'
                gripper_command = 'CLOSE'
                status_message = f'✅ Target locked in gripper zone ({conf*100:.1f}%). Executing PICK.'
                target_locked = True
        else:
            action = 'SEARCHING'
            gripper_command = 'IDLE'
            status_message = 'Searching for competition boxes...'
            target_locked = False

        return {
            'action': action,
            'target_locked': target_locked,
            'classification': cls,
            'confidence': conf,
            'offset_x': int(offset_x),
            'offset_y': int(offset_y),
            'heading_error_deg': round(float(heading_error_deg), 2),
            'distance_proxy': distance_proxy,
            'gripper_command': gripper_command,
            'status_message': status_message
        }
