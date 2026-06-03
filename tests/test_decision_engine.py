"""
Unit tests for Autonomous Robot Decision Engine
"""

import os
import sys
import pytest

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from realfake_perception.decision_engine import DecisionEngine


def test_decision_pick_real():
    engine = DecisionEngine(640, 480, deadzone_px=40)
    detection = {
        'classification': 'REAL',
        'confidence': 0.95,
        'bbox': (240, 160, 400, 320)  # Center (320, 240) - aligned
    }
    decision = engine.evaluate(detection)
    assert decision['action'] == 'PICK_REAL'
    assert decision['target_locked'] is True
    assert decision['gripper_command'] == 'CLOSE'


def test_decision_reject_fake():
    engine = DecisionEngine(640, 480)
    detection = {
        'classification': 'FAKE',
        'confidence': 0.89,
        'bbox': (240, 160, 400, 320)
    }
    decision = engine.evaluate(detection)
    assert decision['action'] == 'REJECT_FAKE'
    assert decision['target_locked'] is False
    assert decision['gripper_command'] == 'BYPASS'


def test_decision_align_center():
    engine = DecisionEngine(640, 480, deadzone_px=30)
    detection = {
        'classification': 'REAL',
        'confidence': 0.90,
        'bbox': (40, 160, 160, 320)  # Left side (Center x = 100 vs Frame center = 320)
    }
    decision = engine.evaluate(detection)
    assert decision['action'] == 'ALIGN_CENTER'
    assert decision['target_locked'] is False
    assert decision['offset_x'] < 0
    assert decision['heading_error_deg'] < 0


def test_decision_searching_target():
    engine = DecisionEngine(640, 480)
    detection = {
        'classification': 'UNKNOWN',
        'confidence': 0.0,
        'bbox': None
    }
    decision = engine.evaluate(detection)
    assert decision['action'] == 'SEARCHING'
    assert decision['target_locked'] is False
    assert decision['gripper_command'] == 'IDLE'
