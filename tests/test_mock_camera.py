"""
Unit test for Mock Camera Publisher Node
"""

import os
import sys
import numpy as np
import pytest

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from realfake_perception.mock_camera_node import MockCameraNode


def test_mock_camera_synthetic_generation():
    node = MockCameraNode()
    frame = node.generate_synthetic_frame()
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8


def test_mock_camera_get_frame():
    node = MockCameraNode()
    frame = node.get_current_frame()
    assert frame is not None
    assert frame.shape == (480, 640, 3)
