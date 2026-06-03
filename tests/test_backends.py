"""
Unit tests for Perception Backends (ONNX, TensorFlow, YOLO)
"""

import os
import sys
import numpy as np
import pytest

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from realfake_perception.backends.onnx_backend import ONNXPerceptionBackend
from realfake_perception.backends.tf_cnn_backend import TensorFlowCNNBackend
from realfake_perception.backends.yolo_backend import YOLOPerceptionBackend


def test_onnx_backend_prediction():
    backend = ONNXPerceptionBackend("models/cnn_realfake.onnx")
    assert backend.is_loaded is True
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    res = backend.predict(dummy)
    assert res['classification'] in ['REAL', 'FAKE', 'UNKNOWN']
    assert 0.0 <= res['confidence'] <= 1.0
    assert res['inference_time_ms'] > 0.0
    assert res['backend'] == 'ONNX_RUNTIME_CPU'


def test_tf_backend_prediction():
    backend = TensorFlowCNNBackend("models/cnn_realfake.keras")
    assert backend.is_loaded is True
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    res = backend.predict(dummy)
    assert res['classification'] in ['REAL', 'FAKE', 'UNKNOWN']
    assert 0.0 <= res['confidence'] <= 1.0
    assert res['backend'] == 'TENSORFLOW_CNN'


def test_yolo_backend_graceful_handling():
    backend = YOLOPerceptionBackend("models/nonexistent_yolo.onnx")
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    res = backend.predict(dummy)
    assert 'classification' in res
    assert 'confidence' in res
