"""
TensorFlow / Keras CNN Backend
Provides native TensorFlow inference backend for training verification and baseline comparison.
"""

import time
import os
from typing import Dict, Any, Tuple
import cv2
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

from .base_backend import BasePerceptionBackend


class TensorFlowCNNBackend(BasePerceptionBackend):
    """TensorFlow / Keras CNN Backend."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.5, input_size: Tuple[int, int] = (224, 224)):
        self.input_size = input_size
        self.model = None
        super().__init__(model_path, confidence_threshold)
        self.backend_name = "TENSORFLOW_CNN"

    def load_model(self) -> None:
        if tf is None:
            raise ImportError("tensorflow is not installed. Run 'pip install tensorflow'.")

        if not os.path.exists(self.model_path):
            print(f"[TensorFlow Backend] Warning: Model file not found at '{self.model_path}'")
            self.is_loaded = False
            return

        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.is_loaded = True
            print(f"[TensorFlow Backend] Successfully loaded Keras model from '{self.model_path}'")
        except Exception as e:
            print(f"[TensorFlow Backend] Error loading model: {e}")
            self.is_loaded = False

    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """Run TensorFlow CNN prediction."""
        if not self.is_loaded or self.model is None:
            return {
                'classification': 'UNKNOWN',
                'confidence': 0.0,
                'raw_score': 0.5,
                'bbox': None,
                'inference_time_ms': 0.0,
                'backend': self.backend_name
            }

        start_time = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.input_size)
        input_tensor = np.expand_dims(resized, axis=0) # Rescaling is built-in inside model layer or normalized

        raw_output = self.model.predict(input_tensor, verbose=0)
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        raw_score = float(raw_output[0][0])
        if raw_score >= self.confidence_threshold:
            label = 'REAL'
            conf = raw_score
        else:
            label = 'FAKE'
            conf = 1.0 - raw_score

        h, w = frame.shape[:2]
        pad_x, pad_y = int(w * 0.15), int(h * 0.15)
        bbox = (pad_x, pad_y, w - pad_x, h - pad_y)

        return {
            'classification': label,
            'confidence': round(float(conf), 4),
            'raw_score': round(float(raw_score), 4),
            'bbox': bbox,
            'inference_time_ms': round(inference_time_ms, 2),
            'backend': self.backend_name
        }
