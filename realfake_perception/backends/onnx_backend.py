"""
ONNX Runtime Backend
Provides ultra-fast, CPU-only inference (30+ FPS) using ONNX Runtime.
"""

import time
import os
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from .base_backend import BasePerceptionBackend


class ONNXPerceptionBackend(BasePerceptionBackend):
    """Ultra-fast ONNX Runtime backend optimized for CPU edge robotics."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.5, input_size: Tuple[int, int] = (224, 224)):
        self.input_size = input_size
        self.session = None
        self.input_name = None
        self.output_name = None
        super().__init__(model_path, confidence_threshold)
        self.backend_name = "ONNX_RUNTIME_CPU"

    def load_model(self) -> None:
        if ort is None:
            raise ImportError("onnxruntime is not installed. Run 'pip install onnxruntime'.")

        if not os.path.exists(self.model_path):
            print(f"[ONNX Backend] Warning: Model file not found at '{self.model_path}'")
            self.is_loaded = False
            return

        # Configure session options for high-performance CPU inference
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.is_loaded = True
        print(f"[ONNX Backend] Successfully loaded ONNX model from '{self.model_path}' (Input: {self.input_name})")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess BGR OpenCV frame into normalized float32 tensor."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.input_size)
        # Check model input shape: [batch, H, W, C] (Channels Last) or [batch, C, H, W] (Channels First)
        input_shape = self.session.get_inputs()[0].shape
        
        if len(input_shape) == 4 and input_shape[1] == 3:
            # NCHW
            tensor = resized.astype(np.float32) / 255.0
            tensor = np.transpose(tensor, (2, 0, 1))
            tensor = np.expand_dims(tensor, axis=0)
        else:
            # NHWC (Keras standard)
            tensor = resized.astype(np.float32) / 255.0
            tensor = np.expand_dims(tensor, axis=0)
            
        return tensor

    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """Run ONNX model prediction."""
        if not self.is_loaded or self.session is None:
            # Simulation / Fallback output if no model file loaded
            return {
                'classification': 'UNKNOWN',
                'confidence': 0.0,
                'raw_score': 0.5,
                'bbox': None,
                'inference_time_ms': 0.0,
                'backend': self.backend_name
            }

        start_time = time.perf_counter()
        input_tensor = self.preprocess(frame)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        raw_output = outputs[0]
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Handle binary classification score
        if raw_output.shape[-1] == 1:
            raw_score = float(raw_output.flatten()[0])
            # In training: Fake = 0, Real = 1 (or sigmoid probability)
            if raw_score >= self.confidence_threshold:
                label = 'REAL'
                conf = raw_score
            else:
                label = 'FAKE'
                conf = 1.0 - raw_score
        else:
            # Multi-class or softmax
            exp_scores = np.exp(raw_output[0] - np.max(raw_output[0]))
            probs = exp_scores / np.sum(exp_scores)
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            label = 'REAL' if pred_idx == 1 else 'FAKE'
            raw_score = float(probs[1])

        # Generate center bounding box proxy for classification
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
