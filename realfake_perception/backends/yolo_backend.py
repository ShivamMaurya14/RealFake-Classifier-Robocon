"""
YOLO Object Detection Backend
Provides real-time YOLOv8 / YOLOv11 object detection & localization backend.
"""

import time
import os
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from .base_backend import BasePerceptionBackend


class YOLOPerceptionBackend(BasePerceptionBackend):
    """YOLO Object Detection Backend (PyTorch / ONNX)."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.4):
        self.model = None
        super().__init__(model_path, confidence_threshold)
        self.backend_name = "YOLO_DETECTION"

    def load_model(self) -> None:
        if YOLO is None:
            print("[YOLO Backend] Ultralytics not installed. Marking backend unavailable.")
            self.is_loaded = False
            return

        if not os.path.exists(self.model_path):
            print(f"[YOLO Backend] Notice: Custom model '{self.model_path}' not found on disk.")
            self.is_loaded = False
            return

        try:
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            print(f"[YOLO Backend] Successfully loaded YOLO model from '{self.model_path}'")
        except Exception as e:
            print(f"[YOLO Backend] Could not load YOLO model: {e}")
            self.is_loaded = False

    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """Run YOLO Object Detection prediction."""
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
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
            device='cpu'
        )
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        best_bbox = None
        best_conf = 0.0
        best_label = 'UNKNOWN'

        if len(results) > 0 and len(results[0].boxes) > 0:
            # Pick highest confidence detection
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}").lower()

                if conf > best_conf:
                    best_conf = conf
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    best_bbox = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
                    
                    if 'real' in cls_name:
                        best_label = 'REAL'
                    elif 'fake' in cls_name:
                        best_label = 'FAKE'
                    else:
                        # Fallback for generic objects or binary mapped classes
                        best_label = 'REAL' if cls_id % 2 == 0 else 'FAKE'

        # If no detection was made, default to centered frame bounding box
        if best_bbox is None:
            h, w = frame.shape[:2]
            pad_x, pad_y = int(w * 0.15), int(h * 0.15)
            best_bbox = (pad_x, pad_y, w - pad_x, h - pad_y)
            best_label = 'UNKNOWN'
            best_conf = 0.0

        return {
            'classification': best_label,
            'confidence': round(float(best_conf), 4),
            'raw_score': round(float(best_conf), 4),
            'bbox': best_bbox,
            'inference_time_ms': round(inference_time_ms, 2),
            'backend': self.backend_name
        }
