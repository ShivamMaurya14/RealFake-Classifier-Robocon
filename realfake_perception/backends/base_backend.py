"""
Base Perception Backend Interface
Provides standardized structure for all hot-swappable AI inference backends (ONNX, YOLO, TensorFlow).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np


class BasePerceptionBackend(ABC):
    """Abstract base class for perception backends."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.backend_name = "base"
        self.is_loaded = False
        self.load_model()

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights into memory."""
        pass

    @abstractmethod
    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on an input frame (BGR image from OpenCV / ROS 2).
        
        Returns:
            dict containing:
                - 'classification': 'REAL' | 'FAKE' | 'UNKNOWN'
                - 'confidence': float (0.0 to 1.0)
                - 'raw_score': float
                - 'bbox': Tuple[int, int, int, int] (x1, y1, x2, y2) or None
                - 'inference_time_ms': float
                - 'backend': str
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} backend={self.backend_name} loaded={self.is_loaded}>"
