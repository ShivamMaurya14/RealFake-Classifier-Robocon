"""
ROS 2 Perception Node for RealFake Robocon Vision Pipeline
Subscribes: /camera/image_raw (sensor_msgs/Image)
Publishes:  /perception/detections (std_msgs/String JSON)
            /perception/decision (std_msgs/String JSON)
            /perception/image_annotated (sensor_msgs/Image)
"""

import os
import sys
import json
import time
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np

# Try importing rclpy; handle mock/standalone gracefully if ROS2 is not sourced
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String, Float32
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    # Mock base class for non-ROS testing
    class Node:
        def __init__(self, name):
            self.name = name
            print(f"[ROS2 Node Mock] Running {name} in standalone mock mode (rclpy not in PYTHONPATH)")

        def declare_parameter(self, name, val): pass
        def get_parameter(self, name):
            class Param:
                value = None
            p = Param()
            p.value = None
            return p
        def create_subscription(self, *args, **kwargs): return None
        def create_publisher(self, *args, **kwargs): return None
        def create_timer(self, *args, **kwargs): return None
        def get_logger(self):
            class Logger:
                def info(self, m): print(f"[INFO] {m}")
                def warn(self, m): print(f"[WARN] {m}")
                def error(self, m): print(f"[ERROR] {m}")
            return Logger()

try:
    from cv_bridge import CvBridge
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    CV_BRIDGE_AVAILABLE = False

from realfake_perception.backends.onnx_backend import ONNXPerceptionBackend
from realfake_perception.backends.tf_cnn_backend import TensorFlowCNNBackend
from realfake_perception.backends.yolo_backend import YOLOPerceptionBackend
from realfake_perception.decision_engine import DecisionEngine


class RealFakePerceptionNode(Node):
    """ROS 2 Perception Node with hot-swappable AI inference backends."""

    def __init__(self):
        super().__init__('realfake_perception_node')
        logger = self.get_logger()
        logger.info("Initializing RealFake Perception Node (ROS 2 Vision Pipeline)...")

        # Declare ROS 2 Parameters
        self.declare_parameter('backend', 'onnx')
        self.declare_parameter('onnx_model_path', 'models/cnn_realfake.onnx')
        self.declare_parameter('tf_model_path', 'models/cnn_realfake.keras')
        self.declare_parameter('yolo_model_path', 'models/yolo_realfake.onnx')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('publish_annotated', True)

        # Retrieve parameter values
        self.backend_name = self.get_parameter('backend').value or 'onnx'
        self.onnx_path = self.get_parameter('onnx_model_path').value or 'models/cnn_realfake.onnx'
        self.tf_path = self.get_parameter('tf_model_path').value or 'models/cnn_realfake.keras'
        self.yolo_path = self.get_parameter('yolo_model_path').value or 'models/yolo_realfake.onnx'
        self.conf_thresh = float(self.get_parameter('confidence_threshold').value or 0.5)
        self.camera_topic = self.get_parameter('camera_topic').value or '/camera/image_raw'

        # Initialize Decision Engine
        self.decision_engine = DecisionEngine()
        self.bridge = CvBridge() if CV_BRIDGE_AVAILABLE else None

        # Load Backends
        self.backends = {}
        self._init_backends()

        # ROS 2 Publishers & Subscribers
        if ROS2_AVAILABLE:
            self.sub_image = self.create_subscription(
                Image,
                self.camera_topic,
                self.image_callback,
                10
            )
            self.pub_detections = self.create_publisher(String, '/perception/detections', 10)
            self.pub_decision = self.create_publisher(String, '/perception/decision', 10)
            self.pub_annotated = self.create_publisher(Image, '/perception/image_annotated', 10)
            logger.info(f"Subscribed to topic: {self.camera_topic}")
            logger.info("Publishing to: /perception/detections, /perception/decision, /perception/image_annotated")
        else:
            logger.warn("ROS 2 rclpy environment not found. Node ready in modular standalone mode.")

    def _init_backends(self):
        """Initializes all available inference backends."""
        logger = self.get_logger()
        try:
            self.backends['onnx'] = ONNXPerceptionBackend(self.onnx_path, self.conf_thresh)
            logger.info(f"Loaded ONNX Backend (CPU Optimized 30+ FPS)")
        except Exception as e:
            logger.warn(f"Could not load ONNX backend: {e}")

        try:
            self.backends['tf'] = TensorFlowCNNBackend(self.tf_path, self.conf_thresh)
            logger.info(f"Loaded TensorFlow Backend")
        except Exception as e:
            logger.warn(f"Could not load TF backend: {e}")

        try:
            self.backends['yolo'] = YOLOPerceptionBackend(self.yolo_path, self.conf_thresh)
            logger.info(f"Loaded YOLO Backend")
        except Exception as e:
            logger.warn(f"Could not load YOLO backend: {e}")

        if self.backend_name not in self.backends:
            self.backend_name = list(self.backends.keys())[0] if self.backends else 'onnx'

    def switch_backend(self, new_backend: str) -> bool:
        """Hot-swaps the active inference backend on the fly."""
        new_backend = new_backend.lower()
        if new_backend in self.backends and self.backends[new_backend].is_loaded:
            self.backend_name = new_backend
            self.get_logger().info(f"🚀 Hot-swapped active backend to: [{new_backend.upper()}]")
            return True
        return False

    def process_frame(self, frame: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
        """Runs perception and decision pipeline on a single frame."""
        backend = self.backends.get(self.backend_name)
        if backend and backend.is_loaded:
            detection = backend.predict(frame)
        else:
            detection = {
                'classification': 'UNKNOWN',
                'confidence': 0.0,
                'raw_score': 0.5,
                'bbox': None,
                'inference_time_ms': 0.0,
                'backend': 'NONE'
            }

        decision = self.decision_engine.evaluate(detection)
        annotated_frame = self.annotate_frame(frame, detection, decision)
        return detection, decision, annotated_frame

    def annotate_frame(self, frame: np.ndarray, detection: dict, decision: dict) -> np.ndarray:
        """Draws visual HUD overlay on output frame."""
        annotated = frame.copy()
        cls = detection.get('classification', 'UNKNOWN')
        conf = detection.get('confidence', 0.0)
        bbox = detection.get('bbox')
        action = decision.get('action', 'SEARCHING')

        theme_color = (0, 230, 77) if cls == 'REAL' else ((40, 40, 255) if cls == 'FAKE' else (0, 215, 255))

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), theme_color, 2)
            label = f"{cls} ({conf*100:.1f}%)"
            cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, theme_color, 2)

        # Top overlay
        cv2.putText(annotated, f"ROS 2 NODE | BACKEND: {self.backend_name.upper()}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"ACTION: {action}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, theme_color, 2)
        return annotated

    def image_callback(self, msg: 'Image'):
        """ROS 2 Image Topic Callback."""
        if not ROS2_AVAILABLE:
            return

        try:
            if self.bridge:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
        except Exception as e:
            self.get_logger().error(f"Failed to decode ROS Image message: {e}")
            return

        # Run pipeline
        detection, decision, annotated = self.process_frame(cv_image)

        # Publish results
        det_msg = String()
        det_msg.data = json.dumps(detection)
        self.pub_detections.publish(det_msg)

        dec_msg = String()
        dec_msg.data = json.dumps(decision)
        self.pub_decision.publish(dec_msg)

        if self.pub_annotated:
            try:
                if self.bridge:
                    ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                    self.pub_annotated.publish(ann_msg)
            except Exception as ex:
                pass


def main(args=None):
    if ROS2_AVAILABLE:
        rclpy.init(args=args)
        node = RealFakePerceptionNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        node = RealFakePerceptionNode()
        print("⚡ RealFakePerceptionNode initialized successfully in standalone mode.")


if __name__ == '__main__':
    main()
