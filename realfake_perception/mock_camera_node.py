"""
ROS 2 Mock Camera Publisher Node
Publishes synthetic or dataset images to /camera/image_raw (sensor_msgs/Image) at 30 FPS for simulation testing.
"""

import os
import sys
import time
import glob
import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    class Node:
        def __init__(self, name):
            self.name = name
            print(f"[Mock Node] {name} running in standalone Python mode (rclpy not installed)")
        def declare_parameter(self, name, val): pass
        def get_parameter(self, name):
            class Param:
                value = None
            p = Param()
            p.value = None
            return p
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


class MockCameraNode(Node):
    """Publishes dataset images or synthetic arena scenes to /camera/image_raw at 30 FPS."""

    def __init__(self):
        super().__init__('mock_camera_node')
        logger = self.get_logger()
        logger.info("Initializing Robocon Mock Camera Publisher Node...")

        # Parameters
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('dataset_dir', 'dataset')
        self.declare_parameter('mode', 'synthetic') # 'synthetic' or 'dataset'
        self.declare_parameter('frame_id', 'camera_link')

        self.camera_topic = self.get_parameter('camera_topic').value or '/camera/image_raw'
        self.fps = float(self.get_parameter('fps').value or 30.0)
        self.dataset_dir = self.get_parameter('dataset_dir').value or 'dataset'
        self.mode = self.get_parameter('mode').value or 'synthetic'
        self.frame_id = self.get_parameter('frame_id').value or 'camera_link'

        self.bridge = CvBridge() if CV_BRIDGE_AVAILABLE else None
        self.image_files = []
        self.img_idx = 0

        if self.mode == 'dataset' and os.path.exists(self.dataset_dir):
            self.image_files = glob.glob(os.path.join(self.dataset_dir, '**', '*.jpg'), recursive=True)
            logger.info(f"Loaded {len(self.image_files)} sample dataset images for playback.")

        if ROS2_AVAILABLE:
            self.publisher = self.create_publisher(Image, self.camera_topic, 10)
            timer_period = 1.0 / max(1.0, self.fps)
            self.timer = self.create_timer(timer_period, self.timer_callback)
            logger.info(f"Publishing mock camera stream on {self.camera_topic} at {self.fps} FPS")

    def generate_synthetic_frame(self) -> np.ndarray:
        """Generates dynamic synthetic competition box scene."""
        frame = np.full((480, 640, 3), 25, dtype=np.uint8)

        # Draw Arena Grid Floor
        for y in range(0, 480, 40):
            cv2.line(frame, (0, y), (640, y), (40, 48, 60), 1)
        for x in range(0, 640, 40):
            cv2.line(frame, (x, 0), (x, 480), (40, 48, 60), 1)

        # Dynamic Moving Competition Box
        t = time.time()
        is_real = ((t % 8.0) < 4.0)
        box_x = int(230 + 110 * np.sin(t * 1.5))
        box_color = (0, 190, 70) if is_real else (40, 40, 230)

        cv2.rectangle(frame, (box_x, 140), (box_x + 180, 340), box_color, -1)
        cv2.rectangle(frame, (box_x, 140), (box_x + 180, 340), (255, 255, 255), 2)

        tag = "REAL BOX (TARGET)" if is_real else "FAKE BOX (OBSTACLE)"
        cv2.putText(frame, tag, (box_x + 10, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return frame

    def get_current_frame(self) -> np.ndarray:
        """Returns the next frame to publish."""
        if self.mode == 'dataset' and self.image_files:
            img_path = self.image_files[self.img_idx]
            frame = cv2.imread(img_path)
            self.img_idx = (self.img_idx + 1) % len(self.image_files)
            if frame is not None:
                return cv2.resize(frame, (640, 480))

        return self.generate_synthetic_frame()

    def timer_callback(self):
        """Timer callback publishing ROS 2 Image msg."""
        if not ROS2_AVAILABLE:
            return

        frame = self.get_current_frame()
        try:
            if self.bridge:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            else:
                msg = Image()
                msg.height, msg.width, _ = frame.shape
                msg.encoding = 'bgr8'
                msg.is_bigendian = False
                msg.step = msg.width * 3
                msg.data = frame.tobytes()

            msg.header.frame_id = self.frame_id
            self.publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing image: {e}")


def main(args=None):
    if ROS2_AVAILABLE:
        rclpy.init(args=args)
        node = MockCameraNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        node = MockCameraNode()
        print("MockCameraNode ready in standalone mode.")


if __name__ == '__main__':
    main()
