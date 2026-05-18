"""
Standalone Perception Pipeline Runner
Runs live camera or test frames with interactive hot-swappable backends (ONNX, YOLO, TensorFlow).
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from realfake_perception.backends.onnx_backend import ONNXPerceptionBackend
from realfake_perception.backends.tf_cnn_backend import TensorFlowCNNBackend
from realfake_perception.backends.yolo_backend import YOLOPerceptionBackend


def draw_hud(frame: np.ndarray, result: dict, active_backend: str, fps: float):
    """Draws styled Robocon Perception HUD overlay on video frame."""
    h, w = frame.shape[:2]
    cls = result.get('classification', 'UNKNOWN')
    conf = result.get('confidence', 0.0)
    lat_ms = result.get('inference_time_ms', 0.0)
    bbox = result.get('bbox')

    # Color scheme: REAL = Neon Green, FAKE = Bright Red, UNKNOWN = Yellow
    if cls == 'REAL':
        theme_color = (0, 230, 77)      # BGR Green
        decision_text = "PICK_REAL (TARGET ACQUIRED)"
    elif cls == 'FAKE':
        theme_color = (40, 40, 255)     # BGR Red
        decision_text = "REJECT_FAKE (BYPASS)"
    else:
        theme_color = (0, 215, 255)     # BGR Yellow
        decision_text = "SEARCHING_TARGET"

    # Draw Bounding Box
    if bbox:
        x1, y1, x2, y2 = bbox
        # Corner brackets styling
        cv2.rectangle(frame, (x1, y1), (x2, y2), theme_color, 2)
        # Corner highlights
        line_len = min(25, (x2 - x1) // 4)
        cv2.line(frame, (x1, y1), (x1 + line_len, y1), theme_color, 4)
        cv2.line(frame, (x1, y1), (x1, y1 + line_len), theme_color, 4)
        cv2.line(frame, (x2, y1), (x2 - line_len, y1), theme_color, 4)
        cv2.line(frame, (x2, y1), (x2, y1 + line_len), theme_color, 4)
        cv2.line(frame, (x1, y2), (x1 + line_len, y2), theme_color, 4)
        cv2.line(frame, (x1, y2), (x1, y2 - line_len), theme_color, 4)
        cv2.line(frame, (x2, y2), (x2 - line_len, y2), theme_color, 4)
        cv2.line(frame, (x2, y2), (x2, y2 - line_len), theme_color, 4)

        # Classification label pill
        label_text = f"{cls} : {conf * 100:.1f}%"
        (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - 26)), (x1 + lw + 12, max(26, y1)), theme_color, -1)
        cv2.putText(frame, label_text, (x1 + 6, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Top-Left Telemetry Box
    cv2.rectangle(frame, (10, 10), (380, 125), (15, 23, 42), -1)
    cv2.rectangle(frame, (10, 10), (380, 125), (51, 65, 85), 1)

    cv2.putText(frame, "🤖 ROBOCON PERCEPTION NODE", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"Backend : {active_backend}", (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (56, 189, 248), 1)
    cv2.putText(frame, f"Decision: {decision_text}", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.48, theme_color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {lat_ms:.1f} ms", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)
    cv2.putText(frame, "[O] ONNX  [Y] YOLO  [T] TF  [Q] Quit", (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 116, 139), 1)


def main():
    parser = argparse.ArgumentParser(description="Standalone Robocon Perception Runner")
    parser.add_argument("--source", type=str, default="0", help="Camera source index (e.g. 0) or path to video/image file")
    parser.add_argument("--backend", type=str, choices=["onnx", "yolo", "tf"], default="onnx", help="Initial inference backend")
    parser.add_argument("--onnx_model", type=str, default="models/cnn_realfake.onnx", help="Path to ONNX weights")
    parser.add_argument("--tf_model", type=str, default="models/cnn_realfake.keras", help="Path to Keras weights")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt", help="Path to YOLO weights")
    parser.add_argument("--dummy", action="store_true", help="Run with simulated mock video frames")
    args = parser.parse_args()

    print("=" * 65)
    print("🤖 Launching Standalone Robocon Perception Pipeline")
    print("=" * 65)

    # Initialize Backends
    backends = {}
    try:
        backends['onnx'] = ONNXPerceptionBackend(args.onnx_model)
    except Exception as e:
        print(f"⚠️ Could not load ONNX backend: {e}")

    try:
        backends['tf'] = TensorFlowCNNBackend(args.tf_model)
    except Exception as e:
        print(f"⚠️ Could not load TF backend: {e}")

    try:
        backends['yolo'] = YOLOPerceptionBackend(args.yolo_model)
    except Exception as e:
        print(f"⚠️ Could not load YOLO backend: {e}")

    active_key = args.backend if args.backend in backends else list(backends.keys())[0]
    print(f"🚀 Active Inference Backend: [{active_key.upper()}]")

    # Setup Video Source
    if args.dummy:
        cap = None
    else:
        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"📷 Live Camera connected on source [{args.source}] (640x480 @ 30 FPS)")
        else:
            print(f"⚠️ Cannot open camera source {args.source}. Switching to simulated test frames.")
            cap = None

    fps = 0.0
    frame_count = 0
    fps_timer = time.time()

    # If test image source
    if isinstance(args.source, str) and args.source.endswith(('.jpg', '.png', '.jpeg')):
        frame = cv2.imread(args.source)
        if frame is not None:
            res = backends[active_key].predict(frame)
            draw_hud(frame, res, backends[active_key].backend_name, 30.0)
            cv2.imshow("Robocon Perception Node", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

    # Video Loop
    while True:
        t_start = time.perf_counter()

        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        else:
            # Generate simulated competition box scene
            frame = np.full((480, 640, 3), 30, dtype=np.uint8)
            # Add grid floor
            for y in range(0, 480, 40):
                cv2.line(frame, (0, y), (640, y), (45, 45, 45), 1)
            for x in range(0, 640, 40):
                cv2.line(frame, (x, 0), (x, 480), (45, 45, 45), 1)
            # Draw synthetic box
            box_x = int(220 + 80 * np.sin(time.time() * 1.5))
            cv2.rectangle(frame, (box_x, 140), (box_x + 180, 340), (160, 110, 60), -1)
            cv2.putText(frame, "COMPETITION BOX", (box_x + 10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Run Active Backend
        current_backend = backends.get(active_key)
        if current_backend:
            result = current_backend.predict(frame)
            backend_label = current_backend.backend_name
        else:
            result = {'classification': 'UNKNOWN', 'confidence': 0.0, 'inference_time_ms': 0.0, 'bbox': None}
            backend_label = "NONE"

        # Calculate FPS
        frame_count += 1
        if time.time() - fps_timer >= 0.5:
            fps = frame_count / (time.time() - fps_timer)
            frame_count = 0
            fps_timer = time.time()

        draw_hud(frame, result, backend_label, fps)

        cv2.imshow("Robocon Perception Node", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('o') and 'onnx' in backends:
            active_key = 'onnx'
            print(f"🔄 Switched backend to: ONNX Runtime (CPU)")
        elif key == ord('y') and 'yolo' in backends:
            active_key = 'yolo'
            print(f"🔄 Switched backend to: YOLO Object Detection")
        elif key == ord('t') and 'tf' in backends:
            active_key = 'tf'
            print(f"🔄 Switched backend to: TensorFlow CNN")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
