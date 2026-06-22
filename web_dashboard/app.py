"""
Web Visualizer & Live Perception Telemetry Dashboard
Flask Server providing live MJPEG streaming, robot decision monitoring, and backend hot-swapping.
"""

import os
import sys
import time
import json
import argparse
import threading
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from realfake_perception.perception_node import RealFakePerceptionNode

app = Flask(__name__)

# Global Perception Node Instance
perception_node = None
camera_source = 0
is_dummy_mode = False
is_pipeline_active = True
camera_cap = None
frame_lock = threading.Lock()
latest_frame = None
latest_telemetry = {
    'classification': 'UNKNOWN',
    'confidence': 0.0,
    'action': 'SEARCHING',
    'target_locked': False,
    'offset_x': 0,
    'offset_y': 0,
    'heading_error_deg': 0.0,
    'distance_proxy': 0.0,
    'gripper_command': 'IDLE',
    'status_message': 'System Ready. Initializing Perception Node...',
    'backend': 'ONNX_RUNTIME_CPU',
    'fps': 0.0,
    'latency_ms': 0.0,
    'is_active': True
}


def camera_loop():
    """Background thread capturing video frames and running perception node."""
    global camera_cap, latest_frame, latest_telemetry, is_dummy_mode, is_pipeline_active

    fps_count = 0
    fps_timer = time.time()
    current_fps = 0.0

    while True:
        if not is_pipeline_active:
            # Standby Mode - Hardware Powered Down (0% CPU)
            if camera_cap is not None:
                with frame_lock:
                    camera_cap.release()
                    camera_cap = None
                print("🛑 Camera hardware released (0% CPU).")

            # Draw Standby Frame
            standby_frame = np.full((480, 640, 3), 15, dtype=np.uint8)
            cv2.rectangle(standby_frame, (100, 180), (540, 300), (30, 41, 59), -1)
            cv2.rectangle(standby_frame, (100, 180), (540, 300), (51, 65, 85), 2)
            cv2.putText(standby_frame, "PERCEPTION STANDBY", (175, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (148, 163, 184), 2)
            cv2.putText(standby_frame, "HARDWARE POWERED OFF (0% CPU)", (140, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (239, 68, 68), 2)

            latest_frame = standby_frame
            latest_telemetry.update({
                'action': 'STANDBY_OFF',
                'status_message': 'Hardware powered off (0% CPU). Click START to resume.',
                'fps': 0.0,
                'latency_ms': 0.0,
                'is_active': False
            })
            time.sleep(0.1)
            continue

        # Pipeline is Active
        if is_dummy_mode:
            # Synthetic Competition Arena Simulation
            frame = np.full((480, 640, 3), 20, dtype=np.uint8)
            # Grid floor
            for y in range(0, 480, 40):
                cv2.line(frame, (0, y), (640, y), (35, 42, 54), 1)
            for x in range(0, 640, 40):
                cv2.line(frame, (x, 0), (x, 480), (35, 42, 54), 1)

            # Draw oscillating Robocon Box
            t = time.time()
            cycle = (t % 10.0)
            is_real = cycle < 5.0
            box_x = int(230 + 100 * np.sin(t * 1.2))
            box_color = (0, 180, 60) if is_real else (40, 40, 220)

            # Draw box body
            cv2.rectangle(frame, (box_x, 140), (box_x + 180, 340), box_color, -1)
            cv2.rectangle(frame, (box_x, 140), (box_x + 180, 340), (255, 255, 255), 2)

            box_title = "REAL BOX (TARGET)" if is_real else "FAKE BOX (OBSTACLE)"
            cv2.putText(frame, box_title, (box_x + 10, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            time.sleep(0.03)
        else:
            if camera_cap is None or not camera_cap.isOpened():
                camera_cap = init_camera(camera_source)
                if camera_cap is None:
                    time.sleep(0.1)
                    continue

            with frame_lock:
                ret, frame = camera_cap.read()
            if not ret:
                time.sleep(0.01)
                continue

        # Run Perception & Decision Pipeline
        if perception_node:
            t0 = time.perf_counter()
            det, dec, annotated = perception_node.process_frame(frame)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            fps_count += 1
            if time.time() - fps_timer >= 0.5:
                current_fps = fps_count / (time.time() - fps_timer)
                fps_count = 0
                fps_timer = time.time()

            latest_telemetry = {
                'classification': det.get('classification', 'UNKNOWN'),
                'confidence': det.get('confidence', 0.0),
                'action': dec.get('action', 'SEARCHING'),
                'target_locked': dec.get('target_locked', False),
                'offset_x': dec.get('offset_x', 0),
                'offset_y': dec.get('offset_y', 0),
                'heading_error_deg': dec.get('heading_error_deg', 0.0),
                'distance_proxy': dec.get('distance_proxy', 0.0),
                'gripper_command': dec.get('gripper_command', 'IDLE'),
                'status_message': dec.get('status_message', ''),
                'backend': perception_node.backend_name.upper(),
                'fps': round(current_fps, 1),
                'latency_ms': round(lat_ms, 1),
                'is_active': True
            }
            latest_frame = annotated
        else:
            latest_frame = frame


def generate_frames():
    """Generates MJPEG video stream."""
    global latest_frame
    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def get_status():
    return jsonify(latest_telemetry)


@app.route('/api/toggle_pipeline', methods=['POST'])
def toggle_pipeline():
    global is_pipeline_active
    data = request.json or {}
    action = data.get('action') # 'start', 'stop', or 'toggle'

    if action == 'start':
        is_pipeline_active = True
    elif action == 'stop':
        is_pipeline_active = False
    else:
        is_pipeline_active = not is_pipeline_active

    return jsonify({'status': 'success', 'is_active': is_pipeline_active})


@app.route('/api/backend', methods=['POST'])
def switch_backend():
    data = request.json or {}
    new_backend = data.get('backend', 'onnx')
    if perception_node:
        success = perception_node.switch_backend(new_backend)
        return jsonify({'status': 'success' if success else 'error', 'backend': perception_node.backend_name})
    return jsonify({'status': 'error', 'message': 'Perception node not ready'}), 500


def init_camera(source):
    """Initializes real-time camera capture with low-latency settings."""
    global camera_cap
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if cap.isOpened():
        # Set low-latency streaming properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"📷 Live Camera successfully connected on source [{source}] (640x480 @ 30 FPS)")
        return cap
    return None


@app.route('/api/process_browser_frame', methods=['POST'])
def process_browser_frame():
    """Receives live base64 frame from browser webcam and runs perception."""
    global latest_telemetry, latest_frame
    try:
        data = request.json or {}
        img_b64 = data.get('image')
        if not img_b64:
            return jsonify({'status': 'error', 'message': 'No image data'}), 400

        # Decode base64 JPEG
        import base64
        header, encoded = img_b64.split(",", 1) if "," in img_b64 else ("", img_b64)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is not None and perception_node:
            t0 = time.perf_counter()
            det, dec, annotated = perception_node.process_frame(frame)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            latest_telemetry = {
                'classification': det.get('classification', 'UNKNOWN'),
                'confidence': det.get('confidence', 0.0),
                'action': dec.get('action', 'SEARCHING'),
                'target_locked': dec.get('target_locked', False),
                'offset_x': dec.get('offset_x', 0),
                'offset_y': dec.get('offset_y', 0),
                'heading_error_deg': dec.get('heading_error_deg', 0.0),
                'distance_proxy': dec.get('distance_proxy', 0.0),
                'gripper_command': dec.get('gripper_command', 'IDLE'),
                'status_message': dec.get('status_message', ''),
                'backend': perception_node.backend_name.upper(),
                'fps': 30.0,
                'latency_ms': round(lat_ms, 1)
            }
            latest_frame = annotated
            return jsonify(latest_telemetry)

        return jsonify({'status': 'error'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


def main():
    global perception_node, camera_cap, is_dummy_mode, camera_source

    parser = argparse.ArgumentParser(description="Robocon Perception Web Dashboard")
    parser.add_argument("--port", type=int, default=5002, help="HTTP Server Port")
    parser.add_argument("--source", type=str, default="0", help="Camera source (e.g. 0, 1, /dev/video0, or video file)")
    parser.add_argument("--dummy", action="store_true", help="Force synthetic simulation mode")
    parser.add_argument("--backend", type=str, default="onnx", help="Default inference backend (onnx, yolo, tf)")
    args = parser.parse_args()

    is_dummy_mode = args.dummy
    camera_source = args.source

    # Initialize Perception Node
    perception_node = RealFakePerceptionNode()
    perception_node.switch_backend(args.backend)

    # Attempt to open physical real-time camera
    if not is_dummy_mode:
        camera_cap = init_camera(camera_source)
        if camera_cap is None:
            # Try index 0 fallback
            if camera_source != "0":
                camera_cap = init_camera(0)

            if camera_cap is None:
                print(f"⚠️ Physical camera source [{camera_source}] not accessible. Switching to simulation arena mode.")
                is_dummy_mode = True

    # Start background capture thread
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

    print(f"\n🌐 Robocon Perception Dashboard active at: http://localhost:{args.port}")
    if is_dummy_mode:
        print("💡 Running in Simulation Mode. To use live camera: python web_dashboard/app.py --source 0")
    else:
        print(f"📹 Streaming REAL-TIME Camera Feed from source [{camera_source}]")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
