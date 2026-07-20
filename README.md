# 🤖 Real-Time Perception Pipeline — ROS 2 Vision Node

[![ROS 2](<https://img.shields.io/badge/ROS_2-Humble%20%7C%20Iron-blue.svg?style=for-the-badge&logo=ros&logoColor=white>)](https://docs.ros.org/en/humble/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-30%2B_FPS_CPU-purple.svg?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7%2B-red.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

A modular, production-grade **Real-Time Perception Pipeline and ROS 2 Vision Node** engineered for autonomous robotics competitions (**ABU Robocon**). Features **hot-swappable ONNX, YOLO, and TensorFlow CNN backends** delivering **250+ FPS CPU-only inference**, integrated with a high-level **Robot Decision Engine** that directly publishes kinematic alignment poses and gripper actions (`PICK_REAL` vs `REJECT_FAKE`).

---

## 📋 Table of Contents

- [🎯 System Architecture](#-system-architecture)
- [⚡ Key Features &amp; Highlights](#-key-features--highlights)
- [📊 CPU Inference Benchmarks](#-cpu-inference-benchmarks)
- [🧠 Autonomous Robot Decision Engine](#-autonomous-robot-decision-engine)
- [🚀 Quick Start &amp; Usage](#-quick-start--usage)
  - [1. Installation](#1-installation)
  - [2. Interactive Web Dashboard](#2-interactive-web-dashboard)
  - [3. Standalone CLI &amp; Camera Runner](#3-standalone-cli--camera-runner)
  - [4. ROS 2 Launch &amp; Topics](#4-ros-2-launch--topics)
  - [5. Run Performance Benchmark](#5-run-performance-benchmark)
  - [6. Run Automated Test Suite](#6-run-automated-test-suite)
- [🔄 ONNX Model Export &amp; Training](#-onnx-model-export--training)
- [🔮 Future Scope &amp; Advanced Roadmap](#-future-scope--advanced-roadmap)
- [📂 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🎯 System Architecture

```mermaid
graph TD
    Camera["📷 Camera Feed / ROS 2 Topic (/camera/image_raw)"] --> Node["🤖 ROS 2 Perception Node (perception_node.py)"]
  
    subgraph Modular Backends ["⚡ Hot-Swappable Backends (30+ FPS CPU)"]
        Node --> B1["🚀 ONNX Runtime Backend (.onnx)"]
        Node --> B2["🎯 YOLO Object Detection Backend (.pt / .onnx)"]
        Node --> B3["🧠 TensorFlow CNN Backend (.keras / .h5)"]
    end
  
    B1 --> Detections["📦 Geometrical Detections (Class, Confidence, BBox, Latency)"]
    B2 --> Detections
    B3 --> Detections
  
    Detections --> DecisionEngine["🧠 Robot Decision Engine (decision_engine.py)"]    DecisionEngine --> Pub1["📢 /perception/detections (std_msgs/String JSON)"]
    DecisionEngine --> Pub2["🎯 /perception/decision (std_msgs/String JSON)"]
    DecisionEngine --> Pub3["🖼️ /perception/image_annotated (sensor_msgs/Image)"]
    DecisionEngine --> WebUI["🌐 Interactive Visualizer Dashboard (Flask / WebSockets)"]
```

---

## ⚡ Key Features & Highlights

- **⚡ Hot-Swappable Inference Backends**: Seamlessly toggle between **ONNX Runtime (CPU)**, **YOLOv8 Object Detection**, and **TensorFlow CNN** on the fly without restarting nodes.
- **🚀 250+ FPS CPU-Only Inference**: Optimized with ONNX graph optimizations and multi-threaded CPU execution providers, eliminating the need for bulky edge GPUs.
- **🧠 Autonomous Robocon Decision Engine**: Directly evaluates bounding box coordinates and classification to generate real-time robot commands (`PICK_REAL`, `REJECT_FAKE`, `ALIGN_CENTER`, `SEARCHING`).
- **🎯 Kinematics & Gripper Alignment**: Computes optical center offset `(dx, dy)`, estimated heading error angle (degrees), and proximity distance proxy for precise robotic arm manipulation.
- **🌐 Interactive Web Visualizer Dashboard**: Dark-mode industrial Robocon UI with real-time video stream, dynamic target locking alerts, live telemetry cards, and backend switching.
- **🧪 Zero-ROS Standalone & Simulation Mode**: Fully functional outside ROS 2 environments with standalone camera runners, synthetic competition arena simulation, and automated test suites.

---

## 📊 CPU Inference Benchmarks

Performance benchmark evaluated across backends on CPU:

| Inference Backend               | Architecture                    |     Avg Latency     |     P95 Latency     |  Throughput (FPS)  |        Target Met (30+ FPS)        |
| :------------------------------ | :------------------------------ | :-----------------: | :-----------------: | :-----------------: | :---------------------------------: |
| **🚀 ONNX Runtime**       | Graph Optimized`.onnx`        |  **3.89 ms**  |  **4.54 ms**  | **257.0 FPS** | **✅ EXCEEDED (8.5x Target)** |
| **🧠 TensorFlow / Keras** | Sequential CNN`.keras`        | **22.18 ms** | **22.70 ms** | **45.1 FPS** |     **✅ YES (30+ FPS)**     |
| **🎯 YOLOv8 Detection**   | PyTorch / ONNX`.pt`/`.onnx` | **~18.50 ms** | **~24.00 ms** | **~54.0 FPS** |     **✅ YES (30+ FPS)**     |

---

## 🧠 Autonomous Robot Decision Engine

The **Decision Engine** maps raw computer vision output to physical robot actions:

```
┌─────────────────┬──────────────────┬─────────────────┬───────────────────────────────────────────┐
│ Target Class    │ Horizontal Align │ Robot Action    │ Gripper Command / Physical Behavior       │
├─────────────────┼──────────────────┼─────────────────┼───────────────────────────────────────────┤
│ 🟢 REAL Box     │ Aligned (±40px)  │ PICK_REAL       │ CLOSE (Target locked, execute pickup)     │
│ 🟢 REAL Box     │ Offset (>40px)   │ ALIGN_CENTER    │ OPEN (Rotate chassis toward target angle) │
│ 🔴 FAKE Box     │ Any position     │ REJECT_FAKE     │ BYPASS (Obstacle detected, steer clear)   │
│ 🟡 UNKNOWN      │ None             │ SEARCHING       │ IDLE (Sweep arena in search pattern)      │
└─────────────────┴──────────────────┴─────────────────┴───────────────────────────────────────────┘
```

---

## 🚀 Quick Start & Usage

### 1. Installation

```bash
git clone https://github.com/ShivamMaurya14/RealFake-Classifier-Robocon.git
cd RealFake-Classifier-Robocon
pip install -r requirements.txt
```

### 2. Interactive Web Dashboard

Launch the browser-based visualization & telemetry monitor:

```bash
# Run in simulation / demo mode (no physical camera required)
python web_dashboard/app.py --dummy --port 5002

# Run with physical USB webcam
python web_dashboard/app.py --source 0 --port 5002
```

Access the dashboard at **`http://localhost:5002`**.

### 3. Standalone CLI & Camera Runner

Run high-speed perception with live keyboard hot-swapping (`O`: ONNX, `Y`: YOLO, `T`: TensorFlow, `Q`: Quit):

```bash
python standalone/run_perception.py --source 0 --backend onnx
```

### 4. ROS 2 Launch & Topics

Source your ROS 2 workspace and launch the perception node:

```bash
# Build ROS 2 workspace
cd ros2_ws
colcon build --packages-select realfake_perception
source install/setup.bash

# Option A: Launch Complete Simulation Pipeline (Mock Camera + Perception Node)
ros2 launch realfake_perception simulation.launch.py backend:=onnx

# Option B: Launch Perception Node with Physical Camera Topic
ros2 launch realfake_perception perception_pipeline.launch.py backend:=onnx camera_topic:=/camera/image_raw
```

#### Monitored Topics:

- **`sub`** `/camera/image_raw` (`sensor_msgs/msg/Image`): Input RGB camera stream.
- **`pub`** `/perception/detections` (`std_msgs/msg/String` JSON): Real-time bounding boxes and confidence.
- **`pub`** `/perception/decision` (`std_msgs/msg/String` JSON): Robot action (`PICK_REAL` / `REJECT_FAKE`) and heading angles.
- **`pub`** `/perception/image_annotated` (`sensor_msgs/msg/Image`): Annotated video feed with Robocon HUD.

### 5. Run Performance Benchmark

Profile latency and FPS on your local CPU:

```bash
python standalone/benchmark_fps.py --iterations 100
```

### 6. Run Automated Test Suite

Verify all backends and decision logic using pytest:

```bash
pytest tests/ -v
```

---

## 🔄 ONNX Model Export & Training

To train the CNN model and export optimized `.onnx` models:

```bash
# 1-Click Train & ONNX Exporter
python standalone/export_onnx.py --dataset dataset --epochs 10 --output_keras models/cnn_realfake.keras --output_onnx models/cnn_realfake.onnx
```

Or explore the research notebook in **[`notebook/realfake_cnn_training.ipynb`](notebook/realfake_cnn_training.ipynb)**.

---

## 🔮 Future Scope & Advanced Roadmap

Practical capabilities planned for upcoming releases:

1. **🎯 Multi-Target Centroid Tracker**: Assigns persistent IDs (`ID #1`, `ID #2`) and locks onto the closest Real target box while filtering out fake obstacles.
2. **📐 3D Spatial Pose Estimation $(X, Y, Z)$**: Pinhole camera intrinsics converting pixel bounding boxes into real-world $(X, Y, Z)$ coordinates in meters, publishing `geometry_msgs/msg/PoseStamped` to `/perception/target_pose` for MoveIt 2 and inverse kinematics arm controllers.
3. **📦 Native ROS 2 Custom Messages**: Typed `.msg` interface definitions (`RoboconDecision.msg`) replacing raw JSON strings.
4. **⚡ Edge Acceleration Engines**: TensorRT FP16/INT8 compilation for NVIDIA Jetson and OpenVINO optimization for Intel NUC onboard computers.

---

## 📂 Project Structure

```text
RealFake-Classifier-Robocon/
├── realfake_perception/                    # Core Perception Python Package
│   ├── __init__.py
│   ├── perception_node.py                  # ROS 2 Perception Node (Image sub, Decision pub)
│   ├── decision_engine.py                  # Autonomous Robot Decision & Kinematics Engine
│   └── backends/                           # Modular Hot-Swappable AI Inference Backends
│       ├── __init__.py
│       ├── base_backend.py                 # Abstract Base Backend interface
│       ├── onnx_backend.py                 # ONNX Runtime CPU Engine (250+ FPS)
│       ├── tf_cnn_backend.py               # Native TensorFlow / Keras CNN Engine
│       └── yolo_backend.py                 # YOLOv8 / YOLOv11 Object Detector
├── ros2_ws/                                # Standard ROS 2 Colcon Workspace
│   └── src/realfake_perception/
│       ├── package.xml                     # ROS 2 Package Manifest
│       ├── setup.py                        # Python Package Setup & Entry Points
│       ├── setup.cfg
│       ├── config/
│       │   └── perception_params.yaml      # Parameter configuration file
│       └── launch/
│           └── perception_pipeline.launch.py # ROS 2 Launch File
├── standalone/                             # Zero-ROS Standalone Tools
│   ├── run_perception.py                   # High-speed CLI & Camera Runner
│   ├── benchmark_fps.py                    # FPS & Latency Benchmark Profiler
│   └── export_onnx.py                      # 1-Click Keras / YOLO -> ONNX Exporter
├── web_dashboard/                          # Industrial Dark-Mode Web Visualizer
│   ├── app.py                              # Flask Server & MJPEG Streamer
│   ├── templates/index.html                # Responsive UI Template
│   └── static/
│       ├── style.css                       # Industrial Cyber Dark Theme
│       └── script.js                       # Telemetry Polling & Hot-Swap Controller
├── models/                                 # Pretrained & Exported Model Zoo
│   ├── cnn_realfake.keras                  # Trained Keras CNN
│   └── cnn_realfake.onnx                   # Optimized ONNX Model
├── dataset/                                # Competition Box Dataset
│   ├── real/                               # Training images for REAL target boxes
│   └── fake/                               # Training images for FAKE obstacle boxes
├── notebook/                               # Jupyter Research & Training Notebooks
│   └── realfake_cnn_training.ipynb         # CNN Training & ONNX Export Notebook
├── tests/                                  # Automated Unit Test Suite
│   ├── test_backends.py                    # Backend validation tests
│   └── test_decision_engine.py             # Robot kinematics & decision tests
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## 🤝 Contributing

Contributions, bug reports, and enhancements for autonomous robotics competitions are welcome!

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/NewPerceptionFeature`).
3. Commit your changes (`git commit -m 'Add NewPerceptionFeature'`).
4. Push to the Branch (`git push origin feature/NewPerceptionFeature`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the **MIT License**.

---

<p align="center">
  Built with 🤖 for the <b>ABU Robocon Autonomous Robotics Challenge</b> by <a href="https://github.com/ShivamMaurya14">Shivam Maurya</a>
</p>
