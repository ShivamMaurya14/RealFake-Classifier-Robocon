"""
Perception Backend Benchmark & Performance Profiler
Evaluates inference latency, FPS throughput, and CPU overhead across ONNX, YOLO, and TensorFlow backends.
"""

import time
import os
import sys
import argparse
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from realfake_perception.backends.onnx_backend import ONNXPerceptionBackend
from realfake_perception.backends.tf_cnn_backend import TensorFlowCNNBackend
from realfake_perception.backends.yolo_backend import YOLOPerceptionBackend


def run_benchmark(backend, dummy_frame: np.ndarray, num_warmup: int = 15, num_iterations: int = 100):
    """Profiles a backend over warm-up and test iterations."""
    name = backend.backend_name
    print(f"\n⚡ Profiling [{name}]...")

    if not backend.is_loaded:
        print(f"⚠️ {name} is not loaded (weights file missing). Skipping.")
        return None

    # Warm-up phase
    for _ in range(num_warmup):
        _ = backend.predict(dummy_frame)

    latencies = []
    start_total = time.perf_counter()

    for _ in range(num_iterations):
        t0 = time.perf_counter()
        res = backend.predict(dummy_frame)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    total_time = time.perf_counter() - start_total
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    fps = num_iterations / total_time

    return {
        'name': name,
        'avg_ms': avg_latency,
        'p50_ms': p50_latency,
        'p95_ms': p95_latency,
        'fps': fps,
        'sample_output': res
    }


def main():
    parser = argparse.ArgumentParser(description="Perception Engine FPS & Latency Benchmark")
    parser.add_argument("--onnx_model", type=str, default="models/cnn_realfake.onnx", help="Path to ONNX model")
    parser.add_argument("--tf_model", type=str, default="models/cnn_realfake.keras", help="Path to Keras model")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    args = parser.parse_args()

    print("=" * 70)
    print("🏎️ ABU ROBOCON PERCEPTION ENGINE - CPU BENCHMARK PROFILER")
    print("=" * 70)

    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    backends = []

    # 1. ONNX Backend
    try:
        onnx_b = ONNXPerceptionBackend(args.onnx_model)
        backends.append(onnx_b)
    except Exception as e:
        print(f"Could not init ONNX backend: {e}")

    # 2. TensorFlow CNN Backend
    try:
        tf_b = TensorFlowCNNBackend(args.tf_model)
        backends.append(tf_b)
    except Exception as e:
        print(f"Could not init TF backend: {e}")

    # 3. YOLO Backend
    try:
        yolo_b = YOLOPerceptionBackend(args.yolo_model)
        backends.append(yolo_b)
    except Exception as e:
        print(f"Could not init YOLO backend: {e}")

    results = []
    for b in backends:
        res = run_benchmark(b, dummy_frame, num_iterations=args.iterations)
        if res:
            results.append(res)

    print("\n" + "=" * 70)
    print("📊 BENCHMARK SUMMARY RESULTS (CPU ONLY)")
    print("=" * 70)
    print(f"{'Backend Engine':<25} | {'Avg Latency':<12} | {'P95 Latency':<12} | {'Throughput (FPS)':<16} | {'Target met (30+ FPS)'}")
    print("-" * 75)

    for r in results:
        fps_str = f"{r['fps']:.1f} FPS"
        passed = "✅ YES (30+ FPS)" if r['fps'] >= 30.0 else "⚠️ BELOW 30 FPS"
        print(f"{r['name']:<25} | {r['avg_ms']:>8.2f} ms   | {r['p95_ms']:>8.2f} ms   | {fps_str:>16} | {passed}")

    print("=" * 75)


if __name__ == "__main__":
    main()
