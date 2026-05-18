"""
Model Exporter & ONNX Optimizer Script
Exports trained Keras CNN or YOLO PyTorch models to optimized ONNX format for 30+ FPS CPU edge inference.
"""

import os
import argparse
import sys
import numpy as np


def train_and_export_cnn(dataset_dir: str, output_keras: str, output_onnx: str, epochs: int = 12):
    """Trains a CNN on dataset/ and exports both .keras and .onnx formats."""
    print("=" * 65)
    print("🚀 Training Robocon Real/Fake CNN & Exporting to ONNX")
    print("=" * 65)

    import tensorflow as tf
    from tensorflow.keras import layers, models

    img_size = (224, 224)
    batch_size = 16

    if not os.path.exists(dataset_dir):
        print(f"❌ Error: Dataset directory '{dataset_dir}' not found.")
        return False

    print(f"📂 Loading dataset from '{dataset_dir}'...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    # Build optimized CNN
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3), name="input_image"),
        layers.Rescaling(1.0 / 255.0),
        layers.Conv2D(32, (3, 3), activation='relu', name="conv2d_1"),
        layers.MaxPooling2D((2, 2), name="maxpool_1"),
        layers.Conv2D(64, (3, 3), activation='relu', name="conv2d_2"),
        layers.MaxPooling2D((2, 2), name="maxpool_2"),
        layers.Conv2D(128, (3, 3), activation='relu', name="conv2d_3"),
        layers.MaxPooling2D((2, 2), name="maxpool_3"),
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation='relu', name="dense_1"),
        layers.Dropout(0.5, name="dropout"),
        layers.Dense(1, activation='sigmoid', name="output_classification")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\n🧠 Training CNN Architecture:")
    model.summary()

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)

    os.makedirs(os.path.dirname(output_keras), exist_ok=True)
    os.makedirs(os.path.dirname(output_onnx), exist_ok=True)

    model.save(output_keras)
    print(f"\n✅ Saved Keras model to: {output_keras}")

    # Export to ONNX
    print(f"🔄 Converting Keras model to ONNX...")
    try:
        import tf2onnx
        @tf.function
        def serving(x):
            return model(x)

        input_signature = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name="input_image")]
        model_proto, _ = tf2onnx.convert.from_function(
            serving,
            input_signature=input_signature,
            opset=13,
            output_path=output_onnx
        )
        print(f"✅ Successfully exported ONNX model to: {output_onnx}")
    except Exception as ex:
        print(f"❌ Failed to convert to ONNX: {ex}")
        return False

    return True


def export_yolo(yolo_weights: str, output_onnx: str):
    """Exports Ultralytics YOLO model to ONNX format."""
    print(f"🔄 Exporting YOLO model '{yolo_weights}' to ONNX...")
    from ultralytics import YOLO
    model = YOLO(yolo_weights)
    success = model.export(format="onnx", imgsz=640, opset=12, dynamic=False)
    print(f"✅ YOLO ONNX Export Complete: {success}")
    return success


def main():
    parser = argparse.ArgumentParser(description="Model Exporter & ONNX Optimizer for Robocon Perception")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to dataset directory containing real/ and fake/")
    parser.add_argument("--output_keras", type=str, default="models/cnn_realfake.keras", help="Path to save .keras model")
    parser.add_argument("--output_onnx", type=str, default="models/cnn_realfake.onnx", help="Path to save .onnx model")
    parser.add_argument("--yolo_weights", type=str, default="yolov8n.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("--mode", type=str, choices=["all", "cnn", "yolo"], default="cnn", help="Export target")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for CNN")
    args = parser.parse_args()

    if args.mode in ["all", "cnn"]:
        train_and_export_cnn(args.dataset, args.output_keras, args.output_onnx, epochs=args.epochs)

    if args.mode in ["all", "yolo"]:
        export_yolo(args.yolo_weights, "models/yolo_realfake.onnx")


if __name__ == "__main__":
    main()
