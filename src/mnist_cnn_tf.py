"""Convolutional Neural Network for MNIST classification using TensorFlow.

Usage:
    python mnist_cnn_tf.py --epochs 5 --batch-size 128

The script:
- Loads MNIST from TensorFlow Datasets (`tf.keras.datasets.mnist`).
- Normalizes and reshapes the data.
- Builds a small CNN with Conv2D, MaxPooling, Dropout, and Dense layers.
- Trains the model, evaluates on the test set, and saves metrics plus sample predictions.
- Saves training history plots if matplotlib is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting optional
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST using TensorFlow")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "reports",
        help="Directory to store metrics and artifacts",
    )
    return parser.parse_args()


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Reshape to (N, 28, 28, 1) and normalize to [0, 1]
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]
    return x_train, y_train, x_test, y_test


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    probs = model.predict(x_test[:5])
    preds = probs.argmax(axis=1)
    return {
        "history": history.history,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "sample_predictions": [
            {
                "true": int(true_label),
                "pred": int(pred_label),
                "probs": probs_row.tolist(),
            }
            for true_label, pred_label, probs_row in zip(y_test[:5], preds, probs)
        ],
    }


def save_metrics(metrics: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "mnist_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_history(history: Dict[str, Any], output_dir: Path) -> None:
    if plt is None:
        print("matplotlib not available; skipping plots")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="train")
    plt.plot(history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plot_path = output_dir / "mnist_training_curves.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def main() -> None:
    args = parse_args()
    x_train, y_train, x_test, y_test = load_data()
    model = build_model()

    metrics = train_and_evaluate(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    save_metrics(metrics, args.output_dir)
    plot_history(metrics["history"], args.output_dir)

    print("MNIST CNN evaluation")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test loss: {metrics['test_loss']:.4f}")
    for idx, pred in enumerate(metrics["sample_predictions"]):
        print(
            f"Sample {idx}: true={pred['true']} pred={pred['pred']} probs={np.round(pred['probs'], 3)}"
        )

    model_path = args.output_dir / "mnist_cnn_model.h5"
    model.save(model_path)
    print("Model saved to", model_path)


if __name__ == "__main__":
    main()
