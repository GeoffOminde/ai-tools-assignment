"""Streamlit UI for the MNIST CNN classifier (bonus task).

Run with:
    streamlit run streamlit_app/app.py

Features:
- Loads the trained Keras model saved at `reports/mnist_cnn_model.h5`.
- Lets users preview test-set digits or upload their own 28x28 grayscale image.
- Displays predicted class with confidence scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent.parent / "reports" / "mnist_cnn_model.h5"
METRICS_PATH = Path(__file__).resolve().parent.parent / "reports" / "mnist_metrics.json"


@st.cache_resource
def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "MNIST model file not found. Train the model with src/mnist_cnn_tf.py first."
        )
    try:
        return tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"tf": tf},
            compile=False
        )
    except Exception as e:
        # Show detailed message directly in Streamlit
        st.error(f"❌ Model loading failed: {type(e).__name__}: {e}")
        raise



@st.cache_resource
def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_test, y_test


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert an uploaded image to model-ready tensor."""
    image = image.convert("L").resize((28, 28))
    arr = np.array(image).astype("float32")
    # Normalize to [0, 1] and invert if background is white
    if arr.mean() > 127:
        arr = 255 - arr
    arr /= 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr


def main() -> None:
    st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✏️")
    st.title("MNIST Digit Classifier")
    st.caption("Bonus Streamlit app for the AI Tools Assignment")

    try:
        model = load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    x_test, y_test = load_dataset()

    st.sidebar.header("Options")
    mode = st.sidebar.radio("Prediction mode", ("Sample from test set", "Upload image"))

    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        st.sidebar.metric("Test accuracy", f"{metrics.get('test_accuracy', 0):.3f}")
    else:
        st.sidebar.info("Train the model to generate metrics in reports/mnist_metrics.json")

    if mode == "Sample from test set":
        idx = st.slider("Select test image index", min_value=0, max_value=len(x_test) - 1, value=0)
        image = x_test[idx]
        label = int(y_test[idx])
        st.subheader(f"Test set image #{idx}")
        st.image(image, caption=f"True label: {label}", width=196)
        batch = image.reshape(1, 28, 28, 1).astype("float32") / 255.0
    else:
        uploaded = st.file_uploader("Upload a 28x28 grayscale digit (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded is None:
            st.info("Upload an image to run a prediction. For best results use white digit on black background.")
            st.stop()
        image = Image.open(uploaded)
        batch = preprocess_image(image)
        st.subheader("Uploaded image (resized to 28x28)")
        st.image(batch.reshape(28, 28), width=196)
        label = None

    probs = model.predict(batch, verbose=0)[0]
    pred_digit = int(np.argmax(probs))
    confidence = float(np.max(probs))

    st.success(f"Predicted digit: {pred_digit} (confidence: {confidence:.2%})")
    st.write("### Probability distribution")
    chart_data = {str(i): probs[i] for i in range(10)}
    st.bar_chart(chart_data)

    if label is not None:
        st.write(f"**True label:** {label}")

    st.divider()
    st.markdown(
        "Model loaded from `reports/mnist_cnn_model.h5`. "
        "Retrain with `py -3 src/mnist_cnn_tf.py --epochs 5` to refresh the weights."
    )


if __name__ == "__main__":
    main()
