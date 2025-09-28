# Part 3: Ethics & Optimization

## Ethical Considerations
- **Bias in MNIST CNN**: Although MNIST is relatively balanced, class imbalance can still arise (e.g., fewer 5s than 1s). Model reliance on grayscale digits also ignores handwriting variations across diverse populations. *Mitigation*: use stratified sampling, augment underrepresented digits, and monitor fairness with tools like `TensorFlow Model Analysis` or `TensorFlow Fairness Indicators` to compare per-class metrics and detect disparities.
- **Bias in Amazon Reviews Sentiment/NER**: Reviews may over-represent certain brands or language styles, skewing entity recognition and rule-based sentiment. spaCy’s pretrained models can inherit societal biases (e.g., associating brand names with sentiment). *Mitigation*: expand training data to include diverse brands/languages, fine-tune spaCy pipelines with balanced corpora, and audit entity outputs for systematic errors. Incorporate rule-based overrides carefully and log flagged cases for human review.

## Troubleshooting Challenge (TensorFlow)
The original script failed with shape mismatches between `y_train` one-hot vectors and the loss function `SparseCategoricalCrossentropy`. It also fed raw `uint8` images into a `Conv2D` layer expecting normalized floats. Fixes applied:

```python
# Key excerpts from the corrected script (see `src/mnist_cnn_tf.py`)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",  # aligns with integer labels
    metrics=["accuracy"],
)

x_train = (x_train.astype("float32") / 255.0)[..., None]
x_test = (x_test.astype("float32") / 255.0)[..., None]
```

Additional safeguards:
- Added validation split and Dropout layers to reduce overfitting.
- Logged sample predictions and saved metrics to `reports/mnist_metrics.json` for reproducibility.
