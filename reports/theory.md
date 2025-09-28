# Part 1: Theoretical Understanding

## Q1. TensorFlow vs. PyTorch
- **Computation Graphs**: TensorFlow offers both static graphs (via graph mode) and eager execution, while PyTorch is eager-first with optional TorchScript for graph compilation. Static graphs can optimize deployment but add complexity during debugging; PyTorch's eager execution feels more Pythonic and debuggable.
- **Ecosystem & Deployment**: TensorFlow integrates tightly with `tf.keras`, TensorFlow Extended (TFX), TensorFlow Lite, and TensorFlow Serving, making it strong for production pipelines and on-device inference. PyTorch shines in research with dynamic graph flexibility, and its ecosystem includes PyTorch Lightning, TorchServe, and ONNX export.
- **When to Choose**: Prefer TensorFlow for mature production pipelines, TPU support, and cross-platform deployment. Choose PyTorch for rapid experimentation, dynamic model architectures, and when leveraging the vibrant research community (e.g., Hugging Face Transformers, PyTorch Geometric).

## Q2. Jupyter Notebook Use Cases
- **Exploratory Data Analysis (EDA)**: Notebooks interleave narratives, code, and inline visualizations to iteratively inspect datasets, test hypotheses, and document findings.
- **Interactive Tutorials & Prototyping**: Ideal for sharing reproducible demos, teaching materials, or quickly prototyping models with immediate feedback through cell-based execution and widget integrations.

## Q3. spaCy vs. Basic String Operations
- **Pretrained NLP Pipeline**: spaCy supplies tokenization, POS tagging, dependency parsing, and NER models optimized for accuracy and speed. Manual string operations miss linguistic structure and context.
- **Efficient NLP Utilities**: spaCy handles lemmatization, stop-word filtering, and vector representations with GPU support, whereas vanilla Python string handling lacks these advanced, performance-tuned features.

## Comparative Analysis: Scikit-learn vs. TensorFlow
- **Target Applications**: `scikit-learn` focuses on classical ML algorithms (linear models, tree-based methods, clustering) and utilities (pipelines, feature engineering). TensorFlow targets deep learning with automatic differentiation, neural network layers, and distributed training.
- **Ease of Use**: `scikit-learn` provides a uniform estimator API (`fit`, `predict`, `score`) that is beginner-friendly. TensorFlow requires understanding computational graphs, tensors, and model architectures, creating a steeper learning curve.
- **Community Support**: Both have large communities; `scikit-learn` is widely used in data science/analytics with extensive documentation and Stack Overflow support. TensorFlow benefits from Google backing, large-scale tutorials, and industry adoption, especially for deep learning.
