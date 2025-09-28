# AI Tools Assignment — "Mastering the AI Toolkit"

## Repository Overview
- **`src/`**: Python scripts for each practical task.
- **`notebooks/`**: (optional) Jupyter notebooks for exploratory work.
- **`data/`**: Sample datasets (e.g., `amazon_reviews_sample.jsonl`).
- **`logos/`**: Placeholder for team branding assets.

## Deliverables Checklist
- **Theory**: `reports/theory.md` contains answers for Part 1 questions.
- **Classical ML**: `src/iris_classifier.py` plus metrics in `reports/iris_metrics.json`.
- **Delivery Notes**: `src/mnist_cnn_tf.py` with outputs in `reports/mnist_metrics.json`, training curves, and saved model.
- **NLP**: `src/spacy_ner_sentiment.py` and results in `reports/spacy_ner_sentiment.json`.
- **Ethics & Debugging**: Analysis documented in `reports/ethics_optimization.md`.
- **Bonus (optional)**: Streamlit interface located in `streamlit_app/` for MNIST demo. Live deployment: [Streamlit App](https://geoffominde-ai-tools-assignment-streamlit-appapp-yg4bxu.streamlit.app/)

## How to Run the Code
{{ ... }}
# Create virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Task 1: Iris classifier
py -3 src\iris_classifier.py

# Task 2: MNIST CNN (TensorFlow)
py -3 src\mnist_cnn_tf.py --epochs 5 --batch-size 128

# Task 3: spaCy NER & Sentiment
py -3 src\spacy_ner_sentiment.py

# Bonus: Streamlit MNIST demo
streamlit run streamlit_app/app.py

Live deployment: [Streamlit App](https://geoffominde-ai-tools-assignment-streamlit-appapp-yg4bxu.streamlit.app/)
```

## Report & Presentation Guidance
- Compile `reports/theory.md` and `reports/ethics_optimization.md` into the PDF submission alongside visuals from `reports/`.
- Include screenshots of `mnist_training_curves.png`, sample console outputs, and NER results.
- Prepare a 3-minute video where each teammate explains their contributions (theory, classical ML, deep learning, NLP, ethics/bonus). Consider recording screen demos while walking through the code and results.

## Requirements File
Make sure to populate `requirements.txt` with the versions used:
```
numpy
pandas
scikit-learn
tensorflow
spacy
en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
matplotlib
```
Adjust versions based on your environment.
