# AI Tools Assignment — "Mastering the AI Toolkit"

## Repository Overview
- **`src/`**: Python scripts for each practical task.
- **`notebooks/`**: (optional) Jupyter notebooks for exploratory work.
- **`data/`**: Sample datasets (e.g., `amazon_reviews_sample.jsonl`).
- **`logos/`**: Placeholder for team branding assets.

## Deliverables Checklist
- **Theory**: `reports/theory.md` contains answers for Part 1 questions.
- **Classical ML**: `src/iris_classifier.py` with metrics saved to `reports/iris_metrics.json`.
- **Deep Learning**: `src/mnist_cnn_tf.py` with outputs in `reports/mnist_metrics.json`, training curves, and saved model.
- **NLP**: `src/spacy_ner_sentiment.py` and results in `reports/spacy_ner_sentiment.json`.
- **Ethics & Debugging**: Analysis documented in `reports/ethics_optimization.md`.
- **Screenshots**: Stored under `reports/screenshots/` for use in the PDF and presentation.
- **Bonus (optional)**: Streamlit interface in `streamlit_app/`. Live deployment: [Streamlit App](https://geoffominde-ai-tools-assignment-streamlit-appapp-yg4bxu.streamlit.app/)

## How to Run the Code
```powershell
# (Optional) create and activate a virtual environment
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

# Bonus: Streamlit MNIST demo (local)
streamlit run streamlit_app/app.py
```

> **Deployment note:** Streamlit Cloud is pinned to Python 3.11 via `runtime.txt` so spaCy installs successfully (Python 3.13 images fail to build `blis`).

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
matplotlib
pillow
spacy==3.7.4
streamlit
en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```
Adjust versions as needed for your environment.
