"""Iris species classification using scikit-learn.

Steps:
1. Load Iris dataset and convert to DataFrame for easier manipulation.
2. Build a preprocessing + model pipeline with SimpleImputer and DecisionTreeClassifier.
3. Split data into train/test sets, fit the pipeline, and report accuracy, precision, recall.
4. Persist metrics and show sample predictions for inspection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    iris = datasets.load_iris()
    features = pd.DataFrame(iris.data, columns=iris.feature_names)
    labels = pd.Series(iris.target, name="species_id")
    # Attach target names for readability
    labels.index = features.index
    return features, labels, iris.target_names


def preprocess_targets(labels: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    return encoded, encoder


def build_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            (
                "clf",
                DecisionTreeClassifier(
                    criterion="entropy",
                    max_depth=4,
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray, encoder: LabelEncoder) -> dict:
    preds = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="macro", zero_division=0)
    recall = recall_score(y_test, preds, average="macro", zero_division=0)
    report = classification_report(
        y_test,
        preds,
        target_names=encoder.classes_,
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "classification_report": report,
        "sample_predictions": {
            int(idx): {
                "true": encoder.inverse_transform([true])[0],
                "pred": encoder.inverse_transform([pred])[0],
            }
            for idx, true, pred in zip(X_test.index[:5], y_test[:5], preds[:5])
        },
    }


def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    X, y, target_names = load_data()
    y_encoded, encoder = preprocess_targets(pd.Series(target_names[cls] for cls in y))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test, encoder)

    metrics_path = Path(__file__).resolve().parent.parent / "reports" / "iris_metrics.json"
    save_metrics(metrics, metrics_path)

    print("Iris Decision Tree Evaluation")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision (macro): {metrics['precision_macro']:.3f}")
    print(f"Recall (macro): {metrics['recall_macro']:.3f}")
    print("Sample predictions:")
    for idx, pred_info in metrics["sample_predictions"].items():
        print(f"  Row {idx}: true={pred_info['true']}, pred={pred_info['pred']}")
    print("Full classification report saved to", metrics_path)


if __name__ == "__main__":
    main()
