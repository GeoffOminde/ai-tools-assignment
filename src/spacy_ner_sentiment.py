"""Named Entity Recognition and rule-based sentiment analysis on Amazon reviews using spaCy.

Usage:
    python spacy_ner_sentiment.py --input data/amazon_reviews_sample.jsonl

The script loads reviews, extracts product/brand entities via spaCy's NER, and applies
simple rule-based sentiment using review ratings and keyword heuristics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import spacy


SENTIMENT_KEYWORDS = {
    "positive": {"love", "great", "top-notch", "unmatched", "worth"},
    "negative": {"disappointed", "unresponsive", "buffer", "drains", "faster"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="spaCy NER and sentiment on Amazon reviews")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "amazon_reviews_sample.jsonl",
        help="Path to JSONL file with reviews",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="en_core_web_sm",
        help="spaCy language model to load",
    )
    return parser.parse_args()


def load_reviews(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def analyze_sentiment(text: str, rating: int) -> str:
    text_lower = text.lower()
    score = 0
    for word in SENTIMENT_KEYWORDS["positive"]:
        if word in text_lower:
            score += 1
    for word in SENTIMENT_KEYWORDS["negative"]:
        if word in text_lower:
            score -= 1
    if rating >= 4:
        score += 1
    elif rating <= 2:
        score -= 1
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def main() -> None:
    args = parse_args()
    try:
        nlp = spacy.load(args.model)
    except OSError:
        print(
            f"Model '{args.model}' not found. Using a blank English pipeline without pretrained NER."
        )
        print("Install with: python -m spacy download en_core_web_sm")
        nlp = spacy.blank("en")
        # Ensure the pipeline has an empty NER component to avoid attribute errors.
        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner")
            # Basic labels to recognize generic entities. This will not produce useful results
            # but keeps the output structure consistent.
            for label in ["PRODUCT", "ORG", "GPE", "PERSON", "WORK_OF_ART"]:
                ner.add_label(label)

    results = []
    for review in load_reviews(args.input):
        doc = nlp(review["review_text"])
        entities = [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
            if ent.label_ in {"PRODUCT", "ORG", "GPE", "PERSON", "WORK_OF_ART"}
        ]
        sentiment = analyze_sentiment(review["review_text"], review.get("rating", 0))
        results.append(
            {
                "review_id": review["review_id"],
                "product_name": review.get("product_name"),
                "brand": review.get("brand"),
                "entities": entities,
                "sentiment": sentiment,
            }
        )

    output_dir = Path(__file__).resolve().parent.parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "spacy_ner_sentiment.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("spaCy NER & Sentiment Analysis")
    for item in results:
        print(
            f"Review {item['review_id']}: product={item['product_name']} sentiment={item['sentiment']}"
        )
        print("  Entities:")
        for ent in item["entities"]:
            print(f"    - {ent['text']} ({ent['label']})")


if __name__ == "__main__":
    main()
