"""Inference pipeline: raw text -> clean -> TF-IDF -> best model -> category + confidence."""
import json
import os
from functools import lru_cache

import joblib
import numpy as np

from .preprocess import clean_text

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


@lru_cache(maxsize=1)
def _load():
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODELS_DIR, "model_info.json")) as f:
        info = json.load(f)
    return vectorizer, model, encoder, info


def model_info() -> dict:
    return _load()[3]


def _probabilities(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    scores = model.decision_function(X)  # e.g. LinearSVC: softmax over margins
    scores = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(scores)
    return e / e.sum(axis=1, keepdims=True)


def predict(text: str) -> dict:
    """Classify one article. Returns {'category', 'confidence', 'probabilities'}."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string.")
    vectorizer, model, encoder, _ = _load()
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Text has no usable words after cleaning.")
    probs = _probabilities(model, vectorizer.transform([cleaned]))[0]
    best = int(np.argmax(probs))
    return {
        "category": str(encoder.classes_[best]),
        "confidence": float(probs[best]),
        "probabilities": {str(c): float(p) for c, p in zip(encoder.classes_, probs)},
    }


if __name__ == "__main__":
    import sys

    print(json.dumps(predict(" ".join(sys.argv[1:]) or "Stocks rally as oil prices fall"), indent=2))
