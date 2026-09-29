"""Offline inference with a trusted, bundled scikit-learn model."""
from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().with_name("intent_model.joblib")
NAMES = {"receive_benefit": "receive money or benefit", "make_payment": "make a payment", "share_credentials": "share credentials", "general_info": "general information"}
_model = None


def predict(text):
    global _model
    if not text.strip():
        return None
    if _model is None:
        # Load only the model shipped with this project; joblib files must be trusted.
        _model = joblib.load(MODEL_PATH)
    probabilities = _model.predict_proba([text])[0]
    best_index = probabilities.argmax()
    best = _model.classes_[best_index]
    coverage = _model["countvectorizer"].transform([text]).nnz
    reliable = coverage >= 4 and probabilities[best_index] >= 0.55
    return {"label": best if reliable else "uncertain", "display": NAMES[best] if reliable else "uncertain", "suggestion": NAMES[best], "coverage": coverage}
