import os
from pathlib import Path
from typing import Dict

from flask import Flask, request, jsonify, render_template, abort
import joblib
import numpy as np

# ----------- Settings ----------
ROOT = Path(__file__).resolve().parents[1]  # repo root (../)
ART = ROOT / "artifacts"
MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "best_pipeline.joblib"


# Uncertain threshold for LinearSVC margins.
# If |margin| < TAU => "uncertain"
TAU = float(os.getenv("UNCERTAIN_TAU", "0.5"))

# Simple label map
LABELS: Dict[int, str] = {0: "fake", 1: "real"}

# Input size limits
MIN_CHARS = 10
MAX_CHARS = 10000

# ----------- App ----------
app = Flask(__name__, template_folder="templates")

# Load once at startup
pipe = joblib.load(MODEL_PATH)
MODEL_NAME = pipe.__class__.__name__

def combine_text(title: str, text: str) -> str:
    title = (title or "").strip()
    text = (text or "").strip()
    combined = (title + " " + text).strip()
    return combined

def predict_label_and_score(text_all: str):
    """
    For LinearSVC: use decision_function as a margin.
    Positive margin -> class 1 (real), negative -> class 0 (fake).
    """
    # Predict class
    pred = int(pipe.predict([text_all])[0])

    # Try to get a margin/score if available
    score = None
    if hasattr(pipe, "decision_function"):
        margin = float(pipe.decision_function([text_all])[0])
        # Convert margin to a "score-like" value in [0,1] using a simple squash.
        # This is *not* a calibrated probability, just a monotonic mapping.
        score = 1 / (1 + np.exp(-margin))

        # Uncertain band by absolute margin
        if abs(margin) < TAU:
            return "Real", score

    return LABELS.get(pred, str(pred)), score

# ----------- Routes ----------
@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": "best_pipeline.joblib", "uncertain_tau": TAU})

@app.post("/predict")
def predict():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        abort(400, description="Invalid JSON.")

    title = data.get("title", "")
    text = data.get("text", "")

    combined = combine_text(title, text)
    if len(combined) < MIN_CHARS:
        abort(400, description=f"Input too short. Provide at least {MIN_CHARS} characters.")
    if len(combined) > MAX_CHARS:
        abort(400, description=f"Input too long. Max {MAX_CHARS} characters.")

    label, score = predict_label_and_score(combined)
    return jsonify({
        "label": label,
        "score": None if score is None else round(float(score), 4),
        "len_chars": len(combined),
        "model": "LinearSVC_TFIDF",
        "uncertain_tau": TAU
    })

@app.get("/")
def index():
    return render_template("index.html", tau=TAU)

if __name__ == "__main__":
    # Dev server
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 50001)), debug=True)
