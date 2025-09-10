from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import joblib, numpy as np, os
from pathlib import Path
from typing import Dict, Optional
from pydantic import BaseModel


# ----------- Settings ----------
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "best_pipeline.joblib"

# Uncertain threshold for LinearSVC margins.
TAU = float(os.getenv("UNCERTAIN_TAU", "0.5"))

# Simple label map
LABELS: Dict[int, str] = {0: "fake", 1: "real"}

# Input size limits
MIN_CHARS = 10
MAX_CHARS = 10000

# ----------- App ----------
app = FastAPI(title="Fake News Detector", version="1.0")

# Load model once
pipe = joblib.load(MODEL_PATH)
MODEL_NAME = pipe.__class__.__name__

# Templates
templates = Jinja2Templates(directory="templates")


# ----------- Helper Functions ----------
def combine_text(title: str, text: str) -> str:
    title = (title or "").strip()
    text = (text or "").strip()
    combined = (title + " " + text).strip()
    return combined


def predict_label_and_score(text_all: str):
    pred = int(pipe.predict([text_all])[0])

    score = None
    if hasattr(pipe, "decision_function"):
        margin = float(pipe.decision_function([text_all])[0])
        score = 1 / (1 + np.exp(-margin))
        if abs(margin) < TAU:
            return "Real", score

    return LABELS.get(pred, str(pred)), score


# ----------- Request Models ----------
class PredictRequest(BaseModel):
    title: Optional[str] = ""
    text: Optional[str] = ""


# ----------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok", "model": "best_pipeline.joblib", "uncertain_tau": TAU}


@app.post("/predict")
def predict(req: PredictRequest):
    combined = combine_text(req.title, req.text)

    if len(combined) < MIN_CHARS:
        raise HTTPException(status_code=400, detail=f"Input too short. Provide at least {MIN_CHARS} characters.")
    if len(combined) > MAX_CHARS:
        raise HTTPException(status_code=400, detail=f"Input too long. Max {MAX_CHARS} characters.")

    label, score = predict_label_and_score(combined)
    return {
        "label": label,
        "score": None if score is None else round(float(score), 4),
        "len_chars": len(combined),
        "model": "LinearSVC_TFIDF",
        "uncertain_tau": TAU
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "tau": TAU})


# ----------- Entry Point ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 50001)), reload=True)
