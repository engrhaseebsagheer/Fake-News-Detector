# train_baselines.py
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

import joblib

# =========================
# HARD-CODED SETTINGS
# =========================
RANDOM_STATE = 42
ENCODING = "utf-8"

# Resolve paths (adjust if your repo layout differs)
ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 2 else Path.cwd()
DATA_PATH = ROOT / "Data" / "Processed" / "NLP_Processed_Data.csv"
ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

# Columns expected in the cleaned CSV
TEXT_COL = "clean_text"
TITLE_COL = "clean_title"
LABEL_COL = "label"  # 0/1

# TF-IDF parameters (strong default for news text)
TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    max_features=50_000,
    min_df=2,
    max_df=0.9,
    lowercase=False,   # already lowercased during cleaning
    dtype=np.float32,  # memory-friendly
    sublinear_tf=True
)

# Small grids for quick tuning
LR_C_GRID = [0.5, 1.0, 2.0]
SVM_C_GRID = [0.5, 1.0, 2.0]
NB_ALPHA_GRID = [0.1, 0.5, 1.0]


# =========================
# Helpers
# =========================
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, encoding=ENCODING)
    for col in [TEXT_COL, TITLE_COL, LABEL_COL]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found. Got: {list(df.columns)[:20]}")
    return df

def build_text_column(df: pd.DataFrame) -> pd.Series:
    # Combine title + text (title often adds useful signal)
    return (df[TITLE_COL].fillna("").astype(str) + " " +
            df[TEXT_COL].fillna("").astype(str)).str.strip()

@dataclass
class EvalResult:
    name: str
    params: dict
    accuracy: float
    precision: float
    recall: float
    f1: float
    report: str

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str, params: dict) -> EvalResult:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    return EvalResult(
        name=name, params=params, accuracy=acc,
        precision=precision, recall=recall, f1=f1, report=report
    )

def fit_and_eval_pipeline(
    model_name: str,
    clf,
    x_train: List[str],
    y_train: np.ndarray,
    x_val: List[str],
    y_val: np.ndarray,
    tfidf_params: dict,
    param_grid: List[dict]
) -> Tuple[Pipeline, EvalResult]:
    """
    Manual grid-search on the validation set.
    Returns the best fitted pipeline and its EvalResult (on val).
    """
    best_pipeline = None
    best_result = None

    for params in param_grid:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", clf.__class__(**params))
        ])
        pipe.fit(x_train, y_train)
        preds = pipe.predict(x_val)
        res = evaluate(y_val, preds, model_name, params)

        if (best_result is None) or (res.f1 > best_result.f1):
            best_result = res
            best_pipeline = pipe

    return best_pipeline, best_result


# =========================
# Main
# =========================
def main():
    # 1) Load data
    df = load_data(DATA_PATH)

    # 2) Build single text field
    df["text_all"] = build_text_column(df)

    # 3) Split: 64/16/20 (train/val/test) via 80/20 then 20% of the 80%
    X = df["text_all"].tolist()
    y = df[LABEL_COL].to_numpy()

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.20, stratify=y_dev, random_state=RANDOM_STATE
    )

    print(f"[SPLIT] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 4) Define models + small grids
    models_and_grids = [
        ("LogReg",
         LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
         [{"C": c, "max_iter": 2000, "solver": "liblinear", "random_state": RANDOM_STATE} for c in LR_C_GRID]),
        ("LinearSVC",
         LinearSVC(random_state=RANDOM_STATE),
         [{"C": c, "random_state": RANDOM_STATE} for c in SVM_C_GRID]),
        ("CompNB",
         ComplementNB(),
         [{"alpha": a} for a in NB_ALPHA_GRID]),
    ]

    # 5) Train + validate; keep best by F1 on val
    best_name, best_pipe, best_val = None, None, None
    all_val_results = []

    for name, clf, grid in models_and_grids:
        pipe, res = fit_and_eval_pipeline(
            model_name=name,
            clf=clf,
            x_train=X_train, y_train=y_train,
            x_val=X_val, y_val=y_val,
            tfidf_params=TFIDF_PARAMS,
            param_grid=grid
        )
        all_val_results.append(res)
        print(f"\n[VAL] {name} | params={res.params}")
        print(f"[VAL] Acc={res.accuracy:.3f} Prec={res.precision:.3f} Rec={res.recall:.3f} F1={res.f1:.3f}")
        print(res.report)

        if (best_val is None) or (res.f1 > best_val.f1):
            best_name, best_pipe, best_val = name, pipe, res

    # 6) Final test with the best pipeline
    y_test_pred = best_pipe.predict(X_test)
    test_res = evaluate(y_test, y_test_pred, f"{best_name}_TEST", best_val.params)

    cm = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(cm, index=["True:0(fake)", "True:1(real)"],
                         columns=["Pred:0(fake)", "Pred:1(real)"])
    cm_path = ART / "confusion_matrix_test.csv"
    cm_df.to_csv(cm_path)

    print("\n[TEST] Best model:", best_name, "| params:", best_val.params)
    print(f"[TEST] Acc={test_res.accuracy:.3f} Prec={test_res.precision:.3f} Rec={test_res.recall:.3f} F1={test_res.f1:.3f}")
    print(test_res.report)
    print(f"[TEST] Confusion matrix saved to: {cm_path}")

    # 7) Save best pipeline + JSON-safe metrics
    best_pipe_path = ART / "best_pipeline.joblib"
    joblib.dump(best_pipe, best_pipe_path)

    # Make TF-IDF params JSON-safe (dtype is a NumPy type)
    tfidf_params_json = dict(TFIDF_PARAMS)
    tfidf_params_json["dtype"] = "float32"

    metrics = {
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "val_results": [
            {"name": r.name, "params": r.params, "accuracy": r.accuracy,
             "precision": r.precision, "recall": r.recall, "f1": r.f1}
            for r in all_val_results
        ],
        "best_val": {
            "name": best_val.name, "params": best_val.params,
            "accuracy": best_val.accuracy, "precision": best_val.precision,
            "recall": best_val.recall, "f1": best_val.f1
        },
        "test": {
            "name": test_res.name, "params": test_res.params,
            "accuracy": test_res.accuracy, "precision": test_res.precision,
            "recall": test_res.recall, "f1": test_res.f1
        },
        "tfidf_params": tfidf_params_json
    }
    with open(ART / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[SAVED] Best pipeline → {best_pipe_path}")
    print(f"[SAVED] Metrics → {ART / 'metrics.json'}")

if __name__ == "__main__":
    main()
