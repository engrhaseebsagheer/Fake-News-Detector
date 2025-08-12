import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# =========================
# HARD-CODED SETTINGS
# =========================
ENCODING = "utf-8"            # change if your CSV uses another encoding
USE_LEMMATIZE = False         # set True to enable lemmatization
RANDOM_STATE = 42             # for reproducible splits
TRAIN_VAL_TEST = (0.8, 0.1, 0.1)  # train/val/test ratios (must sum to 1.0)

# Resolve project root relative to this script (fallback: cwd)
_THIS = Path(__file__).resolve()
_PARENTS = _THIS.parents
ROOT = _PARENTS[2] if len(_PARENTS) >= 3 else _THIS.parent

# Paths
IN_PATH = ROOT / "Data" / "Processed" / "EDA_Processed_Data.csv"
OUT_CLEAN = ROOT / "Data" / "Processed" / "NLP_Processed_Data.csv"
ART = ROOT / "artifacts"

# =========================
# Config
# =========================
@dataclass
class CleanConfig:
    lowercase: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    normalize_spaces: bool = True
    lemmatize: bool = USE_LEMMATIZE  # driven by HARD-CODED flag

# Expected columns
TEXT_COL = "text"
TITLE_COL = "title"
LABEL_COL = "label"  # required for stratified split

# =========================
# Regex helpers
# =========================
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
MULTISPACE_RE = re.compile(r"\s+")

def remove_urls(text: str) -> str:
    return URL_RE.sub("", text)

def remove_emails(text: str) -> str:
    return EMAIL_RE.sub("", text)

def normalize_spaces(text: str) -> str:
    # collapse multiple spaces/newlines/tabs into a single space and strip
    return MULTISPACE_RE.sub(" ", text).strip()

# =========================
# Optional lemmatization (NLTK)
# =========================
try:
    import nltk
    from nltk import pos_tag, word_tokenize
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
    _HAS_NLTK = True
    _POS_MAP = {'J': wn.ADJ, 'V': wn.VERB, 'N': wn.NOUN, 'R': wn.ADV}
except Exception:
    _HAS_NLTK = False
    _POS_MAP = {}

def _pos_to_wn(tag: str):
    if not _HAS_NLTK:
        return None
    return _POS_MAP.get(tag[0], wn.NOUN)

def safe_lemmatize(text: str) -> str:
    if not _HAS_NLTK or not USE_LEMMATIZE:
        return text
    # If you see resource errors once, open a Python shell and run:
    # import nltk
    # nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(tok, _pos_to_wn(tag) or wn.NOUN) for tok, tag in tags]
    return " ".join(lemmas)

# =========================
# Core cleaning
# =========================
def clean_text_once(text: Optional[str], cfg: CleanConfig) -> str:
    if text is None:
        return ""
    out = text
    if cfg.lowercase:
        out = out.lower()
    if cfg.remove_urls:
        out = remove_urls(out)
    if cfg.remove_emails:
        out = remove_emails(out)
    if cfg.normalize_spaces:
        out = normalize_spaces(out)
    if cfg.lemmatize:
        out = safe_lemmatize(out)
    return out

def apply_cleaning(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    # text
    if TEXT_COL not in df.columns:
        raise ValueError(f"Expected column '{TEXT_COL}' not found. Columns are: {list(df.columns)[:20]}")
    df[TEXT_COL] = df[TEXT_COL].fillna("")
    df["clean_text"] = df[TEXT_COL].astype(str).apply(lambda s: clean_text_once(s, cfg))

    # title (optional)
    if TITLE_COL in df.columns:
        df[TITLE_COL] = df[TITLE_COL].fillna("")
        df["clean_title"] = df[TITLE_COL].astype(str).apply(lambda s: clean_text_once(s, cfg))
    else:
        df["clean_title"] = ""

    return df

# =========================
# IO Helpers
# =========================
def read_input_csv(in_path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")
    return pd.read_csv(in_path, encoding=encoding)

def save_labels(path: Path, y: np.ndarray) -> None:
    pd.Series(y, name=LABEL_COL).to_csv(path, index=False)

# =========================
# Split helpers
# =========================
def stratified_splits(
    df: pd.DataFrame,
    label_col: str,
    train_val_test=(0.8, 0.1, 0.1),
    random_state=42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    p_train, p_val, p_test = train_val_test
    if not np.isclose(p_train + p_val + p_test, 1.0):
        raise ValueError("TRAIN_VAL_TEST ratios must sum to 1.0")

    # First split: train vs temp (val+test)
    df_train, df_temp = train_test_split(
        df, test_size=(1.0 - p_train), stratify=df[label_col], random_state=random_state
    )
    # Second split: val vs test from temp
    relative_test = p_test / (p_val + p_test) if (p_val + p_test) > 0 else 0.5
    df_val, df_test = train_test_split(
        df_temp, test_size=relative_test, stratify=df_temp[label_col], random_state=random_state
    )
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

# =========================
# Main
# =========================
def main():
    # Load raw (already EDA-processed) CSV
    df = read_input_csv(IN_PATH, encoding=ENCODING)

    # Check label exists for split
    if LABEL_COL not in df.columns:
        raise ValueError(f"Expected column '{LABEL_COL}' not found. Columns are: {list(df.columns)[:20]}")

    # Phase 2: Cleaning
    cfg = CleanConfig(lemmatize=USE_LEMMATIZE)
    df = apply_cleaning(df, cfg)

    # Save cleaned CSV
    OUT_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CLEAN, index=False)
    print(f"[CLEAN] Saved cleaned dataset to: {OUT_CLEAN.resolve()}")
    print(f"[CLEAN] Lemmatization: {'ON' if USE_LEMMATIZE else 'OFF'}")

    # Build combined text field
    df["text_all"] = (df["clean_title"].astype(str) + " " + df["clean_text"].astype(str)).str.strip()

    # Phase 3: Split BEFORE vectorization (avoid leakage)
    df_train, df_val, df_test = stratified_splits(df, LABEL_COL, TRAIN_VAL_TEST, RANDOM_STATE)
    print(f"[SPLIT] train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    # Prepare texts and labels
    Xtr_texts = df_train["text_all"].fillna("").astype(str).tolist()
    Xv_texts  = df_val["text_all"].fillna("").astype(str).tolist()
    Xte_texts = df_test["text_all"].fillna("").astype(str).tolist()

    y_train = df_train[LABEL_COL].to_numpy()
    y_val   = df_val[LABEL_COL].to_numpy()
    y_test  = df_test[LABEL_COL].to_numpy()

    # TF-IDF (fit on train only)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack, save_npz
    import joblib

    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50_000,
        min_df=2,
        max_df=0.9,
        lowercase=False,      # already lowercased in cleaning
        dtype=np.float32
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.9,
        lowercase=False,
        dtype=np.float32
    )

    Xtr_word = word_vec.fit_transform(Xtr_texts)
    Xv_word  = word_vec.transform(Xv_texts)
    Xte_word = word_vec.transform(Xte_texts)

    Xtr_char = char_vec.fit_transform(Xtr_texts)
    Xv_char  = char_vec.transform(Xv_texts)
    Xte_char = char_vec.transform(Xte_texts)

    Xtr = hstack([Xtr_word, Xtr_char], format="csr")
    Xv  = hstack([Xv_word, Xv_char], format="csr")
    Xte = hstack([Xte_word, Xte_char], format="csr")

    # Save artifacts
    ART.mkdir(parents=True, exist_ok=True)

    # Vectorizers
    joblib.dump(word_vec, ART / "tfidf_word.pkl")
    joblib.dump(char_vec, ART / "tfidf_char.pkl")

    # Matrices
    save_npz(ART / "X_train_word.npz", Xtr_word)
    save_npz(ART / "X_val_word.npz",   Xv_word)
    save_npz(ART / "X_test_word.npz",  Xte_word)

    save_npz(ART / "X_train_char.npz", Xtr_char)
    save_npz(ART / "X_val_char.npz",   Xv_char)
    save_npz(ART / "X_test_char.npz",  Xte_char)

    save_npz(ART / "X_train_combined.npz", Xtr)
    save_npz(ART / "X_val_combined.npz",   Xv)
    save_npz(ART / "X_test_combined.npz",  Xte)

    # Labels
    save_labels(ART / "y_train.csv", y_train)
    save_labels(ART / "y_val.csv",   y_val)
    save_labels(ART / "y_test.csv",  y_test)

    # Meta
    meta = {
        "split": {"train": len(df_train), "val": len(df_val), "test": len(df_test)},
        "y_dist": {
            "train": df_train[LABEL_COL].value_counts().to_dict(),
            "val":   df_val[LABEL_COL].value_counts().to_dict(),
            "test":  df_test[LABEL_COL].value_counts().to_dict(),
        },
        "shapes": {
            "X_train_word": list(Xtr_word.shape),
            "X_val_word":   list(Xv_word.shape),
            "X_test_word":  list(Xte_word.shape),
            "X_train_char": list(Xtr_char.shape),
            "X_val_char":   list(Xv_char.shape),
            "X_test_char":  list(Xte_char.shape),
            "X_train_combined": list(Xtr.shape),
            "X_val_combined":   list(Xv.shape),
            "X_test_combined":  list(Xte.shape),
        },
        "word_vec": {"ngram_range": (1, 2), "max_features": 50_000, "min_df": 2, "max_df": 0.9, "lowercase": False, "dtype": "float32"},
        "char_vec": {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "max_df": 0.9, "lowercase": False, "dtype": "float32"},
        "cleaning": {
            "lowercase": True, "remove_urls": True, "remove_emails": True,
            "normalize_spaces": True, "lemmatize": USE_LEMMATIZE
        },
        "paths": {
            "in": str(IN_PATH.resolve()),
            "out_clean": str(OUT_CLEAN.resolve()),
            "artifacts_dir": str(ART.resolve())
        },
        "random_state": RANDOM_STATE
    }
    with open(ART / "tfidf_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Logs
    print("\n[TF-IDF] Artifacts saved to:", ART.resolve())
    for k, v in meta["shapes"].items():
        print(f" - {k}: {tuple(v)}")
    print("Next phase: train a model using the train matrices, validate on val, and report metrics on test.")

if __name__ == "__main__":
    main()
