from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import MODEL_READY_FILE, RESULTS_DIR


TARGET = "distress_next_4q"


def load_model_data() -> pd.DataFrame:
    if not MODEL_READY_FILE.exists():
        raise FileNotFoundError("Run src/data/build_credit_panel.py first.")
    df = pd.read_csv(MODEL_READY_FILE, low_memory=False)
    df["obs_date"] = pd.to_datetime(df["obs_date"], errors="coerce")
    if "mkt_date" in df.columns:
        df["mkt_date"] = pd.to_datetime(df["mkt_date"], errors="coerce")
    df["obs_year"] = df["obs_date"].dt.year
    df = df[df["obs_date"].notna()].copy()
    df = df[df[TARGET].notna()].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].astype(int)
    return df


def add_log_shrout(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "shrout" in out.columns and "log_shrout" not in out.columns:
        out["log_shrout"] = np.log1p(pd.to_numeric(out["shrout"], errors="coerce").clip(lower=0))
    return out


def get_time_splits(df: pd.DataFrame):
    train_df = df[df["obs_year"].between(2000, 2016)].copy()
    valid_df = df[df["obs_year"].between(2017, 2020)].copy()
    test_df = df[df["obs_year"].between(2021, 2024)].copy()
    return train_df, valid_df, test_df


def get_winsor_bounds(frame: pd.DataFrame, cols: list[str], lower: float = 0.01, upper: float = 0.99):
    bounds = {}
    for c in cols:
        s = pd.to_numeric(frame[c], errors="coerce").dropna()
        if len(s) > 0:
            bounds[c] = (s.quantile(lower), s.quantile(upper))
    return bounds


def apply_winsor_bounds(frame: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    out = frame.copy()
    for c, (lo, hi) in bounds.items():
        out[c] = pd.to_numeric(out[c], errors="coerce").clip(lower=lo, upper=hi)
    return out


def evaluate_predictions(y_true, p):
    return {
        "auc": roc_auc_score(y_true, p),
        "pr_auc": average_precision_score(y_true, p),
        "brier": brier_score_loss(y_true, p),
        "avg_pred_prob": float(np.mean(p)),
        "realized_distress_rate": float(np.mean(y_true)),
    }


def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR
