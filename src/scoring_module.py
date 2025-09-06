import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

def load_model(path: str):
    return joblib.load(path)

def score_batch(model, df: pd.DataFrame) -> np.ndarray:
    """Return probabilities (fraud=1) for a raw batch DataFrame."""
    return model.predict_proba(df)[:,1]

def load_thresholds(summary_path: str):
    s = json.load(open(summary_path))
    return {
        "thr_f1": s["thresholds_cal"]["thr_f1"],
        "thr_p90": s["thresholds_cal"]["thr_p90"],
        "thr_p95": s["thresholds_cal"]["thr_p95"],
        "thr_p98": s["thresholds_cal"]["thr_p98"],
        "thr_budget": s["thresholds_cal"]["thr_budget"],
        "thr_cost": s["thresholds_cal"]["thr_cost"]
    }

def two_tier_decisions(prob, summary_path: str, mode="calibrated", thr_review_key="thr_budget", thr_auto_key="thr_p98"):
    """Return vector of decisions: auto_block / review / allow."""
    thrs = load_thresholds(summary_path)
    thr_review = thrs[thr_review_key]
    thr_auto   = thrs[thr_auto_key]
    out = []
    for p in prob:
        if p >= thr_auto:
            out.append("auto_block")
        elif p >= thr_review:
            out.append("review")
        else:
            out.append("allow")
    return out
