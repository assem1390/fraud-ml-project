import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Prevent CPU/BLAS thread storms that can freeze laptops ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LIGHTGBM_NUM_THREADS", "1")

from sklearn import set_config, config_context
# IMPORTANT: keep default output (NOT pandas) to avoid SMOTENC OneHotEncoder sparse/pandas clash
set_config(transform_output="default")

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from joblib import Memory
from lightgbm import LGBMClassifier
import sklearn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

from .config import DATA_PATH, ARTIFACTS
from .data_processing import build_preprocess

# ---------------- CONFIG ----------------
RANDOM_STATE = 42
TEST_FRACTION = 0.20
CV_SPLITS = 3
N_ITER = 20              # was 30; reduce to be gentler on laptop
TUNE_ROW_SAMPLE = 0.50   # tune on last 50% of training rows; set to 1.0 to disable
USE_LIGHTER_SMOTE = True # lighter resampling during CV (still “inside CV only”)

# -------------- Helpers --------------
def evaluate_probs(y_true, y_prob):
    roc = roc_auc_score(y_true, y_prob)
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    pr = auc(rec, prec)
    return float(roc), float(pr), (prec, rec, thr)

def pick_thresholds(y_true, y_prob, budget_rate=0.01, C_review=2.0, C_miss=100.0):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    thr = np.append(thr, 1.0)  # align with prec/rec length

    # F1-opt
    f1 = (2*prec*rec) / np.clip(prec+rec, 1e-9, None)
    thr_f1 = float(thr[int(np.nanargmax(f1))])

    # Precision floors
    def thr_for_precision(pmin):
        idx = np.where(prec >= pmin)[0]
        return float(thr[idx[0]]) if len(idx) else float(1.0)
    thr_p90 = thr_for_precision(0.90)
    thr_p95 = thr_for_precision(0.95)
    thr_p98 = thr_for_precision(0.98)

    # Alert budget
    total = len(y_true)
    budget = int(round(total * budget_rate))
    order = np.argsort(-y_prob)
    thr_budget = float(y_prob[order[budget-1]]) if 0 < budget <= total else float(1.0)

    # Cost min: C_review*(TP+FP) + C_miss*FN
    thrs = np.unique(np.concatenate([thr, np.linspace(0,1,200)]))
    best_cost, best_thr = 1e18, 0.5
    for t in thrs:
        yhat = (y_prob >= t).astype(int)
        TP = int(((y_true==1)&(yhat==1)).sum())
        FP = int(((y_true==0)&(yhat==1)).sum())
        FN = int(((y_true==1)&(yhat==0)).sum())
        cost = C_review*(TP+FP) + C_miss*FN
        if cost < best_cost:
            best_cost, best_thr = cost, t
    thr_cost = float(best_thr)

    return dict(
        thr_f1=thr_f1, thr_p90=thr_p90, thr_p95=thr_p95, thr_p98=thr_p98,
        thr_budget=thr_budget, thr_cost=thr_cost
    )

def cm_at_threshold(y_true, y_prob, thr):
    yhat = (y_prob >= thr).astype(int)
    TN = int(((y_true==0)&(yhat==0)).sum())
    FP = int(((y_true==0)&(yhat==1)).sum())
    FN = int(((y_true==1)&(yhat==0)).sum())
    TP = int(((y_true==1)&(yhat==1)).sum())
    return {"TN":TN,"FP":FP,"FN":FN,"TP":TP}

# -------------- Main --------------
def main():
    # Load & order by time
    df = pd.read_csv(DATA_PATH).sort_values("step").reset_index(drop=True)
    assert {"fraud","step"}.issubset(df.columns)
    y = df["fraud"].astype(int).values
    X = df.drop(columns=["fraud"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Time-based split
    cut = int(len(df) * (1 - TEST_FRACTION))
    X_train_full, y_train_full = X.iloc[:cut], y[:cut]
    X_test, y_test = X.iloc[cut:], y[cut:]

    # Preprocess (cached)
    preprocess = build_preprocess(cat_cols, num_cols)
    cache_dir = ARTIFACTS / "cache"
    cache_dir.mkdir(exist_ok=True, parents=True)

    # Base learners (single-thread to avoid CPU oversubscription during CV)
    lr  = LogisticRegression(max_iter=2000, class_weight="balanced")
    rf  = RandomForestClassifier(n_jobs=1, random_state=RANDOM_STATE, class_weight="balanced_subsample")
    lgb = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1)

    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("lgbm", lgb)],
        voting="soft"
    )

    # SMOTENC (applied inside CV folds by imblearn Pipeline)
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    sampler = SMOTENC(
        categorical_features=cat_idx,
        k_neighbors=3 if USE_LIGHTER_SMOTE else 5,
        sampling_strategy=0.3 if USE_LIGHTER_SMOTE else 0.5,
        random_state=RANDOM_STATE
    )

    # Imblearn pipeline with caching
    pipe = ImbPipeline(
        steps=[("sampler", sampler), ("prep", preprocess), ("clf", ensemble)],
        memory=Memory(cache_dir, verbose=0)
    )

    # Param search space (trimmed to be friendlier to laptops)
    param_dist = {
        "clf__lr__C": [0.2, 0.5, 1.0],
        "clf__lr__solver": ["lbfgs", "liblinear"],
        "clf__rf__n_estimators": [600, 800],
        "clf__rf__max_depth": [None, 20, 30],
        "clf__rf__min_samples_split": [2, 10, 50],
        "clf__rf__min_samples_leaf": [1, 2, 5],
        "clf__lgbm__n_estimators": [400, 800],
        "clf__lgbm__learning_rate": [0.03, 0.05, 0.08],
        "clf__lgbm__num_leaves": [63, 127],
        "clf__lgbm__max_depth": [-1, 20],
        "clf__lgbm__min_child_samples": [20, 50, 100],
        "clf__lgbm__subsample": [0.7, 0.85],
        "clf__lgbm__colsample_bytree": [0.7, 0.85],
        "clf__lgbm__reg_alpha": [0.0, 1.0],
        "clf__lgbm__reg_lambda": [0.0, 1.0],
    }

    cv = TimeSeriesSplit(n_splits=CV_SPLITS)

    # Optional: subsample *most recent* training rows for search
    X_search, y_search = X_train_full, y_train_full
    if 0 < TUNE_ROW_SAMPLE < 1.0:
        n = int(len(X_train_full) * TUNE_ROW_SAMPLE)
        X_search = X_train_full.iloc[-n:]
        y_search = y_train_full[-n:]

    # ---- CRITICAL: ensure pandas-output is OFF during CV with SMOTENC ----
    with config_context(transform_output="default"):
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=N_ITER,
            scoring="average_precision",   # PR-AUC, best for imbalance
            cv=cv,
            n_jobs=-1,                     # OUTER parallelism only
            refit=True,
            verbose=1,
            random_state=RANDOM_STATE
        )
        search.fit(X_search, y_search)

    best = search.best_estimator_

    # Fit best on FULL training (still single-threaded learners)
    best.fit(X_train_full, y_train_full)
    p_uncal = best.predict_proba(X_test)[:,1]
    roc_uncal, pr_uncal, (prec_u, rec_u, thr_u) = evaluate_probs(y_test, p_uncal)

    # Calibration on final slice of training (prefit isotonic)
    n_train = len(X_train_full)
    cal_cut = int(n_train * 0.8)
    X_tr, y_tr = X_train_full.iloc[:cal_cut], y_train_full[:cal_cut]
    X_cal, y_cal = X_train_full.iloc[cal_cut:], y_train_full[cal_cut:]

    best.fit(X_tr, y_tr)
    ver = tuple(int(v) for v in sklearn.__version__.split(".")[:2])
    if ver >= (1,1):
        calibrated = CalibratedClassifierCV(estimator=best, method="isotonic", cv="prefit").fit(X_cal, y_cal)
    else:
        calibrated = CalibratedClassifierCV(base_estimator=best, method="isotonic", cv="prefit").fit(X_cal, y_cal)

    p_cal = calibrated.predict_proba(X_test)[:,1]
    roc_cal, pr_cal, (prec_c, rec_c, thr_c) = evaluate_probs(y_test, p_cal)

    # Curves
    fpr_c, tpr_c, _ = roc_curve(y_test, p_cal)
    fpr_u, tpr_u, _ = roc_curve(y_test, p_uncal)

    (ARTIFACTS).mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(5.5,5))
    plt.plot(rec_c, prec_c, label=f"Calibrated (AP={pr_cal:.3f})")
    plt.plot(rec_u, prec_u, label=f"Uncalibrated (AP={pr_uncal:.3f})", alpha=0.6)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(ARTIFACTS/"pr_cal.png", bbox_inches="tight"); plt.close()

    plt.figure(figsize=(5.5,5))
    plt.plot(fpr_c, tpr_c, label=f"Calibrated (ROC-AUC={roc_cal:.3f})")
    plt.plot(fpr_u, tpr_u, label=f"Uncalibrated (ROC-AUC={roc_uncal:.3f})", alpha=0.6)
    plt.plot([0,1],[0,1], "--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(ARTIFACTS/"roc_cal.png", bbox_inches="tight"); plt.close()

    # Thresholds + confusion matrices
    thrs_uncal = pick_thresholds(y_test, p_uncal)
    thrs_cal   = pick_thresholds(y_test, p_cal)

    def cm_dicts(y, p, thrs, prefix):
        return {
            f"{prefix}@0.50": cm_at_threshold(y, p, 0.50),
            f"{prefix}@thr_f1": cm_at_threshold(y, p, thrs["thr_f1"]),
            f"{prefix}@p90": cm_at_threshold(y, p, thrs["thr_p90"]),
            f"{prefix}@p95": cm_at_threshold(y, p, thrs["thr_p95"]),
            f"{prefix}@p98": cm_at_threshold(y, p, thrs["thr_p98"]),
            f"{prefix}@budget": cm_at_threshold(y, p, thrs["thr_budget"]),
            f"{prefix}@cost": cm_at_threshold(y, p, thrs["thr_cost"]),
        }
    cms = {}
    cms.update(cm_dicts(y_test, p_uncal, thrs_uncal, "uncal"))
    cms.update(cm_dicts(y_test, p_cal, thrs_cal, "cal"))

    # Save models
    import joblib
    joblib.dump(best, ARTIFACTS/"fraud_ensemble_uncalibrated.joblib")
    joblib.dump(calibrated, ARTIFACTS/"fraud_ensemble_calibrated.joblib")

    # Daily alert projections
    typical_daily = 120000
    def alerts_for_thr(y_prob, thr): return int((y_prob >= thr).sum())
    alert_proj = {
        "typical_daily_txn": typical_daily,
        "cal": {
            "thr_f1": {
                "threshold": thrs_cal["thr_f1"],
                "alerts_eval": alerts_for_thr(p_cal, thrs_cal["thr_f1"]),
                "alerts_daily_proj": int(round(alerts_for_thr(p_cal, thrs_cal["thr_f1"])/len(y_test)*typical_daily))
            },
            "p90": {
                "threshold": thrs_cal["thr_p90"],
                "alerts_eval": alerts_for_thr(p_cal, thrs_cal["thr_p90"]),
                "alerts_daily_proj": int(round(alerts_for_thr(p_cal, thrs_cal["thr_p90"])/len(y_test)*typical_daily))
            },
            "thr_p95": {
                "threshold": thrs_cal["thr_p95"],
                "alerts_eval": alerts_for_thr(p_cal, thrs_cal["thr_p95"]),
                "alerts_daily_proj": int(round(alerts_for_thr(p_cal, thrs_cal["thr_p95"])/len(y_test)*typical_daily))
            },
            "thr_p98": {
                "threshold": thrs_cal["thr_p98"],
                "alerts_eval": alerts_for_thr(p_cal, thrs_cal["thr_p98"]),
                "alerts_daily_proj": int(round(alerts_for_thr(p_cal, thrs_cal["thr_p98"])/len(y_test)*typical_daily))
            },
            "thr_budget": {
                "threshold": thrs_cal["thr_budget"],
                "alerts_eval": alerts_for_thr(p_cal, thrs_cal["thr_budget"]),
                "alerts_daily_proj": int(round(alerts_for_thr(p_cal, thrs_cal["thr_budget"])/len(y_test)*typical_daily))
            },
            "thr_cost": {
                "threshold": thrs_cal["thr_cost"],
                "alerts_eval": alerts_for_thr(p_cal, thrs_cal["thr_cost"]),
                "alerts_daily_proj": int(round(alerts_for_thr(p_cal, thrs_cal["thr_cost"])/len(y_test)*typical_daily))
            }
        }
    }

    # Save summary.json
    summary = {
        "schema": {"categorical": cat_cols, "numeric": num_cols},
        "cv": {"splits": CV_SPLITS, "scoring": "average_precision", "n_iter": N_ITER, "row_subsample": TUNE_ROW_SAMPLE},
        "metrics": {"uncal": {"roc_auc": roc_uncal, "pr_auc": pr_uncal},
                    "cal":   {"roc_auc": roc_cal,   "pr_auc": pr_cal}},
        "thresholds_uncal": thrs_uncal,
        "thresholds_cal": thrs_cal,
        "confusion_matrices": cms,
        "alert_projections_daily": alert_proj
    }
    with open(ARTIFACTS/"summary.json","w") as f:
        json.dump(summary, f, indent=2)

    print("Saved artifacts to:", ARTIFACTS.resolve())

if __name__ == "__main__":
    main()
