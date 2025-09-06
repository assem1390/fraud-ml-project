
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import set_config
set_config(transform_output="pandas")

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, roc_curve
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import sklearn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

from .config import DATA_PATH, ARTIFACTS
from .data_processing import build_preprocess

RANDOM_STATE = 42
TEST_FRACTION = 0.20
CV_SPLITS = 3
N_ITER = 30

def evaluate_probs(y_true, y_prob):
    roc = roc_auc_score(y_true, y_prob)
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    pr = auc(rec, prec)
    return float(roc), float(pr), (prec, rec, thr)

def pick_thresholds(y_true, y_prob, budget_rate=0.01, C_review=2.0, C_miss=100.0):
    # grid of thresholds from PR curve
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    thr = np.append(thr, 1.0)  # align to len=prec==rec

    # F1-opt
    f1 = (2*prec*rec) / np.clip(prec+rec, 1e-9, None)
    i_f1 = int(np.nanargmax(f1))
    thr_f1 = float(thr[i_f1])

    # Precision>=x
    def thr_for_precision(pmin):
        idx = np.where(prec >= pmin)[0]
        return float(thr[idx[0]]) if len(idx) else float(1.0)
    thr_p90 = thr_for_precision(0.90)
    thr_p95 = thr_for_precision(0.95)
    thr_p98 = thr_for_precision(0.98)

    # Budget: choose threshold so alerts ≈ budget_rate
    total = len(y_true)
    budget = int(round(total * budget_rate))
    order = np.argsort(-y_prob)
    thr_budget = float(y_prob[order[budget-1]]) if budget>0 and budget<=total else float(1.0)

    # Cost-based: minimize expected cost
    # cost = C_review*(TP+FP) + C_miss*FN
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

    return dict(thr_f1=thr_f1, thr_p90=thr_p90, thr_p95=thr_p95, thr_p98=thr_p98,
                thr_budget=thr_budget, thr_cost=thr_cost)

def cm_at_threshold(y_true, y_prob, thr):
    yhat = (y_prob >= thr).astype(int)
    TN = int(((y_true==0)&(yhat==0)).sum())
    FP = int(((y_true==0)&(yhat==1)).sum())
    FN = int(((y_true==1)&(yhat==0)).sum())
    TP = int(((y_true==1)&(yhat==1)).sum())
    return {"TN":TN,"FP":FP,"FN":FN,"TP":TP}

def main():
    df = pd.read_csv(DATA_PATH).sort_values("step").reset_index(drop=True)
    assert {"fraud","step"}.issubset(df.columns)
    y = df["fraud"].astype(int).values
    X = df.drop(columns=["fraud"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # time-based split
    cut = int(len(df) * (1 - TEST_FRACTION))
    X_train_full, y_train_full = X.iloc[:cut], y[:cut]
    X_test, y_test = X.iloc[cut:], y[cut:]

    preprocess = build_preprocess(cat_cols, num_cols)

    # Build base learners
    lr  = LogisticRegression(max_iter=2000, class_weight="balanced")
    rf  = RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced_subsample")
    lgb = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    # Ensemble (soft voting)
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("lgbm", lgb)],
        voting="soft"
    )

    # SMOTENC indices for categorical features (after ColumnTransformer, we apply SMOTE BEFORE model, so pass raw indices)
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    sampler = SMOTENC(categorical_features=cat_idx, k_neighbors=5, sampling_strategy=0.5, random_state=RANDOM_STATE)

    # Full pipeline (SMOTE happens before model; preprocess is inside model pipeline to avoid leakage)
    # We'll use an imblearn Pipeline so sampler fits only in training folds during CV.
    pipe = ImbPipeline(steps=[
        ("sampler", sampler),
        ("prep", preprocess),
        ("clf", ensemble)
    ])

    # Param grids (light search around known good ranges)
    param_dist = {
        "clf__lr__C": [0.2, 0.5, 1.0],
        "clf__lr__solver": ["lbfgs", "liblinear"],
        "clf__rf__n_estimators": [600, 800, 1000],
        "clf__rf__max_depth": [None, 20, 30],
        "clf__rf__min_samples_split": [2, 10, 50],
        "clf__rf__min_samples_leaf": [1, 2, 5],
        "clf__lgbm__n_estimators": [400, 800, 1200],
        "clf__lgbm__learning_rate": [0.03, 0.05, 0.08],
        "clf__lgbm__num_leaves": [63, 127, 255],
        "clf__lgbm__max_depth": [-1, 20],
        "clf__lgbm__min_child_samples": [20, 50, 100],
        "clf__lgbm__subsample": [0.7, 0.85, 1.0],
        "clf__lgbm__colsample_bytree": [0.7, 0.85, 1.0],
        "clf__lgbm__reg_alpha": [0.0, 0.1, 1.0],
        "clf__lgbm__reg_lambda": [0.0, 0.1, 1.0],
    }

    cv = TimeSeriesSplit(n_splits=CV_SPLITS)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring="average_precision",  # PR-AUC
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
        random_state=RANDOM_STATE
    )
    search.fit(X_train_full, y_train_full)
    best = search.best_estimator_

    # Fit best on full training and evaluate on holdout (uncalibrated)
    best.fit(X_train_full, y_train_full)
    p_uncal = best.predict_proba(X_test)[:,1]
    roc_uncal, pr_uncal, (prec_u, rec_u, thr_u) = evaluate_probs(y_test, p_uncal)

    # Calibration: split last 20% of train for isotonic
    n_train = len(X_train_full)
    cal_cut = int(n_train * 0.8)
    X_tr, y_tr = X_train_full.iloc[:cal_cut], y_train_full[:cal_cut]
    X_cal, y_cal = X_train_full.iloc[cal_cut:], y_train_full[cal_cut:]

    # Refit on X_tr then calibrate on X_cal (prefit)
    best.fit(X_tr, y_tr)
    ver = tuple(int(v) for v in sklearn.__version__.split(".")[:2])
    if ver >= (1,1):
        calibrated = CalibratedClassifierCV(estimator=best, method="isotonic", cv="prefit").fit(X_cal, y_cal)
    else:
        calibrated = CalibratedClassifierCV(base_estimator=best, method="isotonic", cv="prefit").fit(X_cal, y_cal)

    p_cal = calibrated.predict_proba(X_test)[:,1]
    roc_cal, pr_cal, (prec_c, rec_c, thr_c) = evaluate_probs(y_test, p_cal)

    # Curves (PR & ROC) for calibrated + uncal
    fpr_c, tpr_c, _ = roc_curve(y_test, p_cal)
    fpr_u, tpr_u, _ = roc_curve(y_test, p_uncal)

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

    # Thresholds + conf mats
    thrs_uncal = pick_thresholds(y_test, p_uncal)
    thrs_cal   = pick_thresholds(y_test, p_cal)

    cms = {
        "uncal@0.50": cm_at_threshold(y_test, p_uncal, 0.50),
        "uncal@thr_f1": cm_at_threshold(y_test, p_uncal, thrs_uncal["thr_f1"]),
        "uncal@p90": cm_at_threshold(y_test, p_uncal, thrs_uncal["thr_p90"]),
        "uncal@p95": cm_at_threshold(y_test, p_uncal, thrs_uncal["thr_p95"]),
        "uncal@p98": cm_at_threshold(y_test, p_uncal, thrs_uncal["thr_p98"]),
        "uncal@budget": cm_at_threshold(y_test, p_uncal, thrs_uncal["thr_budget"]),
        "uncal@cost": cm_at_threshold(y_test, p_uncal, thrs_uncal["thr_cost"]),
        "cal@0.50": cm_at_threshold(y_test, p_cal, 0.50),
        "cal@thr_f1": cm_at_threshold(y_test, p_cal, thrs_cal["thr_f1"]),
        "cal@p90": cm_at_threshold(y_test, p_cal, thrs_cal["thr_p90"]),
        "cal@p95": cm_at_threshold(y_test, p_cal, thrs_cal["thr_p95"]),
        "cal@p98": cm_at_threshold(y_test, p_cal, thrs_cal["thr_p98"]),
        "cal@budget": cm_at_threshold(y_test, p_cal, thrs_cal["thr_budget"]),
        "cal@cost": cm_at_threshold(y_test, p_cal, thrs_cal["thr_cost"]),
    }

    # Save models
    import joblib
    joblib.dump(best, ARTIFACTS/"fraud_ensemble_uncalibrated.joblib")
    joblib.dump(calibrated, ARTIFACTS/"fraud_ensemble_calibrated.joblib")

    # Daily alert projections (you can edit daily volume)
    typical_daily = 120000
    def alerts_for_thr(y_prob, thr):
        return int((y_prob >= thr).sum())
    alert_proj = {
        "typical_daily_txn": typical_daily,
        "cal": {
            "thr_f1": {"threshold": thrs_cal["thr_f1"], "alerts_eval": alerts_for_thr(p_cal, thrs_cal["thr_f1"]), "alerts_daily_proj": int(round(alerts_for_thr(p_cal, thrs_cal["thr_f1"])/len(y_test)*typical_daily))},
            "p90": {"threshold": thrs_cal["thr_p90"], "alerts_eval": alerts_for_thr(p_cal, thrs_cal["thr_p90"]), "alerts_daily_proj": int(round(alerts_for_thr(p_cal, thrs_cal["thr_p90"])/len(y_test)*typical_daily))},
            "p95": {"threshold": thrs_cal["thr_p95"], "alerts_eval": alerts_for_thr(p_cal, thrs_cal["thr_p95"]), "alerts_daily_proj": int(round(alerts_for_thr(p_cal, thrs_cal["thr_p95"])/len(y_test)*typical_daily))},
            "p98": {"threshold": thrs_cal["thr_p98"], "alerts_eval": alerts_for_thr(p_cal, thrs_cal["thr_p98"]), "alerts_daily_proj": int(round(alerts_for_thr(p_cal, thrs_cal["thr_p98"])/len(y_test)*typical_daily))},
            "budget": {"threshold": thrs_cal["thr_budget"], "alerts_eval": alerts_for_thr(p_cal, thrs_cal["thr_budget"]), "alerts_daily_proj": int(round(alerts_for_thr(p_cal, thrs_cal["thr_budget"])/len(y_test)*typical_daily))},
            "cost": {"threshold": thrs_cal["thr_cost"], "alerts_eval": alerts_for_thr(p_cal, thrs_cal["thr_cost"]), "alerts_daily_proj": int(round(alerts_for_thr(p_cal, thrs_cal["thr_cost"])/len(y_test)*typical_daily))}
        }
    }

    # Save summary.json
    summary = {
        "schema": {
            "categorical": cat_cols,
            "numeric": num_cols
        },
        "cv": {"splits": CV_SPLITS, "scoring": "average_precision", "n_iter": N_ITER},
        "metrics": {
            "uncal": {"roc_auc": roc_uncal, "pr_auc": pr_uncal},
            "cal":   {"roc_auc": roc_cal,   "pr_auc": pr_cal}
        },
        "thresholds_uncal": thrs_uncal,
        "thresholds_cal": thrs_cal,
        "confusion_matrices": cms,
        "alert_projections_daily": alert_proj
    }
    with open(ARTIFACTS/"summary.json","w") as f:
        json.dump(summary, f, indent=2)

    print("Saved artifacts to:", ARTIFACTS)

if __name__ == "__main__":
    main()
