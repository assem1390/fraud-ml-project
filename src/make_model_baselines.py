import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import sklearn
from sklearn import set_config
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from category_encoders import TargetEncoder

# CRITICAL: emit pandas DataFrames
set_config(transform_output="pandas")

CSV_PATH = "/Users/vuz/Desktop/fraud.csv"
RANDOM_STATE = 42
TEST_FRACTION = 0.20
CALIBRATION_FRACTION = 0.20
CV_SPLITS = 3
N_ITER = 15

ARTIFACT_DIR = Path("./artifacts_model_baselines")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH).sort_values("step").reset_index(drop=True)
y = df["fraud"].astype(int)
X = df.drop(columns=["fraud"])

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

cut = int(len(df) * (1 - TEST_FRACTION))
X_train_full, y_train_full = X.iloc[:cut], y.iloc[:cut]
X_test, y_test = X.iloc[cut:], y.iloc[cut:]

cat_pipe = SkPipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("te", TargetEncoder(cols=None, smoothing=0.2, min_samples_leaf=200, return_df=True))
])

num_pipe = SkPipeline(steps=[
    ("impute", SimpleImputer(strategy="median"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

lr = LogisticRegression(max_iter=2000, class_weight="balanced")
rf = RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced_subsample")
lgbm = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1)

models = {
    "LogisticRegression": {
        "est": lr,
        "param_dist": {"clf__C": [0.2,0.5,1.0,2.0,5.0], "clf__penalty": ["l2"], "clf__solver": ["lbfgs","liblinear"]}
    },
    "RandomForest": {
        "est": rf,
        "param_dist": {"clf__n_estimators":[300,600,900], "clf__max_depth":[None,12,20,30], "clf__min_samples_split":[2,10,50], "clf__min_samples_leaf":[1,2,5]}
    },
    "LightGBM": {
        "est": lgbm,
        "param_dist": {"clf__n_estimators":[400,800,1200], "clf__learning_rate":[0.03,0.05,0.08], "clf__num_leaves":[63,127,255],
                       "clf__max_depth":[-1,12,20], "clf__min_child_samples":[20,50,100], "clf__subsample":[0.7,0.85,1.0],
                       "clf__colsample_bytree":[0.7,0.85,1.0], "clf__reg_alpha":[0.0,0.1,1.0], "clf__reg_lambda":[0.0,0.1,1.0]}
    }
}

def evaluate_probs(y_true, y_prob):
    roc = roc_auc_score(y_true, y_prob)
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    pr = auc(rec, prec)
    return float(roc), float(pr)

def calibrate_prefit(fitted_estimator, X_cal, y_cal, method="isotonic"):
    ver = tuple(int(v) for v in sklearn.__version__.split(".")[:2])
    if ver >= (1, 1):
        calib = CalibratedClassifierCV(estimator=fitted_estimator, method=method, cv="prefit")
    else:
        calib = CalibratedClassifierCV(base_estimator=fitted_estimator, method=method, cv="prefit")
    return calib.fit(X_cal, y_cal)

rows = []
cv = TimeSeriesSplit(n_splits=CV_SPLITS)

n_train = len(X_train_full)
cal_cut = int((1 - CALIBRATION_FRACTION) * n_train)
X_tr, y_tr = X_train_full.iloc[:cal_cut], y_train_full.iloc[:cal_cut]
X_cal, y_cal = X_train_full.iloc[cal_cut:], y_train_full.iloc[cal_cut:]

for name, cfg in models.items():
    print(f"[{name}] tuning…")
    pipe = SkPipeline(steps=[("prep", preprocess), ("clf", cfg["est"])])
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=cfg["param_dist"],
        n_iter=N_ITER,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
        random_state=RANDOM_STATE
    )
    search.fit(X_train_full, y_train_full)
    best = search.best_estimator_
    best.fit(X_train_full, y_train_full)
    proba_uncal = best.predict_proba(X_test)[:, 1]
    roc_uncal, pr_uncal = evaluate_probs(y_test, proba_uncal)

    print(f"[{name}] calibrating…")
    best.fit(X_tr, y_tr)
    calibrated = calibrate_prefit(best, X_cal, y_cal, method="isotonic")
    proba_cal = calibrated.predict_proba(X_test)[:, 1]
    roc_cal, pr_cal = evaluate_probs(y_test, proba_cal)

    rows.append({
        "Model": name,
        "ROC_AUC_uncal": roc_uncal,
        "PR_AUC_uncal": pr_uncal,
        "ROC_AUC_cal": roc_cal,
        "PR_AUC_cal": pr_cal,
        "best_params": search.best_params_
    })

res_df = pd.DataFrame(rows)
csv_path = ARTIFACT_DIR / "model_baseline_comparison.csv"
res_df.to_csv(csv_path, index=False)
print("[DONE] ->", csv_path.resolve())

plt.figure(figsize=(7,4))
order = res_df.sort_values("PR_AUC_cal", ascending=False)
plt.bar(order["Model"], order["PR_AUC_cal"])
for i,v in enumerate(order["PR_AUC_cal"].values):
    plt.text(i, v+0.005, f"{v:.3f}", ha="center")
plt.title("Model Comparison (PR-AUC, calibrated)")
plt.ylabel("PR-AUC"); plt.ylim(0,1); plt.xticks(rotation=20, ha="right")
png_path = ARTIFACT_DIR / "model_baseline_prauc.png"
plt.tight_layout(); plt.savefig(png_path, bbox_inches="tight"); plt.close()
print("[DONE] ->", png_path.resolve())
