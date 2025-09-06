# Fraud Detection — End-to-End ML (Ensembles + Calibration)

This repo contains an end-to-end fraud detection pipeline:
- EDA and preprocessing
- SMOTE-in-CV + Target Encoding + Ensemble (LR + RF + LightGBM)
- Isotonic calibration
- Threshold pickers: F1-opt, Precision≥90/95/98, Cost-based, Alert-budget
- Feature importance (Permutation + LightGBM gain)
- Slide-ready assets (charts/tables) + pipeline diagrams
- Baseline model comparison

## Quickstart

```bash
# 1) Create a virtual env
python3 -m venv .venv
source .venv/bin/activate

# 2) Install requirements
pip install -r requirements.txt

Data path: set an environment variable (recommended):

export FRAUD_DATA=/Users/vuz/Desktop/fraud.csv


If unset, scripts default to the same path.

Run (in order)

EDA charts

python src/eda_charts.py


Train ensemble + calibration + summary.json + curves

python src/train_ensemble.py


Feature importance (permutation + LGBM gain)

python src/feature_importance.py


Slide assets (tables/plots)

python src/slide_assets.py


Baselines (LR / RF / LGBM)

python src/make_model_baselines.py


Pipeline diagram (Graphviz)

python src/pipeline_diagram.py


Use in production (scoring)

from src.scoring_module import load_model, score_batch, two_tier_decisions
model = load_model("artifacts_fraud_time_smote_cal/fraud_ensemble_calibrated.joblib")
# df is a pandas DataFrame with the raw schema
p = score_batch(model, df)  # returns probabilities np.array
decisions = two_tier_decisions(p, summary_path="artifacts_fraud_time_smote_cal/summary.json")

Threshold policy

Two-tier (from summary.json):

Auto-block: P≥0.98 (thresholds_cal.thr_p98)

Manual review: alert-budget (~1%) (thresholds_cal.thr_budget)

Other options: F1-opt, P≥0.90/0.95, Cost-based.

Artifacts

artifacts_fraud_time_smote_cal/summary.json

artifacts_fraud_time_smote_cal/pr_cal.png, roc_cal.png, pr_uncal.png, roc_uncal.png

feature_importance_permutation.csv, feature_importance_lgbm_gain.csv

slides_assets/threshold_metrics_calibrated.png, etc.