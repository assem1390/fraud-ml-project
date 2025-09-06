import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.inspection import permutation_importance
import joblib
from .config import DATA_PATH, ARTIFACTS

def main():
    df = pd.read_csv(DATA_PATH).sort_values("step").reset_index(drop=True)
    y = df["fraud"].astype(int).values
    X = df.drop(columns=["fraud"])

    # load calibrated model
    model = joblib.load(ARTIFACTS/"fraud_ensemble_calibrated.joblib")

    # Permutation importance (Average Precision drop)
    from sklearn.metrics import average_precision_score
    def scorer(est, Xv, yv):
        p = est.predict_proba(Xv)[:,1]
        return average_precision_score(yv, p)

    result = permutation_importance(model, X.iloc[-len(y)//5:], y[-len(y)//5:], n_repeats=5, random_state=42, scoring=scorer)  # use last 20% slice for speed
    perm_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(ARTIFACTS.parent/"feature_importance_permutation.csv", index=False)

    # LightGBM gain (extract inner estimator)
    try:
        # calibrated.estimator_ is the prefit pipeline
        inner = model.estimator_.named_steps["clf"]
        lgbm = inner.named_estimators_["lgbm"]
        gain = lgbm.booster_.feature_importance(importance_type="gain")
        fi_lgbm = pd.DataFrame({"feature": X.columns, "gain": gain})
        fi_lgbm.to_csv(ARTIFACTS.parent/"feature_importance_lgbm_gain.csv", index=False)
    except Exception as e:
        print("Could not extract LGBM gain importance:", e)

    print("Saved permutation & LGBM gain CSVs next to artifacts.")

if __name__ == "__main__":
    main()
