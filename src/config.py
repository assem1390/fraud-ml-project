import os
from pathlib import Path

# Data path (env override allowed)
DATA_PATH = Path(os.getenv("FRAUD_DATA", "/fraud.csv"))

# Main artifacts dir
ARTIFACTS = Path("artifacts_fraud_time_smote_cal")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Slide assets dir
SLIDES = Path("slides_assets")
SLIDES.mkdir(parents=True, exist_ok=True)
