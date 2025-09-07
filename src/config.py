import os
from pathlib import Path

# Default to Desktop path if env not set
DATA_PATH = Path(os.getenv("FRAUD_DATA", "/Users/vuz/Desktop/fraud.csv"))

# All artifacts go under project folder
ARTIFACTS = Path("artifacts_fraud_time_smote_cal")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

SLIDES = Path("slides_assets")
SLIDES.mkdir(parents=True, exist_ok=True)
