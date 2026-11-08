from pathlib import Path
import joblib

ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "churn_model.joblib"

def load_model():
    if not ARTIFACT.exists():
        return None
    return joblib.load(ARTIFACT)
