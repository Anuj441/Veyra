from .model_registry import load_model
from .feature_engineering import engineer_features

def predict_records(records, high_risk_threshold=0.65):
    artifact = load_model()
    if artifact is None:
        raise RuntimeError("No trained model available. Run python -m app.ml.train after adding the dataset.")
    calibrated = bool(artifact.get("calibration", {}).get("applied"))
    pipeline = artifact["calibrated_pipeline"] if calibrated else artifact["pipeline"]
    probabilities = pipeline.predict_proba(engineer_features(records))[:, 1]
    medium_threshold = min(0.35, high_risk_threshold)
    return [{"probability": float(p), "probability_calibrated": calibrated, "risk": "HIGH" if p >= high_risk_threshold else "MEDIUM" if p >= medium_threshold else "LOW"} for p in probabilities]
