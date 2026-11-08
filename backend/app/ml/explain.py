from .model_registry import load_model
from .feature_engineering import engineer_features

def explain_record(record):
    artifact = load_model()
    if artifact is None:
        raise RuntimeError("No trained model available")
    try:
        import shap
        import pandas as pd
        pipeline = artifact["pipeline"]
        transformed = pipeline.named_steps["preprocess"].transform(engineer_features(record))
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        estimator = pipeline.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            explainer = shap.TreeExplainer(estimator)
        else:
            background = artifact.get("explain_background")
            if background is None:
                from .data_loader import load_data
                training_rows = load_data()
                background_rows = training_rows.drop(columns=["customerID", "Churn"]).sample(n=min(100, len(training_rows)), random_state=42)
                background = pipeline.named_steps["preprocess"].transform(engineer_features(background_rows))
            explainer = shap.LinearExplainer(estimator, background)
        values = explainer.shap_values(transformed)
        if isinstance(values, list): values = values[-1]
        if getattr(values, "ndim", 0) == 3: values = values[:, :, -1]
        row = values[0]
        expected = explainer.expected_value
        if hasattr(expected, "__len__"): expected = expected[-1]
        unit = "log-odds (base model)" if estimator.__class__.__name__ == "LogisticRegression" else "raw base-model output"
        return {"base_value": float(expected), "prediction_value": float(expected + row.sum()), "output_unit": unit, "features": sorted([{"feature": str(n), "value": float(v)} for n, v in zip(names, row)], key=lambda x: abs(x["value"]), reverse=True)[:10]}
    except ImportError as exc:
        raise RuntimeError("Install SHAP to enable explanations") from exc
