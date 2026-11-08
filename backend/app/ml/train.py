from datetime import datetime, timezone
import json
from pathlib import Path
import joblib
from packaging.version import Version
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from .data_loader import load_data
from .evaluate import evaluate
from .feature_engineering import split_features_target
from .preprocessing import make_preprocessor

ROOT = Path(__file__).resolve().parents[2]

def train_models():
    df = load_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", C=.5, solver="liblinear", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=350, min_samples_leaf=4, class_weight="balanced_subsample", max_features="sqrt", n_jobs=1, random_state=42),
    }
    skipped_models = {}
    try:
        from xgboost import XGBClassifier
        import xgboost
        if Version(xgboost.__version__) < Version("2.1.4"):
            raise ImportError(f"XGBoost {xgboost.__version__} is incompatible with scikit-learn 1.6; install xgboost>=2.1.4")
        candidates["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=.04, subsample=.85, colsample_bytree=.85, eval_metric="logloss", random_state=42, n_jobs=1)
    except ImportError as exc:
        skipped_models["XGBoost"] = str(exc)
    try:
        from lightgbm import LGBMClassifier
        candidates["LightGBM"] = LGBMClassifier(n_estimators=300, learning_rate=.04, class_weight="balanced", verbosity=-1, random_state=42, n_jobs=1)
    except ImportError as exc:
        skipped_models["LightGBM"] = str(exc)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fitted, scores, comparisons = {}, {}, {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocess", make_preprocessor(X_train)), ("model", estimator)])
        # Limit outer parallelism and each estimator to one thread to prevent nested
        # all-core workers from exhausting developer machines during CV.
        scoring = {"accuracy":"accuracy", "precision":"precision", "recall":"recall", "f1":"f1", "roc_auc":"roc_auc", "pr_auc":"average_precision", "brier_score":"neg_brier_score"}
        cv_result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=2, error_score="raise")
        scores[name] = {f"cv_{metric}": float(-cv_result[f"test_{metric}"].mean() if metric == "brier_score" else cv_result[f"test_{metric}"].mean()) for metric in scoring}
        pipeline.fit(X_train, y_train)
        comparisons[name] = {"cv_metrics": scores[name], "test_metrics": evaluate(pipeline, X_test, y_test)}
        fitted[name] = estimator
    # Select by cross-validated PR-AUC, then recall; test results remain untouched until selection.
    selected = max(scores, key=lambda n: (scores[n]["cv_pr_auc"], scores[n]["cv_recall"]))
    model = Pipeline([("preprocess", make_preprocessor(X_train)), ("model", fitted[selected])])
    model.fit(X_train, y_train)
    uncalibrated_test_metrics = evaluate(model, X_test, y_test)
    calibration_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=24)
    calibration_candidate = CalibratedClassifierCV(estimator=clone(model), method="sigmoid", cv=3, ensemble=False)
    calibration_cv_result = cross_validate(calibration_candidate, X_train, y_train, cv=calibration_cv, scoring="neg_brier_score", n_jobs=1, error_score="raise")
    cv_brier_before = scores[selected]["cv_brier_score"]
    cv_brier_after = float(-calibration_cv_result["test_score"].mean())
    calibration_applied = cv_brier_after < cv_brier_before
    calibrated_model = CalibratedClassifierCV(estimator=clone(model), method="sigmoid", cv=5, ensemble=False)
    calibrated_model.fit(X_train, y_train)
    calibrated_test_metrics = evaluate(calibrated_model, X_test, y_test)
    test_metrics = calibrated_test_metrics if calibration_applied else uncalibrated_test_metrics
    # Reliability points must describe the same pipeline that production serves.
    served_pipeline = calibrated_model if calibration_applied else model
    probabilities = served_pipeline.predict_proba(X_test)[:, 1]
    fraction_positive, mean_predicted = calibration_curve(y_test, probabilities, n_bins=10, strategy="quantile")
    calibration = {
        "method": "sigmoid" if calibration_applied else None,
        "candidate_method": "sigmoid",
        "applied": calibration_applied,
        "reliability_source": "served_pipeline",
        "cv_brier_before": cv_brier_before,
        "cv_brier_after": cv_brier_after,
        "brier_before": uncalibrated_test_metrics["brier_score"],
        "brier_after": calibrated_test_metrics["brier_score"],
        "reliability": [{"predicted": float(predicted), "observed": float(observed)} for predicted, observed in zip(mean_predicted, fraction_positive)],
    }
    background_rows = X_train.sample(n=min(100, len(X_train)), random_state=42)
    explain_background = model.named_steps["preprocess"].transform(background_rows)
    comparisons[selected]["test_metrics"] = test_metrics
    artifact = {"pipeline": model, "calibrated_pipeline": calibrated_model, "explain_background": explain_background, "model_name": selected, "trained_at": datetime.now(timezone.utc).isoformat(), "dataset": "WA_Fn-UseC_-Telco-Customer-Churn.csv", "row_count": len(df), "train_rows": len(X_train), "test_rows": len(X_test), "features": X.columns.tolist(), "class_balance": {"retained": int((y == 0).sum()), "churned": int((y == 1).sum()), "churn_prevalence": float(y.mean())}, "cv_metrics": scores, "comparisons": comparisons, "skipped_models": skipped_models, "test_metrics": test_metrics, "uncalibrated_test_metrics": uncalibrated_test_metrics, "calibration": calibration, "threshold": .5, "high_risk_threshold": .65}
    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_dir / "churn_model.joblib")
    public_artifact = {k:v for k,v in artifact.items() if k not in {"pipeline", "calibrated_pipeline", "explain_background"}}
    (artifact_dir / "metrics.json").write_text(json.dumps(public_artifact, indent=2), encoding="utf-8")
    print(json.dumps(public_artifact, indent=2))
    return artifact

if __name__ == "__main__":
    train_models()
