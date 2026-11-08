"""Small local monitor for scored feature batches and later-known churn labels."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .data_loader import DATA

DB_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "monitoring.sqlite3"
NUMERIC_FEATURES = ("tenure", "MonthlyCharges", "TotalCharges")
CATEGORICAL_FEATURES = ("Contract", "PaymentMethod", "InternetService")


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute("""CREATE TABLE IF NOT EXISTS prediction_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tracking_id TEXT,
        created_at TEXT NOT NULL,
        model_name TEXT NOT NULL,
        probability REAL NOT NULL,
        actual_churn INTEGER,
        input_json TEXT NOT NULL
    )""")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(prediction_events)")}
    if "tracking_id" not in columns:
        connection.execute("ALTER TABLE prediction_events ADD COLUMN tracking_id TEXT")
    connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_prediction_events_tracking_id ON prediction_events(tracking_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_prediction_events_date ON prediction_events(created_at)")
    return connection


def record_predictions(records, predictions, model_name, actual_labels=None):
    """Store inputs without customer IDs and return opaque IDs for delayed outcomes."""
    if len(records) != len(predictions):
        raise ValueError("Records and predictions must have the same length")
    if actual_labels is None:
        actual_labels = [None] * len(predictions)
    if len(actual_labels) != len(predictions):
        raise ValueError("Actual labels and predictions must have the same length")

    tracking_ids = []
    with _connect() as connection:
        for (_, record), prediction, actual in zip(records.iterrows(), predictions, actual_labels):
            values = {}
            for key, value in record.items():
                if key in {"customerID", "Churn"}:
                    continue
                if pd.isna(value):
                    values[key] = None
                elif isinstance(value, np.generic):
                    values[key] = value.item()
                else:
                    values[key] = value
            actual_value = None if actual is None or pd.isna(actual) else int(str(actual).strip().casefold() in {"yes", "1", "true"})
            tracking_id = uuid.uuid4().hex
            connection.execute("INSERT INTO prediction_events(tracking_id,created_at,model_name,probability,actual_churn,input_json) VALUES(?,?,?,?,?,?)", (tracking_id, datetime.now(timezone.utc).isoformat(), model_name, float(prediction["probability"]), actual_value, json.dumps(values, default=str)))
            tracking_ids.append(tracking_id)
    return tracking_ids


def attach_actual_labels(rows):
    """Attach Yes/No outcomes to previous scores using opaque tracking IDs."""
    if not rows:
        raise ValueError("No outcome rows were supplied")
    with _connect() as connection:
        for tracking_id, actual in rows:
            token = str(tracking_id).strip()
            if not token:
                raise ValueError("prediction_event_id cannot be blank")
            label = int(str(actual).strip().casefold() in {"yes", "1", "true"})
            current = connection.execute("SELECT actual_churn FROM prediction_events WHERE tracking_id = ?", (token,)).fetchone()
            if current is None:
                raise ValueError(f"Unknown prediction_event_id: {token}")
            if current[0] is not None and current[0] != label:
                raise ValueError(f"Outcome already recorded differently for prediction_event_id: {token}")
            connection.execute("UPDATE prediction_events SET actual_churn = ? WHERE tracking_id = ?", (label, token))
    return len(rows)

def _psi(reference, current, numeric):
    if numeric:
        base = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
        recent = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
        if not len(base) or not len(recent):
            return None
        edges = np.unique(np.quantile(base, [0, .2, .4, .6, .8, 1]))
        if len(edges) < 2:
            edges = np.array([base[0] - .5, base[0] + .5])
        edges[0], edges[-1] = -np.inf, np.inf
        expected = np.histogram(base, bins=edges)[0] / len(base)
        observed = np.histogram(recent, bins=edges)[0] / len(recent)
    else:
        base = reference.fillna("(missing)").astype(str)
        recent = current.fillna("(missing)").astype(str)
        categories = sorted(set(base.unique()) | set(recent.unique()))
        expected = base.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
        observed = recent.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    expected = np.clip(expected, 1e-6, None)
    observed = np.clip(observed, 1e-6, None)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def _performance(rows, threshold):
    actual = np.array([row["actual_churn"] for row in rows], dtype=int)
    probability = np.array([row["probability"] for row in rows], dtype=float)
    predicted = probability >= threshold
    return {
        "samples": int(len(rows)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probability)) if len(set(actual)) == 2 else None,
        "pr_auc": float(average_precision_score(actual, probability)),
        "brier_score": float(brier_score_loss(actual, probability)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
    }


def monitoring_report(reference, threshold=0.65, window_days=30):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    with _connect() as connection:
        rows = connection.execute("SELECT created_at, model_name, probability, actual_churn, input_json FROM prediction_events WHERE created_at >= ? ORDER BY created_at DESC LIMIT 10000", (cutoff,)).fetchall()
    events = [{"created_at": r[0], "model_name": r[1], "probability": r[2], "actual_churn": r[3], "input": json.loads(r[4])} for r in rows]
    labeled = [event for event in events if event["actual_churn"] is not None]
    drift = []
    if events:
        recent = pd.DataFrame([event["input"] for event in events])
        for feature in (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES):
            if feature not in reference or feature not in recent:
                continue
            score = _psi(reference[feature], recent[feature], feature in NUMERIC_FEATURES)
            drift.append({"feature": feature, "psi": score, "level": "low" if score is not None and score < .1 else "moderate" if score is not None and score < .25 else "high" if score is not None else "unavailable"})
    probabilities = [event["probability"] for event in events]
    distribution = None
    if probabilities:
        counts, edges = np.histogram(probabilities, bins=np.linspace(0, 1, 6))
        distribution = {"samples": len(probabilities), "mean_probability": float(np.mean(probabilities)), "median_probability": float(np.median(probabilities)), "high_risk_count": int(sum(p >= threshold for p in probabilities)), "bins": [{"range": f"{edges[i]:.1f}–{edges[i+1]:.1f}", "count": int(counts[i])} for i in range(len(counts))]}
    by_month = {}
    for event in labeled:
        month = event["created_at"][:7]
        by_month.setdefault(month, []).append(event)
    return {
        "window_days": window_days,
        "scored_samples": len(events),
        "labeled_samples": len(labeled),
        "data_drift": drift,
        "prediction_distribution": distribution,
        "performance": _performance(labeled, threshold) if labeled else None,
        "performance_over_time": [{"month": month, **_performance(samples, threshold)} for month, samples in sorted(by_month.items())],
        "message": "Monitoring compares scored requests with the local IBM sample as a baseline. Performance is shown only for uploaded rows with actual Churn labels.",
    }
