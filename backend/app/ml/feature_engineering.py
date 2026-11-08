DROP_COLUMNS = ("customerID", "Churn")

def engineer_features(X):
    """Create deterministic, row-local customer features without using the target."""
    X = X.copy()
    service_columns = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    present = [column for column in service_columns if column in X.columns]
    if present:
        X["active_service_count"] = sum(X[column].astype(str).eq("Yes") for column in present)
    if {"TotalCharges", "tenure", "MonthlyCharges"}.issubset(X.columns):
        tenure = X["tenure"].replace(0, float("nan"))
        X["average_charge_per_tenure_month"] = (X["TotalCharges"] / tenure).fillna(X["MonthlyCharges"])
    return X

def split_features_target(df):
    X = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    y = df["Churn"].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Churn must contain only Yes and No labels")
    y = y.astype(int)
    return engineer_features(X), y
