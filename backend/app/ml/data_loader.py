from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def load_data(path: str | Path = DATA) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Telco churn CSV not found: {source}. Download the IBM Telco Customer Churn dataset and place it at backend/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(source)
    required = {
        "customerID", "Churn", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    df = df.drop_duplicates(subset="customerID").copy()
    if df.empty:
        raise ValueError("Dataset contains no customer records")
    if df["customerID"].isna().any() or df["Churn"].isna().any():
        raise ValueError("Dataset contains missing customer IDs or churn labels")
    if not set(df["Churn"].unique()).issubset({"Yes", "No"}):
        raise ValueError("Churn must contain only Yes and No labels")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df
