"""Normalize IBM and Maven telecom churn files to Veyra's trained model schema."""
import io
import re

import pandas as pd


def _key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


ALIASES = {
    "customerid": "customerID",
    "custid": "customerID",
    "gender": "gender",
    "seniorcitizen": "SeniorCitizen",
    "partner": "Partner",
    "married": "Partner",
    "dependents": "Dependents",
    "tenure": "tenure",
    "tenureinmonths": "tenure",
    "phoneservice": "PhoneService",
    "multiplelines": "MultipleLines",
    "internetservice": "InternetService",
    "internettype": "InternetService",
    "onlinesecurity": "OnlineSecurity",
    "onlinebackup": "OnlineBackup",
    "deviceprotection": "DeviceProtection",
    "deviceprotectionplan": "DeviceProtection",
    "techsupport": "TechSupport",
    "premiumtechsupport": "TechSupport",
    "streamingtv": "StreamingTV",
    "streamingmovies": "StreamingMovies",
    "contract": "Contract",
    "contracttype": "Contract",
    "paperlessbilling": "PaperlessBilling",
    "paymentmethod": "PaymentMethod",
    "monthlycharges": "MonthlyCharges",
    "monthlycharge": "MonthlyCharges",
    "totalcharges": "TotalCharges",
    "totalcharge": "TotalCharges",
    "churn": "Churn",
    "churnlabel": "Churn",
    "customerstatus": "Churn",
    "actualchurn": "Churn",
}


def parse_customer_csv(content: bytes, feature_names, require_labels=False, max_rows=100_000):
    try:
        frame = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", sep=None, engine="python")
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    original_columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
    keys = [_key(column) for column in original_columns]
    # The Maven customer table has both a Yes/No Internet Service flag and a
    # richer Internet Type field. Prefer Internet Type for the model input.
    preferred = {"InternetService": "internettype", "Churn": "churnlabel"}
    canonical_columns = [ALIASES.get(key, column) for column, key in zip(original_columns, keys)]
    winners = {}
    for index, (canonical, key) in enumerate(zip(canonical_columns, keys)):
        if canonical not in winners:
            winners[canonical] = index
            continue
        old_index = winners[canonical]
        old_key = keys[old_index]
        if preferred.get(canonical) == key:
            # Keep the less-specific source field under its original header
            # so it can help fill missing values (for example Internet Service).
            canonical_columns[old_index] = original_columns[old_index]
            winners[canonical] = index
        elif preferred.get(canonical) == old_key:
            canonical_columns[index] = original_columns[index]
        else:
            raise ValueError("CSV contains duplicate columns after normalizing names")
    frame = frame.copy()
    frame.columns = canonical_columns

    # Population lookup files have no per-customer target or account attributes.
    if {"zipcode", "population"}.issubset(set(keys)) and not any(key in keys for key in ("customerid", "tenure", "tenureinmonths")):
        raise ValueError("This is a Zip Code Population lookup file, not customer churn records. Upload the main Customer Churn CSV instead.")

    # Maven's customer fact table exposes age and dependent count rather than
    # IBM's senior-citizen and dependent yes/no fields. Derive those equivalents.
    if "SeniorCitizen" not in frame and "Age" in frame:
        age = pd.to_numeric(frame["Age"], errors="coerce")
        frame["SeniorCitizen"] = (age >= 65).astype("int64")
    if "Dependents" not in frame and "Number_of_Dependents" in frame:
        dependent_count = pd.to_numeric(frame["Number_of_Dependents"], errors="coerce")
        frame["Dependents"] = dependent_count.gt(0).map({True: "Yes", False: "No"})

    missing = set(feature_names) - set(frame.columns)
    if missing:
        raise ValueError("This CSV is missing required telecom customer fields: " + ", ".join(sorted(missing)) + ". Veyra supports IBM Telco and Maven Customer Churn schemas; lookup/reference tables are not scoreable.")
    if frame.empty:
        raise ValueError("CSV has a header but no customer rows")
    if len(frame) > max_rows:
        raise ValueError(f"CSV may contain at most {max_rows:,} customer rows")

    if "SeniorCitizen" in frame:
        senior = frame["SeniorCitizen"].astype(str).str.strip().str.casefold().replace({"yes": "1", "no": "0", "true": "1", "false": "0"})
        frame["SeniorCitizen"] = pd.to_numeric(senior, errors="coerce")
    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        raw = frame[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & raw.astype(str).str.strip().ne("") & numeric.isna()
        if invalid.any():
            raise ValueError(f"Column {column} contains non-numeric values (for example: {raw[invalid].iloc[0]!r})")
        if column != "TotalCharges" and numeric.isna().any():
            raise ValueError(f"Column {column} must contain a number for every customer")
        frame[column] = numeric

    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].map(lambda value: value.strip() if isinstance(value, str) else value)

    case_maps = {
        "InternetService": {"dsl": "DSL", "fiber optic": "Fiber optic", "no": "No", "none": "No", "no internet": "No", "no internet service": "No"},
        "Contract": {"month-to-month": "Month-to-month", "month to month": "Month-to-month", "one year": "One year", "two year": "Two year"},
        "PaymentMethod": {"electronic check": "Electronic check", "mailed check": "Mailed check", "bank transfer": "Bank transfer (automatic)", "credit card": "Credit card (automatic)"},
        "MultipleLines": {"no phone service": "No phone service"},
        "OnlineSecurity": {"no internet service": "No internet service"},
        "OnlineBackup": {"no internet service": "No internet service"},
        "DeviceProtection": {"no internet service": "No internet service", "no device protection": "No"},
        "TechSupport": {"no internet service": "No internet service"},
        "StreamingTV": {"no internet service": "No internet service"},
        "StreamingMovies": {"no internet service": "No internet service"},
    }
    # The Maven table has both a Yes/No internet-service flag and an Internet
    # Type. Use the type where supplied and represent no-service rows correctly.
    raw_internet = next((column for column, key in zip(original_columns, keys) if key == "internetservice" and column in frame.columns), None)
    if "InternetService" in frame:
        internet_type = frame["InternetService"].astype("string").str.strip()
        if raw_internet and raw_internet != "InternetService":
            no_service = frame[raw_internet].astype("string").str.strip().str.casefold().eq("no")
            internet_type = internet_type.mask(internet_type.isna() & no_service, "No")
        # If a file only supplies an Internet Service flag, avoid pretending
        # that Yes means DSL or fiber; unknown types remain an unseen category.
        internet_type = internet_type.mask(internet_type.str.casefold().eq("yes"), "Unknown")
        frame["InternetService"] = internet_type
    for column, mapping in case_maps.items():
        if column in frame:
            frame[column] = frame[column].map(lambda value: mapping.get(str(value).strip().casefold(), value) if pd.notna(value) else value)
    for column in ("Partner", "Dependents", "PhoneService", "PaperlessBilling", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"):
        if column in frame:
            frame[column] = frame[column].map(lambda value: {"yes": "Yes", "no": "No"}.get(str(value).strip().casefold(), value) if pd.notna(value) else value)
    if "MultipleLines" in frame and "PhoneService" in frame:
        frame.loc[frame["MultipleLines"].isna() & frame["PhoneService"].eq("No"), "MultipleLines"] = "No phone service"
    if "InternetService" in frame:
        no_internet = frame["InternetService"].eq("No")
        for column in ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"):
            if column in frame:
                frame.loc[frame[column].isna() & no_internet, column] = "No internet service"

    excluded_unlabeled = 0
    if "Churn" in frame:
        raw_labels = frame["Churn"].astype("string").str.strip().str.casefold()
        labels = raw_labels.map({"yes": "Yes", "true": "Yes", "1": "Yes", "churned": "Yes", "no": "No", "false": "No", "0": "No", "stayed": "No", "joined": pd.NA, "": pd.NA})
        invalid = raw_labels.notna() & ~raw_labels.isin(["yes", "true", "1", "churned", "no", "false", "0", "stayed", "joined", ""])
        if invalid.any():
            raise ValueError("Churn labels must be Yes/No or Maven Customer Status values Churned/Stayed/Joined")
        frame["Churn"] = labels
        excluded_unlabeled = int(frame["Churn"].isna().sum())
        if require_labels:
            frame = frame.loc[frame["Churn"].notna()].copy()
            if frame.empty:
                raise ValueError("No labeled customer outcomes remain after excluding Joined/unlabeled rows")
            if frame["Churn"].nunique() < 2:
                raise ValueError("Evaluation CSV must include both churned and retained outcomes")
    elif require_labels:
        raise ValueError("Include a Churn or Maven Customer Status column to evaluate actual outcomes")

    frame.attrs["excluded_unlabeled"] = excluded_unlabeled
    return frame
