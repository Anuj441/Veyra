import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from .data_loader import load_data

def cluster_customers(k=None):
    df = load_data()
    if len(df) < 3:
        raise ValueError("At least three customers are required for segmentation")
    cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    X = df[cols].fillna(df[cols].median())
    scaled = StandardScaler().fit_transform(X)
    candidate_scores = []
    if k is None:
        candidates = range(2, min(8, len(df) - 1) + 1)
        candidate_scores = []
        for candidate in candidates:
            labels = KMeans(n_clusters=candidate, random_state=42, n_init=10).fit_predict(scaled)
            candidate_scores.append({"k": candidate, "silhouette": float(silhouette_score(scaled, labels, sample_size=min(2000, len(df)), random_state=42))})
        k = max(candidate_scores, key=lambda item: item["silhouette"])["k"]
    else:
        k = max(2, min(int(k), min(8, len(df) - 1)))
    model = KMeans(n_clusters=k, random_state=42, n_init=20).fit(scaled)
    points = PCA(n_components=2, random_state=42).fit_transform(scaled)
    clustered = df.assign(cluster=model.labels_, churned=(df["Churn"] == "Yes").astype(float))
    profiles = clustered.groupby("cluster").agg(
        customers=("customerID", "count"),
        tenure=("tenure", "mean"),
        MonthlyCharges=("MonthlyCharges", "mean"),
        TotalCharges=("TotalCharges", "mean"),
        churn_rate=("churned", "mean"),
    ).reset_index()
    charge_median = float(df["MonthlyCharges"].median())
    for profile in profiles.to_dict(orient="records"):
        tenure_label = "newer customers" if profile["tenure"] < 24 else "established customers" if profile["tenure"] < 48 else "long-tenure customers"
        charge_label = "above-median" if profile["MonthlyCharges"] >= charge_median else "below-median"
        profiles.loc[profiles.cluster == profile["cluster"], "description"] = (
            f"{tenure_label} with {charge_label} monthly charges; "
            f"observed churn is {profile['churn_rate']:.0%}."
        )
    score = float(silhouette_score(scaled, model.labels_, sample_size=min(2000, len(df)), random_state=42))
    return {"selected_k": k, "silhouette": score, "candidate_scores": candidate_scores, "clusters": [{"customer_id": str(cid), "cluster": int(label), "x": float(p[0]), "y": float(p[1])} for cid, label, p in zip(df.customerID, model.labels_, points)], "profiles": profiles.to_dict(orient="records")}
