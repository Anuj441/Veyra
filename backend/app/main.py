from pathlib import Path
import io
import json
import os
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .ml.data_loader import DATA, load_data
from .ml.model_registry import load_model
from .ml.predict import predict_records
from .ml.train import train_models
from .ml.explain import explain_record
from .ml.clustering import cluster_customers
from .ml.forecasting import forecast_status
from .ml.feature_engineering import engineer_features, split_features_target
from .ml.monitoring import attach_actual_labels, monitoring_report, record_predictions
from .ml.csv_validation import parse_customer_csv

app = FastAPI(title="Veyra API", version="1.0.0")
cors_origins = [origin.strip() for origin in os.getenv("CHURNIQ_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    # Vite moves to the next port when its default is occupied (for example 5174).
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1):\d{1,5}$",
    allow_methods=["*"],
    allow_headers=["*"],
)

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

def _data():
    try: return load_data()
    except (FileNotFoundError, ValueError) as exc: raise HTTPException(503, str(exc))

@app.get("/api/health")
def health(): return {"status":"ok", "model_ready": load_model() is not None, "dataset_present": DATA.exists()}

@app.get("/api/overview")
def overview(high_risk_threshold:float=Query(.65,ge=.35,le=.95)):
    df = _data(); artifact = load_model()
    count = len(df); churned = int((df.Churn == "Yes").sum())
    monthly_revenue = float(df.MonthlyCharges.sum())
    if artifact:
        X = engineer_features(df.drop(columns=["customerID", "Churn"]))
        scoring_pipeline = artifact["calibrated_pipeline"] if artifact.get("calibration", {}).get("applied") else artifact["pipeline"]
        probs = scoring_pipeline.predict_proba(X)[:,1]
        risk = int((probs >= high_risk_threshold).sum())
        revenue = float(df.loc[probs >= high_risk_threshold, "MonthlyCharges"].sum())
    else: risk, revenue = None, None
    return {"status":"ready" if artifact else "unavailable", "metrics":{"total_customers":count,"churned_customers":churned,"retained_customers":count-churned,"churn_rate":churned/count if count else 0,"monthly_revenue":monthly_revenue,"revenue_at_risk":revenue,"revenue_at_risk_share":revenue/monthly_revenue if revenue is not None and monthly_revenue else None,"high_risk_customers":risk,"high_risk_threshold":high_risk_threshold},"trend":[],"customers":[]}

@app.get("/api/customers")
def customers(limit:int=50, offset:int=0, q:str="", risk:str|None=None, sort:str="desc", high_risk_threshold:float=Query(.65,ge=.35,le=.95)):
    if limit < 1 or limit > 200: raise HTTPException(422,"limit must be between 1 and 200")
    if offset < 0: raise HTTPException(422,"offset must be non-negative")
    if risk is not None and risk.upper() not in {"LOW","MEDIUM","HIGH"}: raise HTTPException(422,"risk must be LOW, MEDIUM or HIGH")
    if sort not in {"asc","desc"}: raise HTTPException(422,"sort must be asc or desc")
    df=_data(); artifact=load_model()
    if not artifact: return {"items":[],"total":len(df),"model_ready":False}
    X=engineer_features(df.drop(columns=["customerID","Churn"])); scoring_pipeline=artifact["calibrated_pipeline"] if artifact.get("calibration", {}).get("applied") else artifact["pipeline"]; probs=scoring_pipeline.predict_proba(X)[:,1]
    scored=df.assign(probability=probs, risk=["HIGH" if p >= high_risk_threshold else "MEDIUM" if p >= .35 else "LOW" for p in probs])
    if q.strip():
        needle=q.strip().casefold()
        mask=scored["customerID"].astype(str).str.casefold().str.contains(needle,regex=False) | scored["Contract"].astype(str).str.casefold().str.contains(needle,regex=False)
        scored=scored.loc[mask]
    if risk: scored=scored.loc[scored.risk==risk.upper()]
    total=len(scored); out=scored.sort_values("probability",ascending=sort=="asc").iloc[offset:offset+limit]
    return {"items":[{"customer_id":r.customerID,"contract":r.Contract,"tenure":int(r.tenure),"monthly_charges":float(r.MonthlyCharges),"probability":float(r.probability),"risk":r.risk,"churn":r.Churn} for r in out.itertuples()],"total":total,"model_ready":True}

@app.post("/api/predict")
def predict(customer:Customer, high_risk_threshold:float=Query(.65,ge=.35,le=.95)):
    try:
        record=pd.DataFrame([customer.model_dump()]); result=predict_records(record,high_risk_threshold)[0]
        tracking_ids=record_predictions(record,[result],load_model()["model_name"])
        return {**result,"prediction_event_id":tracking_ids[0]}
    except RuntimeError as exc: raise HTTPException(503,str(exc))

@app.post("/api/predict/bulk")
async def bulk(file:UploadFile=File(...), high_risk_threshold:float=Query(.65,ge=.35,le=.95)):
    if not file.filename.lower().endswith(".csv"): raise HTTPException(400,"Upload a CSV file")
    try:
        content=await file.read()
        if len(content)>20_000_000: raise HTTPException(413,"CSV files must be 20 MB or smaller")
        try: df=parse_customer_csv(content,list(Customer.model_fields))
        except ValueError as exc: raise HTTPException(422,str(exc))
        features=list(Customer.model_fields)
        actual_labels=None
        if "Churn" in df:
            actual_labels=df["Churn"].astype(object).where(df["Churn"].notna(),None).tolist()
        records=df[features]; preds=predict_records(records,high_risk_threshold)
        tracking_ids=record_predictions(records,preds,load_model()["model_name"],actual_labels)
        predictions=[]
        for index,(row,prediction) in enumerate(zip(df.to_dict(orient="records"),preds),1):
            item={"customer_id":str(row.get("customerID") or f"ROW {index}"),"prediction_event_id":tracking_ids[index-1],"churn_probability":prediction["probability"],"probability_calibrated":prediction["probability_calibrated"],"risk":prediction["risk"]}
            if actual_labels is not None: item["actual_churn"]=actual_labels[index-1]
            predictions.append(item)
        return {"count":len(preds),"labeled_count":sum(label is not None for label in (actual_labels or [])),"predictions":predictions}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(400,f"Could not process CSV: {exc}")

@app.post("/api/explain")
def explain(customer:Customer):
    try: return explain_record(pd.DataFrame([customer.model_dump()]))
    except RuntimeError as exc: raise HTTPException(503,str(exc))

@app.get("/api/segments")
def segments(k:int|None=None):
    try: return cluster_customers(None if k is None else max(2,min(k,8)))
    except (ValueError,FileNotFoundError) as exc: raise HTTPException(503,str(exc))

@app.get("/api/forecast")
def forecast(): return forecast_status()

@app.post("/api/model/train")
def train():
    try: return {k:v for k,v in train_models().items() if k not in {"pipeline","calibrated_pipeline","explain_background"}}
    except (FileNotFoundError,ValueError) as exc: raise HTTPException(400,str(exc))

@app.get("/api/model")
def model_info():
    artifact=load_model()
    if not artifact: return {"status":"unavailable","message":"No persisted model. Add the dataset and train a model."}
    result={k:v for k,v in artifact.items() if k not in {"pipeline","calibrated_pipeline","explain_background"}}
    estimator=artifact["pipeline"].named_steps["model"]
    names=artifact["pipeline"].named_steps["preprocess"].get_feature_names_out()
    weights=getattr(estimator,"feature_importances_",None)
    if weights is None and hasattr(estimator,"coef_"): weights=abs(estimator.coef_[0])
    if weights is not None:
        result["feature_importance"]=[{"feature":str(names[i]),"importance":float(weights[i])} for i in sorted(range(len(weights)),key=lambda i:float(weights[i]),reverse=True)[:20]]
    return result

@app.post("/api/model/evaluate")
async def evaluate_external(file:UploadFile=File(...), threshold:float=Query(.65,ge=.05,le=.95)):
    """Evaluate a separate labeled file without training on or retaining its rows."""
    if not file.filename.lower().endswith(".csv"): raise HTTPException(400,"Upload a CSV file")
    content=await file.read()
    if len(content)>20_000_000: raise HTTPException(413,"CSV files must be 20 MB or smaller")
    try:
        frame=parse_customer_csv(content,list(Customer.model_fields),require_labels=True)
        X,y=split_features_target(frame)
        artifact=load_model()
        if artifact is None: raise HTTPException(503,"Train a model before evaluating external data")
        pipeline=artifact["calibrated_pipeline"] if artifact.get("calibration",{}).get("applied") else artifact["pipeline"]
        from .ml.evaluate import evaluate
        return {"filename":Path(file.filename).name,"source":"external labeled CSV","samples":len(frame),"excluded_unlabeled":frame.attrs.get("excluded_unlabeled",0),"threshold":threshold,"model_name":artifact["model_name"],"metrics":evaluate(pipeline,X,y,threshold)}
    except HTTPException: raise
    except ValueError as exc: raise HTTPException(422,str(exc))
    except Exception as exc: raise HTTPException(400,f"Could not evaluate CSV: {exc}")

@app.get("/api/analytics")
def analytics():
    df=_data(); artifact=load_model()
    if artifact is None: raise HTTPException(503,"Train a model to view prediction analytics")
    X=engineer_features(df.drop(columns=["customerID","Churn"]))
    scoring_pipeline=artifact["calibrated_pipeline"] if artifact.get("calibration", {}).get("applied") else artifact["pipeline"]
    probabilities=scoring_pipeline.predict_proba(X)[:,1]
    scored=df.assign(probability=probabilities)
    by_contract=scored.groupby("Contract").agg(customers=("customerID","count"),churn_rate=("Churn",lambda s:float((s=="Yes").mean())),mean_probability=("probability","mean")).reset_index()
    by_tenure=pd.cut(scored["tenure"],bins=[-1,6,12,24,48,72],labels=["0-6","7-12","13-24","25-48","49-72"]).astype(str)
    tenure=scored.assign(tenure_band=by_tenure).groupby("tenure_band",observed=False).agg(customers=("customerID","count"),churn_rate=("Churn",lambda s:float((s=="Yes").mean()))).reset_index()
    by_payment=scored.groupby("PaymentMethod").agg(customers=("customerID","count"),churn_rate=("Churn",lambda s:float((s=="Yes").mean()))).reset_index()
    bins=pd.cut(scored["MonthlyCharges"],bins=5)
    charges=scored.assign(charge_band=bins.astype(str)).groupby("charge_band",observed=False).agg(customers=("customerID","count"),churn_rate=("Churn",lambda s:float((s=="Yes").mean()))).reset_index()
    return {"model":artifact["model_name"],"contract":by_contract.to_dict(orient="records"),"tenure":tenure.to_dict(orient="records"),"payment":by_payment.to_dict(orient="records"),"monthly_charges":charges.to_dict(orient="records")}

def _retention_recommendations(values, probability, monthly_charge_median, high_risk_threshold=.65):
    actions=[]
    if probability >= high_risk_threshold: actions.append("Prioritize a personal check-in because the model assigns high churn risk.")
    elif probability >= .35: actions.append("Include this customer in a monitored retention outreach because the model assigns medium churn risk.")
    if values.get("Contract")=="Month-to-month": actions.append("Discuss a 12-month plan with a transparent price comparison because the current contract is month-to-month.")
    if values.get("TechSupport")=="No" and values.get("InternetService")!="No": actions.append("Offer a technical support check-in because internet service is active and support is not subscribed.")
    if values.get("OnlineSecurity")=="No" and values.get("InternetService")!="No": actions.append("Ask whether an online security option would be useful because internet service is active and security is not subscribed.")
    if values.get("tenure",0)<12: actions.append("Schedule an onboarding follow-up because the customer has been active for less than a year.")
    if values.get("MonthlyCharges",0) >= monthly_charge_median: actions.append("Review the bill and plan fit because monthly charges are at or above the dataset median.")
    if values.get("PaymentMethod")=="Electronic check": actions.append("Confirm the customer is comfortable with electronic-check billing and explain available automatic payment options.")
    return actions or ["Review recent customer interactions before choosing an outreach offer."]

@app.post("/api/retention/recommendations")
def retention_recommendations(customer:Customer, high_risk_threshold:float=Query(.65,ge=.35,le=.95)):
    df=_data()
    record=pd.DataFrame([customer.model_dump()])
    try: result=predict_records(record,high_risk_threshold)[0]
    except RuntimeError as exc: raise HTTPException(503,str(exc))
    values=record.iloc[0].to_dict()
    return {**result,"recommendations":_retention_recommendations(values,result["probability"],float(df.MonthlyCharges.median()),high_risk_threshold),"recommendation_basis":"Recommendations are based on model probability and the entered customer attributes. They are not experimentally validated retention outcomes."}


@app.get("/api/retention/{customer_id}")
def retention(customer_id:str, high_risk_threshold:float=Query(.65,ge=.35,le=.95)):
    df=_data(); artifact=load_model()
    if artifact is None: raise HTTPException(503,"Train a model before requesting retention actions")
    row=df.loc[df.customerID.astype(str)==customer_id]
    if row.empty: raise HTTPException(404,"Customer not found in the dataset")
    record=row.drop(columns=["customerID","Churn"]); result=predict_records(record,high_risk_threshold)[0]
    values=record.iloc[0].to_dict(); actions=_retention_recommendations(values,result["probability"],float(df.MonthlyCharges.median()),high_risk_threshold)
    return {"customer_id":customer_id,**result,"customer":{"contract":values.get("Contract"),"tenure":int(values.get("tenure",0)),"monthly_charges":float(values.get("MonthlyCharges",0)),"internet_service":values.get("InternetService"),"payment_method":values.get("PaymentMethod"),"actual_churn":str(row.iloc[0].Churn)},"recommendations":actions,"recommendation_basis":"Recommendations are based on model probability and observed customer attributes. They are not experimentally validated retention outcomes."}

@app.get("/api/monitoring")
def monitoring(high_risk_threshold:float=Query(.65,ge=.35,le=.95),window_days:int=Query(30,ge=7,le=365)):
    return monitoring_report(_data(),high_risk_threshold,window_days)

@app.post("/api/monitoring/labels")
async def monitoring_labels(file:UploadFile=File(...)):
    if not file.filename.lower().endswith(".csv"): raise HTTPException(400,"Upload a CSV file")
    content=await file.read()
    if len(content)>10_000_000: raise HTTPException(413,"Outcome CSV files must be 10 MB or smaller")
    try:
        frame=pd.read_csv(io.BytesIO(content),encoding="utf-8-sig",sep=None,engine="python")
        canonical={"".join(ch for ch in str(c).casefold() if ch.isalnum()):str(c) for c in frame.columns}
        token=next((canonical[key] for key in ("predictioneventid","eventid","trackingid") if key in canonical),None)
        label=next((canonical[key] for key in ("churn","actualchurn","actualoutcome") if key in canonical),None)
        if not token or not label: raise ValueError("Include prediction_event_id and Churn columns")
        if frame.empty: raise ValueError("Outcome CSV has no rows")
        if len(frame)>100_000: raise ValueError("Outcome CSV may contain at most 100,000 rows")
        outcomes=frame[[token,label]].copy()
        outcomes.columns=["prediction_event_id","Churn"]
        outcomes["Churn"]=outcomes["Churn"].astype(str).str.strip().str.title()
        if outcomes["prediction_event_id"].astype(str).str.strip().eq("").any(): raise ValueError("prediction_event_id cannot be blank")
        if not outcomes["Churn"].isin(["Yes","No"]).all(): raise ValueError("Churn outcomes must be Yes or No")
        if outcomes["prediction_event_id"].duplicated().any(): raise ValueError("Outcome CSV contains duplicate prediction_event_id values")
        updated=attach_actual_labels(list(outcomes.itertuples(index=False,name=None)))
        return {"updated":updated,"message":"Actual outcomes attached to matching scored rows."}
    except HTTPException: raise
    except ValueError as exc: raise HTTPException(422,str(exc))
    except Exception as exc: raise HTTPException(400,f"Could not process outcome CSV: {exc}")

frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
