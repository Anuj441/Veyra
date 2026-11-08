# Veyra

AI-powered customer intelligence for telecom churn analysis. The frontend is backed by a FastAPI service, a stratified model comparison pipeline, and the public IBM Telco Customer Churn sample dataset.

## Start locally

1. Install Python 3.10+ and Node.js 20+.
2. The IBM CSV is expected at `backend/data/WA_Fn-UseC_-Telco-Customer-Churn.csv`. Its source and retrieval notes are in [`backend/data/README.md`](backend/data/README.md). The raw CSV is ignored by Git.
3. From `backend`, create an environment and install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. Train and persist the selected model:

   ```powershell
   .\.venv\Scripts\python.exe -m app.ml.train
   ```

5. Start the API from `backend`:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8001
   ```

6. In another terminal, start the frontend from `frontend`:

   ```powershell
   npm install
   npm run dev
   ```

The selected pipeline and metrics are saved under `backend/artifacts/` and ignored by Git. Training uses a stratified holdout split, five-fold stratified cross-validation on training data, and fold-fitted preprocessing. Selection prioritizes cross-validated PR-AUC, then recall. The holdout set is not used to select the model. Duplicate customer IDs are removed; missing `TotalCharges` values are numerically coerced and imputed within the model pipeline. Deterministic row-local features include active service count and average charge per tenure month.

The IBM dataset is a single snapshot. Time-series forecasting is therefore reported as unavailable until dated customer snapshots are supplied. Copilot answers summarize live API data and do not infer causal reasons for churn. Retention actions are suggestions based on model scores and observed attributes, not proven campaign outcomes.

## Implemented features

- Overview metrics and contract, tenure, payment, and charge analytics.
- Custom and bulk CSV churn predictions using the persisted pipeline; IBM Telco headers and the Maven Customer Churn master-table equivalents are normalized. Do not upload Maven's Zip Code Population lookup table as customer rows.
- Separate labeled IBM or Maven customer-table CSV evaluation reports metrics at the selected risk threshold without training on or retaining the uploaded evaluation rows. Maven `Joined` records are excluded because they do not yet have a churn outcome.
- Prediction responses include opaque `prediction_event_id` tokens. The scored-results download preserves these tokens so later actual outcomes can be uploaded without adding customer IDs to the monitoring database.
- Logistic Regression, Random Forest, XGBoost, and LightGBM comparison with holdout metrics and confusion matrices.
- Sigmoid calibration is selected by nested training cross-validation and applied only when it improves Brier score; an adjustable high-risk cutoff is available.
- Local drift, prediction-distribution, and labeled performance monitoring stored in SQLite.
- Revenue-at-risk scenarios with editable save-rate and contact-cost assumptions; displayed financial projections are estimates.
- Searchable, risk-filtered and probability-sorted customer records with customer-specific retention actions.
- K-Means segmentation with silhouette-based selection from k=2 through k=8, segment profiles, and PCA visualization.
- Model comparison, holdout metrics, class balance, confusion matrix, feature importance, and local SHAP explanations.
- Guided, dataset-backed Copilot prompts and an explicit unavailable state for time-series forecasts.

Optional model libraries are checked for compatibility. XGBoost must be version 2.1.4 or newer to work with the pinned scikit-learn 1.6; incompatible XGBoost and unavailable LightGBM candidates are reported in training metadata rather than silently assigned failed scores.

## API

FastAPI docs: `http://127.0.0.1:8001/docs`

- `GET /api/health`, `/api/overview`, `/api/customers`, `/api/analytics`, `/api/segments`, `/api/forecast`, `/api/model`
- `POST /api/model/train`, `/api/predict`, `/api/predict/bulk`, `/api/explain`, `/api/retention/recommendations`
- `POST /api/model/evaluate` accepts a labeled IBM Telco shaped CSV and an optional `threshold` query parameter.
- `GET /api/monitoring`; `POST /api/monitoring/labels` accepts a CSV with `prediction_event_id,Churn` columns (`Churn` values `Yes`/`No`).
- `GET /api/retention/{customer_id}`

## Automated integration checks

After installing backend requirements and training a model, run from `backend`:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
