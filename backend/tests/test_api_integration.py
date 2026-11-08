import io
import json
import unittest
import pandas as pd

from fastapi.testclient import TestClient

from app.main import app
from app.ml.data_loader import load_data
from app.ml.model_registry import load_model


@unittest.skipUnless(load_model() is not None, "Train the model before running integration tests")
class ApiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.data = load_data()
        cls.customer = cls.data.iloc[0]
        cls.customer_id = str(cls.customer.customerID)

    def test_dashboard_and_model_routes(self):
        self.assertEqual(self.client.get("/api/health").status_code, 200)
        overview = self.client.get("/api/overview")
        self.assertEqual(overview.status_code, 200)
        metrics = overview.json()["metrics"]
        self.assertEqual(metrics["total_customers"], len(self.data))
        self.assertEqual(metrics["churned_customers"] + metrics["retained_customers"], len(self.data))
        analytics = self.client.get("/api/analytics")
        self.assertEqual(analytics.status_code, 200)
        self.assertGreater(len(analytics.json()["contract"]), 0)
        segments = self.client.get("/api/segments")
        self.assertEqual(segments.status_code, 200)
        self.assertIn(segments.json()["selected_k"], range(2, 9))
        self.assertGreater(len(segments.json()["profiles"]), 1)
        self.assertEqual(self.client.get("/api/forecast").status_code, 200)
        model = self.client.get("/api/model")
        self.assertEqual(model.status_code, 200)
        self.assertIn("feature_importance", model.json())
        self.assertNotIn("explain_background", model.json())

    def test_vite_fallback_port_cors(self):
        for origin in ("http://127.0.0.1:5173", "http://localhost:5174"):
            response = self.client.get("/api/health", headers={"Origin": origin})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers.get("access-control-allow-origin"), origin)

    def test_customer_search_detail_and_retention(self):
        result = self.client.get("/api/customers", params={"q": self.customer_id, "limit": 5})
        self.assertEqual(result.status_code, 200)
        self.assertTrue(any(item["customer_id"] == self.customer_id for item in result.json()["items"]))
        retention = self.client.get(f"/api/retention/{self.customer_id}")
        self.assertEqual(retention.status_code, 200)
        self.assertIn("recommendations", retention.json())

    def test_single_and_bulk_predictions(self):
        record = json.loads(self.customer.drop(labels=["customerID", "Churn"]).to_json())
        prediction = self.client.post("/api/predict", json=record)
        self.assertEqual(prediction.status_code, 200, prediction.text)
        self.assertTrue(0 <= prediction.json()["probability"] <= 1)
        self.assertEqual(len(prediction.json()["prediction_event_id"]), 32)
        explanation = self.client.post("/api/explain", json=record)
        self.assertEqual(explanation.status_code, 200, explanation.text)
        self.assertTrue(explanation.json()["features"])
        self.assertTrue(any(abs(item["value"]) > 1e-8 for item in explanation.json()["features"]))
        csv = self.data.drop(columns=["Churn"]).head(2).to_csv(index=False).encode()
        bulk = self.client.post("/api/predict/bulk", files={"file": ("customers.csv", io.BytesIO(csv), "text/csv")})
        self.assertEqual(bulk.status_code, 200, bulk.text)
        self.assertEqual(bulk.json()["count"], 2)
        self.assertTrue(all(row.get("prediction_event_id") for row in bulk.json()["predictions"]))
        event_id = bulk.json()["predictions"][0]["prediction_event_id"]
        outcome = f"prediction_event_id,Churn\n{event_id},Yes\n".encode()
        attached = self.client.post("/api/monitoring/labels", files={"file": ("outcomes.csv", io.BytesIO(outcome), "text/csv")})
        self.assertEqual(attached.status_code, 200, attached.text)
        self.assertEqual(attached.json()["updated"], 1)

    def test_external_evaluation_is_independent(self):
        churned = self.data.loc[self.data.Churn == "Yes"].head(20)
        retained = self.data.loc[self.data.Churn == "No"].head(20)
        evaluation = pd.concat([churned, retained]).to_csv(index=False).encode()
        response = self.client.post("/api/model/evaluate?threshold=0.6", files={"file": ("future-labeled.csv", io.BytesIO(evaluation), "text/csv")})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["samples"], len(churned) + len(retained))
        self.assertEqual(body["threshold"], 0.6)
        self.assertEqual(len(body["metrics"]["confusion_matrix"]), 2)
        self.assertEqual(body["source"], "external labeled CSV")

    def test_maven_customer_schema_and_population_lookup(self):
        churned = self.data.loc[self.data.Churn == "Yes"].head(20).copy()
        retained = self.data.loc[self.data.Churn == "No"].head(20).copy()
        customer_rows = pd.concat([churned, retained])
        maven = customer_rows.rename(columns={
            "customerID": "Cust_ID", "gender": "Gender", "Partner": "Married",
            "tenure": "Tenure_in_Months", "PhoneService": "Phone_Service",
            "MultipleLines": "Multiple_Lines", "InternetService": "Internet_Type",
            "OnlineSecurity": "Online Security", "OnlineBackup": "Online Backup",
            "DeviceProtection": "Device Protection Plan", "TechSupport": "Premium Tech Support",
            "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming Movies",
            "Contract": "Contract", "PaperlessBilling": "Paperless_Billing",
            "PaymentMethod": "Payment_Method", "MonthlyCharges": "Monthly_Charge",
            "TotalCharges": "Total_Charges", "Churn": "Customer_Status",
        })
        maven["Age"] = customer_rows["SeniorCitizen"].map({0: 42, 1: 70})
        maven["Number_of_Dependents"] = customer_rows["Dependents"].map({"No": 0, "Yes": 2})
        maven = maven.drop(columns=["SeniorCitizen", "Dependents"])
        maven["Internet_Service"] = maven["Internet_Type"].map({"No": "No", "DSL": "Yes", "Fiber optic": "Yes"})
        maven["Internet_Type"] = maven["Internet_Type"].replace({"Fiber optic": "Fiber Optic", "No": pd.NA})
        maven["Customer_Status"] = maven["Customer_Status"].map({"Yes": "Churned", "No": "Stayed"})
        joined = maven.head(2).copy()
        joined["Customer_Status"] = "Joined"
        maven = pd.concat([maven, joined], ignore_index=True)
        payload = maven.to_csv(index=False).encode()

        evaluation = self.client.post("/api/model/evaluate", files={"file": ("maven-customers.csv", io.BytesIO(payload), "text/csv")})
        self.assertEqual(evaluation.status_code, 200, evaluation.text)
        self.assertEqual(evaluation.json()["samples"], 40)
        self.assertEqual(evaluation.json()["excluded_unlabeled"], 2)

        bulk = self.client.post("/api/predict/bulk", files={"file": ("maven-customers.csv", io.BytesIO(payload), "text/csv")})
        self.assertEqual(bulk.status_code, 200, bulk.text)
        self.assertEqual(bulk.json()["count"], 42)
        self.assertEqual(bulk.json()["labeled_count"], 40)
        self.assertIsNone(bulk.json()["predictions"][-1]["actual_churn"])

        population = b"Zip Code,Population\n90001,10000\n"
        invalid = self.client.post("/api/model/evaluate", files={"file": ("telecom_zipcode_population.csv", io.BytesIO(population), "text/csv")})
        self.assertEqual(invalid.status_code, 422)
        self.assertIn("Zip Code Population lookup file", invalid.json()["detail"])

    def test_training_api_persists_metrics(self):
        response = self.client.post("/api/model/train")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["row_count"], len(self.data))
        self.assertIn("Logistic Regression", response.json()["cv_metrics"])
        self.assertEqual(response.json()["calibration"]["reliability_source"], "served_pipeline")
        self.assertNotIn("explain_background", response.json())
        self.assertTrue(load_model()["test_metrics"]["pr_auc"] > 0)


if __name__ == "__main__":
    unittest.main()
