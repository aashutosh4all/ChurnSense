from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# ----------------------------------------------------
# App initialization
# ----------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting bank customer churn using a trained Random Forest model.",
    version="1.0"
)


# ----------------------------------------------------
# Load model artifacts
# ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

model_path = ARTIFACTS_DIR / "final_random_forest_churn_model.pkl"
columns_path = ARTIFACTS_DIR / "model_columns.json"
config_path = ARTIFACTS_DIR / "model_config.json"

model = joblib.load(model_path)

with open(columns_path, "r") as file:
    model_columns = json.load(file)

with open(config_path, "r") as file:
    model_config = json.load(file)

threshold = model_config.get("threshold", 0.4)


# ----------------------------------------------------
# Input schema
# ----------------------------------------------------
class CustomerData(BaseModel):
    CreditScore: int = Field(..., example=650)
    Geography: str = Field(..., example="Germany")
    Gender: str = Field(..., example="Female")
    Age: int = Field(..., example=45)
    Tenure: int = Field(..., example=3)
    Balance: float = Field(..., example=120000)
    NumOfProducts: int = Field(..., example=2)
    HasCrCard: int = Field(..., example=1)
    IsActiveMember: int = Field(..., example=0)
    EstimatedSalary: float = Field(..., example=100000)


# ----------------------------------------------------
# Convert churn probability into risk level
# ----------------------------------------------------
def get_risk_level(churn_probability: float) -> str:
    if churn_probability < 0.30:
        return "Low Risk"
    elif churn_probability < 0.50:
        return "Moderate Risk"
    elif churn_probability < 0.70:
        return "High Risk"
    else:
        return "Critical Risk"


# ----------------------------------------------------
# Basic retention suggestion
# ----------------------------------------------------
def get_retention_suggestion(risk_level: str) -> str:
    if risk_level == "Low Risk":
        return "Customer appears stable. Continue regular engagement."
    elif risk_level == "Moderate Risk":
        return "Monitor customer activity and provide personalized offers."
    elif risk_level == "High Risk":
        return "Prioritize retention outreach and offer targeted benefits."
    else:
        return "Immediate retention action recommended with premium support or personalized intervention."


# ----------------------------------------------------
# Preprocess input data
# ----------------------------------------------------
def preprocess_input(customer: CustomerData) -> pd.DataFrame:
    input_data = pd.DataFrame(
        [{
            "CreditScore": customer.CreditScore,
            "Age": customer.Age,
            "Tenure": customer.Tenure,
            "Balance": customer.Balance,
            "NumOfProducts": customer.NumOfProducts,
            "HasCrCard": customer.HasCrCard,
            "IsActiveMember": customer.IsActiveMember,
            "EstimatedSalary": customer.EstimatedSalary,

            # One-hot encoded columns
            # France is baseline, so Germany = 0 and Spain = 0 means France
            "Geography_Germany": 1 if customer.Geography == "Germany" else 0,
            "Geography_Spain": 1 if customer.Geography == "Spain" else 0,

            # Female is baseline, so Gender_Male = 0 means Female
            "Gender_Male": 1 if customer.Gender == "Male" else 0
        }]
    )

    # Ensure columns are in the same order as training data
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    return input_data


# ----------------------------------------------------
# API routes
# ----------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running.",
        "model": model_config.get("model_name", "Random Forest"),
        "threshold": threshold
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):
    processed_input = preprocess_input(customer)

    churn_probability = model.predict_proba(processed_input)[0][1]
    prediction = "Churn" if churn_probability >= threshold else "No Churn"
    risk_level = get_risk_level(churn_probability)
    suggestion = get_retention_suggestion(risk_level)

    return {
        "churn_probability": round(float(churn_probability), 4),
        "churn_probability_percentage": round(float(churn_probability) * 100, 2),
        "prediction": prediction,
        "risk_level": risk_level,
        "threshold_used": threshold,
        "retention_suggestion": suggestion
    }