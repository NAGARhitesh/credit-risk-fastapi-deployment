import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.path.join("artifacts", "credit_risk_model.joblib")

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts whether a borrower is risky or safe using a trained ML pipeline.",
    version="1.0.0"
)

model = None


class CreditRiskInput(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: float


@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            f"Run train_credit_risk.py first."
        )
    model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Credit Risk Prediction API is running.",
        "docs": "/docs"
    }


@app.post("/predict")
def predict_credit_risk(data: CreditRiskInput):
    try:
        input_df = pd.DataFrame([{
            "person_age": data.person_age,
            "person_income": data.person_income,
            "person_home_ownership": data.person_home_ownership,
            "person_emp_length": data.person_emp_length,
            "loan_intent": data.loan_intent,
            "loan_grade": data.loan_grade,
            "loan_amnt": data.loan_amnt,
            "loan_int_rate": data.loan_int_rate,
            "loan_percent_income": data.loan_percent_income,
            "cb_person_default_on_file": data.cb_person_default_on_file,
            "cb_person_cred_hist_length": data.cb_person_cred_hist_length,
        }])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        label = "Risky Borrower" if int(prediction) == 1 else "Safe Borrower"

        return {
            "prediction": int(prediction),
            "prediction_label": label,
            "risk_probability": round(float(probability), 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))