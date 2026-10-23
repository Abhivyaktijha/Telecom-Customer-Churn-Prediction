from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

MODEL_PATH = "best_ada_model.pkl"
TENURE_BINS = list(range(1, 80, 12))
TENURE_LABELS = [f"{i}-{i + 11}" for i in range(1, 72, 12)]

app = FastAPI(title="Telecom Churn API")
model = joblib.load(MODEL_PATH)  # loaded once at startup


class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=1, le=72)
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
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


class PredictionOut(BaseModel):
    churn_prediction: str
    churn_probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(customer: CustomerInput):
    df = pd.DataFrame([customer.model_dump()])
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS,
        right=False, include_lowest=True
    )
    df = df.drop(columns=["tenure"])

    proba = float(model.predict_proba(df)[0][1])
    proba = float(model.predict_proba(df)[0][1])
    label = str(model.predict(df)[0])
    return PredictionOut(
        churn_prediction=label,
        churn_probability=round(proba, 4),
    )                                                                                                                                                                                               
    return PredictionOut(
        churn_prediction="Yes" if pred == 1 else "No",
        churn_probability=round(proba, 4),
    )