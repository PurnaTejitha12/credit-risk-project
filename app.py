from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from xgboost import XGBClassifier


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

VECTOR_PATH = BASE_DIR / "text_vectorizer.pkl"
MODEL_PATH = BASE_DIR / "xgboost_credit_risk_model.json"


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Customer Support Intent Classification API",
    description="API for customer support intent prediction using TF-IDF and XGBoost.",
    version="1.0.0",
)


# --------------------------------------------------
# CORS configuration
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Load vectorizer
# --------------------------------------------------

vectorizer = joblib.load(VECTOR_PATH)


# --------------------------------------------------
# Load XGBoost model
# --------------------------------------------------

model = XGBClassifier()
model.load_model(MODEL_PATH)


print("Model loaded successfully!")
print("Model expects:", model.n_features_in_)
print(
    "Vectorizer produces:",
    len(vectorizer.get_feature_names_out())
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class PredictionRequest(BaseModel):
    text: str


# --------------------------------------------------
# Health check endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Customer Support Intent Classification API is running",
        "status": "healthy",
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    text = request.text.strip()

    if not text:
        return {
            "error": "Text cannot be empty."
        }


    # --------------------------------------------------
    # Convert text into TF-IDF features
    # --------------------------------------------------

    X = vectorizer.transform([text])

    X = X.toarray()


    # --------------------------------------------------
    # Handle model/vectorizer feature mismatch
    # --------------------------------------------------

    expected_features = model.n_features_in_
    actual_features = X.shape[1]

    if actual_features < expected_features:

        missing = expected_features - actual_features

        X = np.pad(
            X,
            ((0, 0), (0, missing)),
            mode="constant",
        )

    elif actual_features > expected_features:

        X = X[:, :expected_features]


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    prediction = model.predict(X)

    probabilities = model.predict_proba(X)


    predicted_class = prediction[0]

    confidence = float(
        np.max(probabilities[0]) * 100
    )


    # --------------------------------------------------
    # API response
    # --------------------------------------------------

    return {
        "intent": str(predicted_class),
        "confidence": round(confidence, 2),
    }