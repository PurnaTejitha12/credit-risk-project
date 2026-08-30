import joblib
import numpy as np
from xgboost import XGBClassifier
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# FastAPI application
# -----------------------------
app = FastAPI(title="Credit Risk API")


# -----------------------------
# Load vectorizer
# -----------------------------
vectorizer = joblib.load("text_vectorizer.pkl")


# -----------------------------
# Load XGBoost model
# -----------------------------
model = XGBClassifier()
model.load_model("xgboost_credit_risk_model.json")


print("Model loaded successfully!")
print("Model expects:", model.n_features_in_)
print("Vectorizer produces:", len(vectorizer.get_feature_names_out()))


# -----------------------------
# Request format
# -----------------------------
class PredictionRequest(BaseModel):
    text: str


# -----------------------------
# Home endpoint
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Credit Risk API is running"
    }


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(request: PredictionRequest):

    # Convert text to TF-IDF features
    X = vectorizer.transform([request.text])

    # Convert sparse matrix to numpy array
    X = X.toarray()

    # -----------------------------
    # Compatibility adjustment
    # -----------------------------
    expected_features = model.n_features_in_
    actual_features = X.shape[1]

    if actual_features < expected_features:
        missing = expected_features - actual_features

        X = np.pad(
            X,
            ((0, 0), (0, missing)),
            mode="constant"
        )

    elif actual_features > expected_features:
        X = X[:, :expected_features]

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(X)

    probabilities = model.predict_proba(X)

    predicted_class = prediction[0]
    confidence = float(np.max(probabilities[0]) * 100)

    return {
        "prediction": str(predicted_class),
        "confidence": round(confidence, 2)
    }
