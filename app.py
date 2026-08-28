import joblib
import numpy as np
from xgboost import XGBClassifier

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
# User input
# -----------------------------
text = input("\nEnter your message: ")

# Convert text to TF-IDF features
X = vectorizer.transform([text])

# Convert sparse matrix to numpy array
X = X.toarray()

print("Original feature shape:", X.shape)

# -----------------------------
# TEMPORARY compatibility test
# -----------------------------
expected_features = model.n_features_in_
actual_features = X.shape[1]

if actual_features < expected_features:
    missing = expected_features - actual_features

    print(f"Adding {missing} temporary feature(s) for testing...")

    X = np.pad(
        X,
        ((0, 0), (0, missing)),
        mode="constant"
    )

elif actual_features > expected_features:
    X = X[:, :expected_features]

print("Final feature shape:", X.shape)

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(X)

# Get probability
probabilities = model.predict_proba(X)

predicted_class = prediction[0]
confidence = np.max(probabilities[0]) * 100

print("\n-----------------------------")
print("Prediction")
print("-----------------------------")
print("Intent:", predicted_class)
print(f"Confidence: {confidence:.2f}%")
