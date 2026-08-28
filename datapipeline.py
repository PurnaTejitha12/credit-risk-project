import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# 1. Load dataset
data = pd.read_csv("aiml_training_data.csv")

# 2. Separate features and labels
X_text = data["text"]
y_text = data["intent"]

# 3. Encode intent labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

# 4. Split dataset
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. TF-IDF vectorization
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

# 6. Train XGBoost
model = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)

# 7. Save preprocessing and model
joblib.dump(vectorizer, "text_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

model.save_model("xgboost_credit_risk_model.json")

print("Pipeline completed successfully!")

