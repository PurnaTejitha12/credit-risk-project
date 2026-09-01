import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data_preprocessing import (
    load_data,
    clean_data,
    split_features_target,
    create_preprocessor
)


DATA_PATH = "../data/raw/credit_risk_dataset.csv"
MODEL_PATH = "../models/xgboost_credit_risk_model.pkl"
PREPROCESSOR_PATH = "../models/credit_risk_preprocessor.pkl"
FEATURE_NAMES_PATH = "../models/feature_names.pkl"


def train_model():

    # Load data
    df = load_data(DATA_PATH)

    # Clean data
    df = clean_data(df)

    # Separate features and target
    X, y = split_features_target(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Create preprocessing pipeline
    preprocessor = create_preprocessor(X_train)

    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Calculate class imbalance ratio
    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()

    scale_pos_weight = negative / positive

    # XGBoost model
    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )

    # Train
    model.fit(
        X_train_processed,
        y_train
    )

    # Feature names
    feature_names = preprocessor.get_feature_names_out()

    # Save model
    joblib.dump(
        model,
        MODEL_PATH
    )

    # Save preprocessor
    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    # Save feature names
    joblib.dump(
        feature_names,
        FEATURE_NAMES_PATH
    )

    print("Model training completed.")
    print("Model saved to:", MODEL_PATH)
    print("Preprocessor saved to:", PREPROCESSOR_PATH)
    print("Feature names saved to:", FEATURE_NAMES_PATH)

    return model, preprocessor, X_test, y_test


if __name__ == "__main__":
    train_model()
