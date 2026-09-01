import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from data_preprocessing import (
    load_data,
    clean_data,
    split_features_target
)


DATA_PATH = "../data/raw/credit_risk_dataset.csv"
MODEL_PATH = "../models/xgboost_credit_risk_model.pkl"
PREPROCESSOR_PATH = "../models/credit_risk_preprocessor.pkl"


def evaluate_model():

    # Load dataset
    df = load_data(DATA_PATH)

    # Clean dataset
    df = clean_data(df)

    # Split features and target
    X, y = split_features_target(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Load model and preprocessor
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Transform test data
    X_test_processed = preprocessor.transform(X_test)

    # Predictions
    probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # ROC-AUC
    auc = roc_auc_score(
        y_test,
        probabilities
    )

    # Gini
    gini = (2 * auc) - 1

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print(f"ROC-AUC: {auc:.4f}")
    print(f"Gini:    {gini:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    if gini >= 0.45:
        print("\nGini Acceptance Criteria: PASS")
    else:
        print("\nGini Acceptance Criteria: NOT MET")


if __name__ == "__main__":
    evaluate_model()
