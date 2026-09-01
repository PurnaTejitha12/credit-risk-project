from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data_preprocessing import (
    load_data,
    clean_data,
    split_features_target,
    create_preprocessor
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "credit_risk_dataset.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "baseline_xgboost_model.pkl"
)

PREPROCESSOR_PATH = (
    MODEL_DIR
    / "baseline_preprocessor.pkl"
)

FEATURE_NAMES_PATH = (
    MODEL_DIR
    / "baseline_feature_names.pkl"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# TRAIN BASELINE MODEL
# ============================================================

def train_baseline_model():

    print("=" * 60)
    print("FINWISE CREDIT RISK - BASELINE MODEL")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load raw dataset
    # --------------------------------------------------------

    df = load_data(DATA_PATH)

    print(
        f"\nOriginal dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # 2. Clean data
    # --------------------------------------------------------

    df = clean_data(df)

    print(
        f"Cleaned dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # 3. Separate features and target
    # --------------------------------------------------------

    X, y = split_features_target(df)

    print(
        f"\nFeatures: {X.shape}"
    )

    print(
        f"Target: {y.shape}"
    )

    # --------------------------------------------------------
    # 4. Train/Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print("\nTrain/Test Split:")
    print(
        f"Training samples: {len(X_train)}"
    )
    print(
        f"Testing samples:  {len(X_test)}"
    )

    # --------------------------------------------------------
    # 5. Create preprocessor
    # --------------------------------------------------------

    preprocessor = create_preprocessor(
        X_train
    )

    # --------------------------------------------------------
    # 6. Fit ONLY on training data
    # --------------------------------------------------------

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    # Transform test data using the
    # already-fitted preprocessor

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    print(
        "\nProcessed training shape:",
        X_train_processed.shape
    )

    print(
        "Processed testing shape:",
        X_test_processed.shape
    )

    # --------------------------------------------------------
    # 7. Calculate class imbalance
    # --------------------------------------------------------

    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()

    if positive_count > 0:

        scale_pos_weight = (
            negative_count
            / positive_count
        )

    else:

        scale_pos_weight = 1.0

    print(
        "\nClass distribution:"
    )

    print(
        f"Class 0: {negative_count}"
    )

    print(
        f"Class 1: {positive_count}"
    )

    print(
        f"Scale positive weight: "
        f"{scale_pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # 8. Create baseline XGBoost model
    # --------------------------------------------------------

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.10,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )

    # --------------------------------------------------------
    # 9. Train model
    # --------------------------------------------------------

    print("\nTraining baseline XGBoost model...")

    model.fit(
        X_train_processed,
        y_train
    )

    print(
        "Baseline model training completed."
    )

    # --------------------------------------------------------
    # 10. Get transformed feature names
    # --------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # --------------------------------------------------------
    # 11. Create model directory
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 12. Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH
    )

    # --------------------------------------------------------
    # 13. Save preprocessor
    # --------------------------------------------------------

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    # --------------------------------------------------------
    # 14. Save feature names
    # --------------------------------------------------------

    joblib.dump(
        feature_names,
        FEATURE_NAMES_PATH
    )

    # --------------------------------------------------------
    # 15. Save test data
    # --------------------------------------------------------

    test_data = X_test.copy()

    test_data["loan_status"] = (
        y_test.values
    )

    test_data_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "baseline_test_data.csv"
    )

    test_data.to_csv(
        test_data_path,
        index=False
    )

    # --------------------------------------------------------
    # 16. Save predictions
    # --------------------------------------------------------

    test_probabilities = (
        model.predict_proba(
            X_test_processed
        )[:, 1]
    )

    predictions = (
        test_probabilities >= 0.5
    ).astype(int)

    predictions_df = pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted": predictions,
            "probability": test_probabilities
        }
    )

    predictions_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "baseline_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # --------------------------------------------------------
    # 17. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BASELINE MODEL ARTIFACTS")
    print("=" * 60)

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Preprocessor: "
        f"{PREPROCESSOR_PATH}"
    )

    print(
        f"Feature names: "
        f"{FEATURE_NAMES_PATH}"
    )

    print(
        f"Test data: "
        f"{test_data_path}"
    )

    print(
        f"Predictions: "
        f"{predictions_path}"
    )

    print("\nBaseline training completed successfully.")


if __name__ == "__main__":
    train_baseline_model()
