from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "baseline_xgboost_model.pkl"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "baseline_preprocessor.pkl"
)

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_test_data.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORT_PATH = (
    REPORT_DIR
    / "baseline_report.txt"
)


TARGET_COLUMN = "loan_status"


# ============================================================
# EVALUATION
# ============================================================

def evaluate_baseline():

    print("=" * 60)
    print("FINWISE CREDIT RISK - BASELINE EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = joblib.load(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # 2. Load preprocessor
    # --------------------------------------------------------

    print("Loading preprocessor...")

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    # --------------------------------------------------------
    # 3. Load test data
    # --------------------------------------------------------

    print("Loading test data...")

    test_df = pd.read_csv(
        TEST_DATA_PATH
    )

    X_test = test_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_test = test_df[TARGET_COLUMN]

    # --------------------------------------------------------
    # 4. Transform test data
    # --------------------------------------------------------

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    # --------------------------------------------------------
    # 5. Generate predictions
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            X_test_processed
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # 6. Calculate metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    # Gini coefficient
    gini = (
        2 * roc_auc
    ) - 1

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # 7. Print results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        f"Gini     : {gini:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # 8. Check acceptance criteria
    # --------------------------------------------------------

    gini_target = 0.45

    if gini >= gini_target:

        gini_status = "PASS"

    else:

        gini_status = "NOT MET"

    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA")
    print("=" * 60)

    print(
        f"Required Gini: >= {gini_target:.2f}"
    )

    print(
        f"Actual Gini  : {gini:.4f}"
    )

    print(
        f"Status        : {gini_status}"
    )

    # --------------------------------------------------------
    # 9. Create reports directory
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 10. Write baseline report
    # --------------------------------------------------------

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as report:

        report.write(
            "FINWISE LENDING - BASELINE MODEL REPORT\n"
        )

        report.write(
            "=" * 60 + "\n\n"
        )

        report.write(
            "1. MODEL OVERVIEW\n"
        )

        report.write(
            "-" * 60 + "\n"
        )

        report.write(
            "Model: XGBoost Classifier\n"
        )

        report.write(
            "Task: Binary Credit Risk Classification\n"
        )

        report.write(
            "Test Split: 20%\n"
        )

        report.write(
            "Random State: 42\n\n"
        )

        report.write(
            "2. EVALUATION METRICS\n"
        )

        report.write(
            "-" * 60 + "\n"
        )

        report.write(
            f"Accuracy : {accuracy:.4f}\n"
        )

        report.write(
            f"Precision: {precision:.4f}\n"
        )

        report.write(
            f"Recall   : {recall:.4f}\n"
        )

        report.write(
            f"F1 Score : {f1:.4f}\n"
        )

        report.write(
            f"ROC-AUC  : {roc_auc:.4f}\n"
        )

        report.write(
            f"Gini     : {gini:.4f}\n\n"
        )

        report.write(
            "3. GINI ACCEPTANCE CRITERION\n"
        )

        report.write(
            "-" * 60 + "\n"
        )

        report.write(
            "Required Gini: >= 0.45\n"
        )

        report.write(
            f"Actual Gini: {gini:.4f}\n"
        )

        report.write(
            f"Status: {gini_status}\n\n"
        )

        report.write(
            "4. CONFUSION MATRIX\n"
        )

        report.write(
            "-" * 60 + "\n"
        )

        report.write(
            str(cm)
        )

        report.write(
            "\n\n"
        )

        report.write(
            "5. CLASSIFICATION REPORT\n"
        )

        report.write(
            "-" * 60 + "\n"
        )

        report.write(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

        report.write(
            "\n6. BASELINE CONCLUSION\n"
        )

        report.write(
            "-" * 60 + "\n"
        )

        if gini >= gini_target:

            report.write(
                "The baseline model meets the "
                "minimum Gini acceptance criterion "
                "of 0.45.\n"
            )

            report.write(
                "Further model improvement can focus "
                "on calibration, hyperparameter tuning, "
                "explainability, and fairness.\n"
            )

        else:

            report.write(
                "The baseline model does not meet "
                "the minimum Gini acceptance criterion "
                "of 0.45.\n"
            )

            report.write(
                "Further feature engineering, "
                "hyperparameter tuning, and model "
                "optimization are required.\n"
            )

    print(
        f"\nBaseline report saved to:"
        f"\n{REPORT_PATH}"
    )

    print(
        "\nBaseline evaluation completed."
    )


if __name__ == "__main__":
    evaluate_baseline()
