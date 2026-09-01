from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "credit_risk_dataset.csv"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "credit_risk_processed.csv"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "preprocessor.pkl"
)


TARGET_COLUMN = "loan_status"


# ============================================================
# LOAD DATA
# ============================================================

def load_data(file_path=RAW_DATA_PATH):
    """
    Load the raw credit risk dataset.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    print(f"Loaded dataset: {df.shape}")

    return df


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(df):
    """
    Perform basic data-quality cleaning.
    """

    df = df.copy()

    print("\n--- DATA CLEANING ---")

    # Remove duplicate records
    duplicates = df.duplicated().sum()

    df = df.drop_duplicates()

    print(f"Duplicates removed: {duplicates}")

    # --------------------------------------------------------
    # Invalid age values
    # --------------------------------------------------------

    invalid_age = (
        (df["person_age"] < 18)
        | (df["person_age"] > 100)
    )

    print(
        f"Invalid age values: {invalid_age.sum()}"
    )

    df.loc[invalid_age, "person_age"] = np.nan

    # --------------------------------------------------------
    # Invalid employment length
    # --------------------------------------------------------

    invalid_emp_length = (
        df["person_emp_length"].notna()
        &
        (
            df["person_emp_length"]
            > df["person_age"]
        )
    )

    print(
        "Invalid employment lengths: "
        f"{invalid_emp_length.sum()}"
    )

    df.loc[
        invalid_emp_length,
        "person_emp_length"
    ] = np.nan

    # --------------------------------------------------------
    # Invalid income
    # --------------------------------------------------------

    invalid_income = (
        df["person_income"] <= 0
    )

    print(
        f"Invalid income values: "
        f"{invalid_income.sum()}"
    )

    df.loc[
        invalid_income,
        "person_income"
    ] = np.nan

    # --------------------------------------------------------
    # Invalid loan amount
    # --------------------------------------------------------

    invalid_loan_amount = (
        df["loan_amnt"] <= 0
    )

    print(
        f"Invalid loan amounts: "
        f"{invalid_loan_amount.sum()}"
    )

    df.loc[
        invalid_loan_amount,
        "loan_amnt"
    ] = np.nan

    # --------------------------------------------------------
    # Invalid interest rate
    # --------------------------------------------------------

    invalid_interest_rate = (
        df["loan_int_rate"] <= 0
    )

    print(
        f"Invalid interest rates: "
        f"{invalid_interest_rate.sum()}"
    )

    df.loc[
        invalid_interest_rate,
        "loan_int_rate"
    ] = np.nan

    # --------------------------------------------------------
    # Invalid loan percent income
    # --------------------------------------------------------

    invalid_loan_percent = (
        df["loan_percent_income"] < 0
    )

    print(
        f"Invalid loan/income ratios: "
        f"{invalid_loan_percent.sum()}"
    )

    df.loc[
        invalid_loan_percent,
        "loan_percent_income"
    ] = np.nan

    return df


# ============================================================
# FEATURE / TARGET SPLIT
# ============================================================

def split_features_target(df):
    """
    Separate input features from target.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "not found."
        )

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    return X, y


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(X):
    """
    Create preprocessing pipeline for
    numerical and categorical features.
    """

    numerical_columns = (
        X.select_dtypes(
            include=[
                "int64",
                "float64",
                "int32",
                "float32"
            ]
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    print("\nNumerical columns:")
    print(numerical_columns)

    print("\nCategorical columns:")
    print(categorical_columns)

    # Numerical pipeline
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    # Categorical pipeline
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    return preprocessor


# ============================================================
# RUN PREPROCESSING
# ============================================================

def run_preprocessing():

    print("=" * 60)
    print("FINWISE CREDIT RISK - PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load raw data
    # --------------------------------------------------------

    df = load_data()

    print(
        f"Original shape: {df.shape}"
    )

    # --------------------------------------------------------
    # 2. Clean data
    # --------------------------------------------------------

    df = clean_data(df)

    print(
        f"Cleaned shape: {df.shape}"
    )

    # --------------------------------------------------------
    # 3. Separate features and target
    # --------------------------------------------------------

    X, y = split_features_target(df)

    print(
        f"\nFeature shape: {X.shape}"
    )

    print(
        f"Target shape: {y.shape}"
    )

    print("\nTarget distribution:")
    print(y.value_counts())

    # --------------------------------------------------------
    # 4. Create preprocessing pipeline
    # --------------------------------------------------------

    preprocessor = create_preprocessor(X)

    # --------------------------------------------------------
    # 5. Fit preprocessing
    # --------------------------------------------------------

    X_processed = (
        preprocessor.fit_transform(X)
    )

    # --------------------------------------------------------
    # 6. Get feature names
    # --------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    processed_df = pd.DataFrame(
        X_processed,
        columns=feature_names,
        index=X.index
    )

    # Add target
    processed_df[TARGET_COLUMN] = (
        y.values
    )

    # --------------------------------------------------------
    # 7. Create directories
    # --------------------------------------------------------

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        PROJECT_ROOT / "models"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 8. Save processed dataset
    # --------------------------------------------------------

    processed_df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    # --------------------------------------------------------
    # 9. Save preprocessor
    # --------------------------------------------------------

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    # --------------------------------------------------------
    # 10. Print final information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)

    print(
        f"Processed dataset: "
        f"{PROCESSED_DATA_PATH}"
    )

    print(
        f"Processed shape: "
        f"{processed_df.shape}"
    )

    print(
        f"Preprocessor saved: "
        f"{PREPROCESSOR_PATH}"
    )

    print(
        f"Generated features: "
        f"{len(feature_names)}"
    )


if __name__ == "__main__":
    run_preprocessing()
