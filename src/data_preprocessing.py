import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


TARGET_COLUMN = "loan_status"


def load_data(file_path):
    """Load the raw credit risk dataset."""
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    """Perform basic data-quality cleaning."""

    df = df.copy()

    # Replace invalid employment lengths with missing values
    invalid_emp = (
        df["person_emp_length"].notna()
        & (df["person_emp_length"] > df["person_age"])
    )

    df.loc[invalid_emp, "person_emp_length"] = np.nan

    # Remove duplicate rows
    df = df.drop_duplicates()

    return df


def split_features_target(df):
    """Separate features and target."""

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def create_preprocessor(X):
    """Create preprocessing pipeline for numerical and categorical features."""

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
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
                "num",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    return preprocessor


if __name__ == "__main__":

    input_path = "../data/raw/credit_risk_dataset.csv"

    df = load_data(input_path)

    print("Original dataset shape:", df.shape)

    df = clean_data(df)

    print("Cleaned dataset shape:", df.shape)

    X, y = split_features_target(df)

    print("Features:", X.shape)
    print("Target:", y.shape)

    print("\nTarget distribution:")
    print(y.value_counts())
