from pathlib import Path
import pandas as pd


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "credit_risk_dataset.csv"


def load_raw_data(file_path=RAW_DATA_PATH):
    """
    Load the raw credit risk dataset.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    print("Dataset loaded successfully.")
    print(f"File: {file_path}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


def validate_dataset(df):
    """
    Perform basic dataset validation.
    """

    required_columns = [
        "person_age",
        "person_income",
        "person_home_ownership",
        "person_emp_length",
        "loan_intent",
        "loan_grade",
        "loan_amnt",
        "loan_int_rate",
        "loan_status",
        "loan_percent_income",
        "cb_person_default_on_file",
        "cb_person_cred_hist_length"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Dataset validation passed.")

    return True


def save_raw_summary(df):
    """
    Save a basic summary of the raw dataset.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_path = (
        PROCESSED_DATA_DIR /
        "raw_data_summary.txt"
    )

    with open(summary_path, "w") as file:

        file.write("FINWISE CREDIT RISK DATASET SUMMARY\n")
        file.write("=" * 50 + "\n\n")

        file.write(
            f"Rows: {df.shape[0]}\n"
        )

        file.write(
            f"Columns: {df.shape[1]}\n\n"
        )

        file.write("Columns:\n")

        for column in df.columns:
            file.write(f"- {column}\n")

        file.write("\nMissing Values:\n")
        file.write(
            df.isnull().sum().to_string()
        )

        file.write("\n\nDuplicate Rows:\n")
        file.write(
            str(df.duplicated().sum())
        )

    print(
        f"Dataset summary saved to: {summary_path}"
    )


def run_ingestion():

    print("=" * 60)
    print("FINWISE CREDIT RISK - DATA INGESTION")
    print("=" * 60)

    # Load data
    df = load_raw_data()

    # Validate data
    validate_dataset(df)

    # Save dataset summary
    save_raw_summary(df)

    print("\nData ingestion completed successfully.")

    return df


if __name__ == "__main__":
    run_ingestion()
