import pandas as pd
from pathlib import Path

from src.config import load_config


def preprocess(raw_path=None, processed_dir=None):
    config = load_config()
    data_config = config["data"]

    # -------------------------
    # Configuration
    # -------------------------
    # Allow Airflow to override paths (isolated airflow_data/)
    # Defaults to local data/ for `uv run python -m src.preprocessing`
    if raw_path is None:
        raw_path = data_config["raw_path"]

    if processed_dir is None:
        processed_dir = Path(data_config["processed_dir"])
    else:
        processed_dir = Path(processed_dir)

    columns = data_config["columns"]

    test_size = data_config["split"]["test_size"]
    random_state = data_config["split"]["random_state"]

    train_filename = data_config["output"]["train"]
    test_filename = data_config["output"]["test"]

    # -------------------------
    # Create output directory
    # -------------------------
    processed_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(
        raw_path,
        usecols=columns
    )

    print(f"Original shape: {df.shape}")

    # -------------------------
    # Clean column names
    # -------------------------
    df.columns = df.columns.str.strip()

    # -------------------------
    # Remove whitespace from string values
    # -------------------------
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()

    # -------------------------
    # Convert all columns to numeric
    #
    # Invalid values such as:
    # "abc", "hello", "" -> NaN
    # -------------------------
    for column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # -------------------------
    # Remove rows containing NaN
    # -------------------------
    before_drop = len(df)

    df = df.dropna()

    removed_rows = before_drop - len(df)

    print(f"Removed rows: {removed_rows}")
    print(f"Clean shape: {df.shape}")

    # -------------------------
    # Make sure every column is numeric
    # -------------------------
    df = df.select_dtypes(include="number")

    # -------------------------
    # Split data
    # -------------------------
    train_df = df.sample(
        frac=1 - test_size,
        random_state=random_state
    )

    test_df = df.drop(train_df.index)

    # -------------------------
    # Save as Parquet
    # -------------------------
    train_df.to_parquet(
        processed_dir / train_filename,
        index=False
    )

    test_df.to_parquet(
        processed_dir / test_filename,
        index=False
    )

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Saved to: {processed_dir}")


if __name__ == "__main__":
    preprocess()