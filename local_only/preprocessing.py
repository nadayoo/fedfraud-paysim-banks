from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
import os

# Configuration
DATASET_NAME = "flwrlabs/fed-fraud-paysim-banks"
TARGET = "isFraud"
BANK_IDS = [0, 1, 2, 3, 4]

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

CATEGORICAL_FEATURES = ["type"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_bank_data():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)

    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    return train_df, test_df


def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def prepare_bank(bank_id, train_df, test_df):
    print(f"\nPreparing Bank {bank_id}...")

    bank_train = train_df[train_df["BankID"] == bank_id].copy()
    bank_test = test_df[test_df["BankID"] == bank_id].copy()

    X = bank_train[FEATURES]
    y = bank_train[TARGET].astype(np.float32)

    X_test = bank_test[FEATURES]
    y_test = bank_test[TARGET].astype(np.float32)

    # Split training data into training and validation.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Fit preprocessing ONLY on training data.
    preprocessor = create_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Convert to dense float32 arrays for PyTorch.
    X_train_processed = np.asarray(
        X_train_processed, dtype=np.float32
    )
    X_val_processed = np.asarray(
        X_val_processed, dtype=np.float32
    )
    X_test_processed = np.asarray(
        X_test_processed, dtype=np.float32
    )

    output_dir = f"results/local_only/bank_{bank_id}"
    os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        f"{output_dir}/processed_data.npz",
        X_train=X_train_processed,
        y_train=y_train.to_numpy(),
        X_val=X_val_processed,
        y_val=y_val.to_numpy(),
        X_test=X_test_processed,
        y_test=y_test.to_numpy(),
    )

    joblib.dump(
        preprocessor,
        f"{output_dir}/preprocessor.joblib",
    )

    print(f"Training samples:   {len(y_train):,}")
    print(f"Validation samples: {len(y_val):,}")
    print(f"Test samples:       {len(y_test):,}")
    print(f"Input features:     {X_train_processed.shape[1]}")
    print(f"Training frauds:    {int(y_train.sum()):,}")
    print(f"Validation frauds:  {int(y_val.sum()):,}")
    print(f"Test frauds:        {int(y_test.sum()):,}")

    return X_train_processed.shape[1]


def main():
    train_df, test_df = load_bank_data()

    print("\nDataset loaded successfully.")
    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows:  {len(test_df):,}")

    for bank_id in BANK_IDS:
        prepare_bank(bank_id, train_df, test_df)

    print("\nAll five banks have been preprocessed successfully.")


if __name__ == "__main__":
    main()
