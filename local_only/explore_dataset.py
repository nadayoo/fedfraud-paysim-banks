from datasets import load_dataset
import pandas as pd

print("Loading PaySim federated fraud dataset...")
dataset = load_dataset("flwrlabs/fed-fraud-paysim-banks")

print("\nAvailable splits:")
for split_name, split_data in dataset.items():
    print(f"{split_name}: {len(split_data):,} rows")

train = dataset["train"]
test = dataset["test"]

print("\nDataset columns:")
print(train.column_names)

print("\nFirst training example:")
print(train[0])

# Select only the columns needed for this initial analysis.
# This avoids loading the entire dataset into pandas.
train_df = train.select_columns(["BankID", "isFraud"]).to_pandas()
test_df = test.select_columns(["BankID", "isFraud"]).to_pandas()

print("\nTRAINING DATA — BANK SUMMARY")
summary = (
    train_df.groupby("BankID")
    .agg(
        total_transactions=("isFraud", "size"),
        fraud_cases=("isFraud", "sum"),
        fraud_rate=("isFraud", "mean"),
    )
)

summary["normal_cases"] = (
    summary["total_transactions"] - summary["fraud_cases"]
)

print(summary.to_string())

print("\nTEST DATA — BANK SUMMARY")
test_summary = (
    test_df.groupby("BankID")
    .agg(
        total_transactions=("isFraud", "size"),
        fraud_cases=("isFraud", "sum"),
        fraud_rate=("isFraud", "mean"),
    )
)

test_summary["normal_cases"] = (
    test_summary["total_transactions"] - test_summary["fraud_cases"]
)

print(test_summary.to_string())

summary.to_csv("results/local_only/train_bank_summary.csv")
test_summary.to_csv("results/local_only/test_bank_summary.csv")

print("\nSaved summaries in results/local_only/")
print("\nDataset exploration completed.")
