from datasets import load_dataset
import pandas as pd

dataset = load_dataset("flwrlabs/fed-fraud-paysim-banks")

train = dataset["train"].to_pandas()

print("\nDATA TYPES")
print(train.dtypes)

print("\nMISSING VALUES")
print(train.isnull().sum())

print("\nTRANSACTION TYPES")
print(train["type"].value_counts())

print("\nTRANSACTION TYPES PER BANK")
print(pd.crosstab(train["BankID"], train["type"]))

print("\nFRAUD RATE BY TRANSACTION TYPE")
print(train.groupby("type")["isFraud"].agg(["count", "sum", "mean"]))

print("\nFLAGGED FRAUD ANALYSIS")
print(pd.crosstab(train["isFlaggedFraud"], train["isFraud"]))

print("\nFLAGGED FRAUD BY BANK")
print(pd.crosstab(train["BankID"], train["isFlaggedFraud"]))

print("\nDUPLICATE ROWS")
print(train.duplicated().sum())

print("\nFEATURE ANALYSIS COMPLETED")
