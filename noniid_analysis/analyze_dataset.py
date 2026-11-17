from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = Path("data_cache/raw")
OUT = Path("results/noniid/analysis")
OUT.mkdir(parents=True, exist_ok=True)

NUM = ["step", "amount", "oldbalanceOrg", "newbalanceOrig",
       "oldbalanceDest", "newbalanceDest"]


def load():
    train = pd.read_parquet(RAW / "train.parquet")
    test = pd.read_parquet(RAW / "test.parquet")
    return train, test


def bank_summary(df, split):
    g = df.groupby("BankID")["isFraud"].agg(transactions="size", fraud="sum")
    g["legitimate"] = g["transactions"] - g["fraud"]
    g["fraud_pct"] = 100 * g["fraud"] / g["transactions"]
    g["share_of_total_pct"] = 100 * g["transactions"] / g["transactions"].sum()
    g.insert(0, "split", split)
    return g.reset_index()

def type_analysis(df, split):
    counts = pd.crosstab(df["BankID"], df["type"])
    pct = 100 * counts.div(counts.sum(axis=1), axis=0)
    counts.to_csv(OUT / f"type_counts_{split}.csv")
    pct.to_csv(OUT / f"type_pct_by_bank_{split}.csv")

    fraud = df.groupby("type")["isFraud"].agg(transactions="size", fraud="sum")
    fraud["fraud_pct"] = 100 * fraud["fraud"] / fraud["transactions"]
    fraud.to_csv(OUT / f"fraud_by_type_{split}.csv")

    print(f"\n--- Type mix per bank, % ({split}) ---")
    print(pct.round(2).to_string())
    print(f"\n--- Fraud by type ({split}) ---")
    print(fraud.to_string())

def quality_checks(df, split):
    r = {}
    r["rows"] = len(df)
    r["duplicate_rows"] = int(df.duplicated().sum())
    r["missing_values"] = int(df.isna().sum().sum())
    r["unexpected_types"] = sorted(
        set(df["type"].unique()) - {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
    )
    for c in NUM:
        r[f"negative_{c}"] = int((df[c] < 0).sum())
    r["zero_amount_rows"] = int((df["amount"] == 0).sum())
    r["flagged_fraud_total"] = int(df["isFlaggedFraud"].sum())
    r["flagged_and_fraud"] = int(((df["isFlaggedFraud"] == 1) & (df["isFraud"] == 1)).sum())
    mismatch = (df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]).abs() > 0.01
    r["orig_balance_mismatch_rows"] = int(mismatch.sum())
    banks_per_account = df.groupby("nameOrig")["BankID"].nunique()
    r["nameOrig_unique"] = int(len(banks_per_account))
    r["nameOrig_in_multiple_banks"] = int((banks_per_account > 1).sum())

    out = pd.Series(r, name="value")
    out.to_csv(OUT / f"quality_checks_{split}.csv", header=True)
    print(f"\n--- Data quality ({split}) ---")
    print(out.to_string())

def make_plots(train):
    s = bank_summary(train, "train").set_index("BankID")
    banks = s.index.astype(str)

    # 1. Bank size and fraud rate
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(banks, s["transactions"] / 1e6)
    ax[0].set(xlabel="Bank", ylabel="Transactions (millions)",
              title="Transactions per bank (train)")
    ax[1].bar(banks, s["fraud_pct"], color="tab:red")
    ax[1].set(xlabel="Bank", ylabel="Fraud prevalence (%)",
              title="Fraud prevalence per bank (train)")
    fig.tight_layout()
    fig.savefig(OUT / "bank_size_and_fraud_rate.png", dpi=150)
    plt.close(fig)

    # 2. Transaction type mix per bank
    pct = 100 * pd.crosstab(train["BankID"], train["type"], normalize="index")
    ax = pct.plot(kind="bar", stacked=True, figsize=(8, 5))
    ax.set(xlabel="Bank", ylabel="% of transactions", title="Type mix per bank (train)")
    ax.legend(title="type", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.figure.tight_layout()
    ax.figure.savefig(OUT / "type_mix_per_bank.png", dpi=150)
    plt.close(ax.figure)

    # 3. Amount distribution per bank (log scale, 100k-row sample per bank)
    sample = train.groupby("BankID").sample(n=100_000, random_state=0)
    data = [np.log10(sample.loc[sample["BankID"] == b, "amount"] + 1) for b in range(5)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([str(b) for b in range(5)])
    ax.set(xlabel="Bank", ylabel="log10(amount + 1)",
           title="Transaction amount per bank (train, 100k sample each)")
    fig.tight_layout()
    fig.savefig(OUT / "amount_per_bank_log.png", dpi=150)
    plt.close(fig)

    # 4. Fraud rate by transaction type
    f = train.groupby("type")["isFraud"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(f.index, f.values, color="tab:red")
    ax.set(xlabel="Transaction type", ylabel="Fraud prevalence (%)",
           title="Fraud prevalence by type (train)")
    fig.tight_layout()
    fig.savefig(OUT / "fraud_by_type.png", dpi=150)
    plt.close(fig)



def numeric_analysis(df, split):
    rows = []
    for col in NUM:
        g = df.groupby("BankID")[col].describe(percentiles=[0.5, 0.99])
        g.insert(0, "feature", col)
        rows.append(g.reset_index())
    out = pd.concat(rows)
    out.to_csv(OUT / f"numeric_stats_{split}.csv", index=False)

    print(f"\n--- Amount per bank ({split}) ---")
    print(out[out["feature"] == "amount"].round(1).to_string(index=False))


def main():
    train, test = load()
    summary = pd.concat([bank_summary(train, "train"), bank_summary(test, "test")])
    summary.to_csv(OUT / "bank_summary.csv", index=False)
    print(summary.to_string(index=False))

    type_analysis(train, "train")
    type_analysis(test, "test")
    numeric_analysis(train, "train")
    numeric_analysis(test, "test")
    quality_checks(train, "train")
    quality_checks(test, "test")
    make_plots(train)


if __name__ == "__main__":
    main()