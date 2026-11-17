import pandas as pd
from partitions import ROOT, load_train, original, iid, fraud_skew, type_skew

OUT = ROOT / "results" / "noniid" / "partitions"
OUT.mkdir(parents=True, exist_ok=True)


def describe(train, assignment, name):
    df = train.assign(BankID=assignment)
    g = df.groupby("BankID")["isFraud"].agg(transactions="size", fraud="sum")
    g["fraud_pct"] = 100 * g["fraud"] / g["transactions"]
    g["share_pct"] = 100 * g["transactions"] / g["transactions"].sum()
    mix = pd.crosstab(df["BankID"], df["type"], normalize="index")
    overall = df["type"].value_counts(normalize=True).reindex(mix.columns)
    # total variation distance between a bank's type mix and the overall mix (0 to 1)
    g["type_tv_distance"] = 0.5 * (mix - overall).abs().sum(axis=1)
    g.insert(0, "partition", name)
    return g.reset_index()


def summarize(g):
    return {
        "partition": g["partition"].iloc[0],
        "size_ratio_max_min": g["transactions"].max() / g["transactions"].min(),
        "fraud_pct_min": g["fraud_pct"].min(),
        "fraud_pct_max": g["fraud_pct"].max(),
        "fraud_rate_ratio_max_min": g["fraud_pct"].max() / g["fraud_pct"].min(),
        "min_fraud_cases": g["fraud"].min(),
        "mean_type_tv": g["type_tv_distance"].mean(),
        "max_type_tv": g["type_tv_distance"].max(),
    }


def main():
    train = load_train()
    parts = {
        "original": original(train),
        "iid": iid(train, seed=0),
        "fraud_skew": fraud_skew(train, seed=0),
        "type_skew_a0.5": type_skew(train, seed=0, alpha=0.5),
    }
    per_bank = pd.concat([describe(train, a, n) for n, a in parts.items()])
    per_bank.to_csv(OUT / "partition_per_bank.csv", index=False)
    summ = pd.DataFrame([summarize(per_bank[per_bank["partition"] == n]) for n in parts])
    summ.to_csv(OUT / "partition_summary.csv", index=False)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()