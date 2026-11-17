from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_cache" / "raw"
N_BANKS = 5


def load_train():
    return pd.read_parquet(RAW / "train.parquet")


def original(train, seed=0):
    """Baseline: the dataset's own BankID."""
    return train["BankID"].to_numpy().copy()


def iid(train, seed=0):
    """Random, stratified by (isFraud, type), keeping the original bank sizes."""
    rng = np.random.default_rng(seed)
    shares = train["BankID"].value_counts(normalize=True).sort_index().to_numpy()
    cum = np.cumsum(shares)
    new = np.empty(len(train), dtype=np.int64)
    for _, idx in train.groupby(["isFraud", "type"]).indices.items():
        idx = rng.permutation(idx)
        cuts = np.round(cum[:-1] * len(idx)).astype(int)
        for bank, part in enumerate(np.split(idx, cuts)):
            new[part] = bank
    return new

def fraud_skew(train, seed=0, fraud_share=(0.50, 0.25, 0.15, 0.07, 0.03)):
    """Keep original bank sizes; spread fraud cases unevenly across banks."""
    rng = np.random.default_rng(seed)
    sizes = train["BankID"].value_counts().sort_index().to_numpy()
    is_fraud = train["isFraud"].to_numpy() == 1
    fraud_idx = rng.permutation(np.flatnonzero(is_fraud))
    legit_idx = rng.permutation(np.flatnonzero(~is_fraud))

    share = np.asarray(fraud_share, dtype=float)
    share /= share.sum()
    cuts_f = np.round(np.cumsum(share)[:-1] * len(fraud_idx)).astype(int)
    fraud_parts = np.split(fraud_idx, cuts_f)

    capacity = sizes - np.array([len(p) for p in fraud_parts])
    legit_parts = np.split(legit_idx, np.cumsum(capacity)[:-1])

    new = np.empty(len(train), dtype=np.int64)
    for b in range(N_BANKS):
        new[fraud_parts[b]] = b
        new[legit_parts[b]] = b
    return new

def type_skew(train, seed=0, alpha=0.5, min_fraud=100, max_tries=200):
    """Split each transaction type across banks with Dirichlet(alpha) proportions."""
    types = train["type"].to_numpy()
    fraud = train["isFraud"].to_numpy() == 1
    for attempt in range(max_tries):
        rng = np.random.default_rng([seed, attempt])
        new = np.empty(len(train), dtype=np.int64)
        for t in np.unique(types):
            idx = rng.permutation(np.flatnonzero(types == t))
            p = rng.dirichlet(alpha * np.ones(N_BANKS))
            cuts = np.round(np.cumsum(p)[:-1] * len(idx)).astype(int)
            for b, part in enumerate(np.split(idx, cuts)):
                new[part] = b
        fraud_per_bank = np.bincount(new[fraud], minlength=N_BANKS)
        if fraud_per_bank.min() >= min_fraud:
            return new
    raise ValueError("No draw gave every bank enough fraud; raise alpha or lower min_fraud")


def summarize(train, assignment, name):
    df = train.assign(BankID=assignment)
    g = df.groupby("BankID")["isFraud"].agg(transactions="size", fraud="sum")
    g["fraud_pct"] = 100 * g["fraud"] / g["transactions"]
    types = 100 * pd.crosstab(df["BankID"], df["type"], normalize="index")
    print(f"\n=== {name} ===")
    print(g.round(4).to_string())
    print(types.round(2).to_string())


if __name__ == "__main__":
    train = load_train()
    summarize(train, original(train), "original")
    summarize(train, iid(train, seed=0), "iid (seed 0)")
    summarize(train, fraud_skew(train, seed=0), "fraud skew (seed 0)")
    summarize(train, type_skew(train, seed=0, alpha=0.5), "type skew alpha=0.5 (seed 0)")