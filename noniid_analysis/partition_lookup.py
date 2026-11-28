"""Build row-hash -> bank lookups, so partitions match rows by content, not position."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from partitions import PARTITIONS, ROOT, load_train

KEY_COLS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg",
            "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
            "isFlaggedFraud"]
OUT = ROOT / "data_cache" / "partitions"


def row_hashes(df):
    return pd.util.hash_pandas_object(df[KEY_COLS], index=False).to_numpy(np.uint64)


def main(seed=0):
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_train()
    h = row_hashes(train)
    order = np.argsort(h)
    for name, fn in PARTITIONS.items():
        banks = fn(train, seed).astype(np.int8)
        path = OUT / f"lookup_{name}_seed{seed}.npz"
        np.savez(path, hashes=h[order], banks=banks[order])
        print(name, "saved ->", path.name, "| rows per bank:", np.bincount(banks).tolist())


if __name__ == "__main__":
    main()