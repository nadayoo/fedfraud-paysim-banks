from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
raw = pd.read_parquet(ROOT / "data_cache" / "raw" / "train.parquet")

key_cols = [c for c in raw.columns if c not in ("BankID", "isFraud")]
print("key columns:", key_cols)
print("rows:", len(raw))
print("duplicate keys:", int(raw.duplicated(subset=key_cols).sum()))

h = pd.util.hash_pandas_object(raw[key_cols], index=False)
print("duplicate hashes:", int(h.duplicated().sum()))