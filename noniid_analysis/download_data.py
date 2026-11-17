from pathlib import Path
from datasets import load_dataset

out = Path("data_cache/raw")
out.mkdir(parents=True, exist_ok=True)

ds = load_dataset("flwrlabs/fed-fraud-paysim-banks")
print(ds)

for split in ds:
    path = out / f"{split}.parquet"
    ds[split].to_parquet(str(path))
    print("saved", path, len(ds[split]), "rows")