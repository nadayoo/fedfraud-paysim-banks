from pathlib import Path
import pandas as pd
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner

ROOT = Path(__file__).resolve().parents[1]
raw = pd.read_parquet(ROOT / "data_cache" / "raw" / "train.parquet")

fds = FederatedDataset(
    dataset="flwrlabs/fed-fraud-paysim-banks",
    partitioners={"train": NaturalIdPartitioner(partition_by="BankID")},
)
part0 = fds.load_partition(0, "train").to_pandas()
raw0 = raw[raw["BankID"] == 0]

print("rows (flower, raw):", len(part0), len(raw0))
print("same order as raw:", part0["nameOrig"].head(1000).tolist() == raw0["nameOrig"].head(1000).tolist())
print(part0[["step", "type", "amount"]].head(3))
print(raw0[["step", "type", "amount"]].head(3))
