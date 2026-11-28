import numpy as np
from flwr_datasets import FederatedDataset
from fed_fraud.lookup_partitioner import LookupPartitioner

for stem in ["original_seed0", "iid_seed0", "fraud_skew_seed0", "type_skew_a0.5_seed0"]:
    fds = FederatedDataset(
        dataset="flwrlabs/fed-fraud-paysim-banks",
        partitioners={"train": LookupPartitioner("lookup_" + stem, 5)},
    )
    rows, fraud = [], []
    for b in range(5):
        p = fds.load_partition(b, "train")
        rows.append(len(p))
        fraud.append(int(np.sum(p["isFraud"])))
        if stem.startswith("original"):
            print("original: bank", b, "rows all have BankID", b, ":",
                  np.unique(p["BankID"]).tolist() == [b])
    print(stem, "| rows:", rows, "| fraud:", fraud)