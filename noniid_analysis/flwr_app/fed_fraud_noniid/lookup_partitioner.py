"""Partitioner driven by a precomputed lookup (row hash -> bank).

Rows are matched by content, so Flower's shuffle does not matter.
Lookups are built by noniid_analysis/partition_lookup.py.
Rows are hashed in chunks to keep memory low.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from flwr_datasets.partitioner import Partitioner

KEY_COLS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg",
            "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
            "isFlaggedFraud"]
CHUNK = 200_000


def lookup_dir() -> Path:
    env = os.environ.get("FEDFRAUD_LOOKUP_DIR")
    if env:
        return Path(env)
    return Path.home() / "fedfraud_lookups"


class LookupPartitioner(Partitioner):
    def __init__(self, lookup_name: str, num_partitions: int = 5) -> None:
        super().__init__()
        self._lookup_name = lookup_name
        self._num_partitions = num_partitions
        self._assignment: np.ndarray | None = None

    @property
    def num_partitions(self) -> int:
        return self._num_partitions

    def _build(self) -> None:
        path = lookup_dir() / f"{self._lookup_name}.npz"
        with np.load(path) as f:
            hashes, banks = f["hashes"], f["banks"]
        ds = self.dataset.select_columns(KEY_COLS)
        n = len(ds)
        if n != len(hashes):
            raise ValueError(f"Row count mismatch: dataset {n}, lookup {len(hashes)}")
        assignment = np.empty(n, dtype=np.int8)
        missing = 0
        for start in range(0, n, CHUNK):
            stop = min(start + CHUNK, n)
            chunk = ds.select(range(start, stop)).to_pandas()
            h = pd.util.hash_pandas_object(chunk[KEY_COLS], index=False).to_numpy(np.uint64)
            pos = np.minimum(np.searchsorted(hashes, h), len(hashes) - 1)
            missing += int((hashes[pos] != h).sum())
            assignment[start:stop] = banks[pos]
            del chunk, h, pos
        if missing:
            raise ValueError(f"{missing} rows not found in lookup {self._lookup_name}")
        self._assignment = assignment

    def load_partition(self, partition_id: int) -> Dataset:
        if self._assignment is None:
            self._build()
        idx = np.flatnonzero(self._assignment == partition_id)
        return self.dataset.select(idx)
