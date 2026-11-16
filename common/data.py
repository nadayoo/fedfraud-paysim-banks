"""Shared data pipeline for ALL experiments (import it, don't copy it).

Reproduces the Flower Hub fed-fin-fraud preprocessing (task.py, v1.0.0):
  6 standardized numeric cols + 4 engineered balance features (NOT standardized,
  as in the reference code) + 5 one-hot transaction types = 15 features.
Dropped: nameOrig, nameDest (identifiers), isFlaggedFraud.

Differences from the reference, on purpose:
  * a validation split (grouped by nameOrig, so no account is in train AND val)
  * scaler statistics are fitted on the training rows only
  * vectorised with pandas/numpy instead of a per-row Python loop (same values)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

HF_DATASET_ID = "flwrlabs/fed-fraud-paysim-banks"
NUMERIC_COLUMNS = ["step", "amount", "oldbalanceOrg", "newbalanceOrig",
                   "oldbalanceDest", "newbalanceDest"]
TYPE_VALUES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
LABEL_COL, BANK_COL, GROUP_COL = "isFraud", "BankID", "nameOrig"
RAW_COLUMNS = [*NUMERIC_COLUMNS, "type", LABEL_COL, BANK_COL, GROUP_COL]
INPUT_DIM = len(NUMERIC_COLUMNS) + 4 + len(TYPE_VALUES)  # 15


def val_mask(groups: pd.Series, val_fraction: float) -> np.ndarray:
    """Deterministic account-level split: hash(nameOrig) decides train vs val."""
    h = pd.util.hash_pandas_object(groups, index=False).to_numpy()
    return (h % np.uint64(1000)) < np.uint64(round(val_fraction * 1000))


def fit_scaler(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = df[NUMERIC_COLUMNS].fillna(0).to_numpy(np.float64)
    means, stds = x.mean(axis=0), x.std(axis=0)
    stds[stds <= 1e-12] = 1.0
    return means, stds


def make_features(df: pd.DataFrame, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    num = df[NUMERIC_COLUMNS].fillna(0).to_numpy(np.float64)
    z = (num - means) / stds
    old_org, new_org = df["oldbalanceOrg"].fillna(0).to_numpy(), df["newbalanceOrig"].fillna(0).to_numpy()
    old_dst, new_dst = df["oldbalanceDest"].fillna(0).to_numpy(), df["newbalanceDest"].fillna(0).to_numpy()
    amount = df["amount"].fillna(0).to_numpy()
    origin_delta = old_org - new_org
    dest_delta = new_dst - old_dst
    eng = np.stack([origin_delta, dest_delta, origin_delta - amount, amount - dest_delta], axis=1)

    codes = pd.Categorical(df["type"].astype(str).str.strip(), categories=TYPE_VALUES).codes
    onehot = np.zeros((len(df), len(TYPE_VALUES)), dtype=np.float64)
    ok = codes >= 0
    onehot[np.flatnonzero(ok), codes[ok]] = 1.0
    return np.concatenate([z, eng, onehot], axis=1).astype(np.float32)


def build_arrays(train_df: pd.DataFrame, test_df: pd.DataFrame, val_fraction: float = 0.1) -> dict:
    is_val = val_mask(train_df[GROUP_COL], val_fraction)
    parts = {"train": train_df[~is_val], "val": train_df[is_val], "test": test_df}
    means, stds = fit_scaler(parts["train"])
    out = {"scaler_means": means, "scaler_stds": stds}
    for name, df in parts.items():
        out[f"X_{name}"] = make_features(df, means, stds)
        out[f"y_{name}"] = df[LABEL_COL].to_numpy(np.float32)
        out[f"bank_{name}"] = df[BANK_COL].to_numpy(np.int64)
    return out


def load_or_build(cache_path: str | None = None, val_fraction: float = 0.1) -> dict:
    """Download from Hugging Face once, preprocess, cache as .npz (gitignored)."""
    if cache_path and Path(cache_path).exists():
        with np.load(cache_path) as f:
            return {k: f[k] for k in f.files}
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET_ID)
    train_df = ds["train"].select_columns(RAW_COLUMNS).to_pandas()
    test_df = ds["test"].select_columns(RAW_COLUMNS).to_pandas()
    data = build_arrays(train_df, test_df, val_fraction)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **data)
    return data


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, *, balanced: bool = False,
                shuffle: bool = False, seed: int = 0) -> DataLoader:
    """balanced=True -> WeightedRandomSampler with 1/class_count weights (as Flower Hub)."""
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    g = torch.Generator().manual_seed(seed)
    if balanced:
        yi = y.astype(np.int64)
        counts = np.bincount(yi, minlength=2).astype(np.float64)
        w = np.zeros(2)
        w[counts > 0] = 1.0 / counts[counts > 0]
        sampler = WeightedRandomSampler(torch.from_numpy(w[yi]), num_samples=len(y),
                                        replacement=True, generator=g)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)
