"""Local-only baseline: one independent fraud detection model per bank.

Run from the repository root:
    python -m local_only.train --banks 0 --seeds 0 --epochs 1 --limit-train-rows 20000

Full experiment:
    python -m local_only.train --banks 0 1 2 3 4 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score

from common.data import INPUT_DIM, load_or_build, make_loader
from common.metrics import (
    best_f1_threshold,
    compute_metrics,
    sigmoid,
    threshold_metrics,
)
from common.model import Net, count_parameters
from common.seeding import set_seed


def cosine_annealing(cur, total, lr_max, lr_min):
    return lr_min + 0.5 * (lr_max - lr_min) * (
        1 + math.cos(math.pi * cur / total)
    )


@torch.no_grad()
def predict_logits(model, X, device, batch=65536):
    model.eval()
    if len(X) == 0:
        return np.array([], dtype=np.float32)

    outputs = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        outputs.append(model(xb).cpu())

    return torch.cat(outputs).numpy()


def bce(logits, y):
    return float(
        F.binary_cross_entropy_with_logits(
            torch.from_numpy(logits),
            torch.from_numpy(y),
        )
    )

def scale_engineered_features(data):
    """Standardize engineered features (columns 6-9) using training data only."""
    data = dict(data)

    feature_slice = slice(6, 10)

    train = data["X_train"]
    means = train[:, feature_slice].mean(axis=0)
    stds = train[:, feature_slice].std(axis=0)

    stds[stds < 1e-12] = 1.0

    for split in ("train", "val", "test"):
        key = f"X_{split}"
        data[key] = data[key].copy()
        data[key][:, feature_slice] = (
            data[key][:, feature_slice] - means
        ) / stds

    print("\nExperimental engineered-feature scaling:")
    print("Means:", means)
    print("Standard deviations:", stds)

    return data

def plot_results(history, test_metrics, bank_id, seed, out_dir):
    epochs = [item["epoch"] for item in history]

    plt.figure(figsize=(5, 3.5))
    plt.plot(
        epochs,
        [item["train_loss_natural"] for item in history],
        label="Train (natural distribution)",
    )
    plt.plot(
        epochs,
        [item["val_loss"] for item in history],
        label="Validation",
    )
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.legend()
    plt.title(f"Bank {bank_id}, seed {seed}")
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close()

    cm = np.array([
        [test_metrics["tn"], test_metrics["fp"]],
        [test_metrics["fn"], test_metrics["tp"]],
    ])

    plt.figure(figsize=(3.8, 3.5))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
            )
    plt.xticks([0, 1], ["Legitimate", "Fraud"])
    plt.yticks([0, 1], ["Legitimate", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Bank {bank_id}, threshold={test_metrics['threshold']:.3f}")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def run_bank_seed(cfg, data, bank_id, seed, out_dir, limit_train_rows=None):
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select only this bank's transactions.
    train_mask = data["bank_train"] == bank_id
    val_mask = data["bank_val"] == bank_id
    test_mask = data["bank_test"] == bank_id

    Xtr = data["X_train"][train_mask]
    ytr = data["y_train"][train_mask]
    Xval = data["X_val"][val_mask]
    yval = data["y_val"][val_mask]
    Xtest = data["X_test"][test_mask]
    ytest = data["y_test"][test_mask]

    if len(Xtr) == 0 or len(Xval) == 0 or len(Xtest) == 0:
        raise ValueError(f"Bank {bank_id} has an empty train, validation, or test split.")

    # Optional small subset for smoke testing only.
    if limit_train_rows is not None and len(Xtr) > limit_train_rows:
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(Xtr), limit_train_rows, replace=False)
        Xtr = Xtr[selected]
        ytr = ytr[selected]

    print(
        f"\nBank {bank_id} | seed {seed} | "
        f"train={len(Xtr):,} ({int(ytr.sum()):,} frauds) | "
        f"val={len(Xval):,} ({int(yval.sum()):,} frauds) | "
        f"test={len(Xtest):,} ({int(ytest.sum()):,} frauds)"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = cfg["model"]

    model = Net(
        INPUT_DIM,
        m["hidden_dim_1"],
        m["hidden_dim_2"],
        m["dropout"],
        m["use_layernorm"],
    ).to(device)

    mode = cfg["imbalance"]
    pos_weight = None

    if mode in ("pos_weight", "both"):
        n_pos = float(ytr.sum())
        if n_pos == 0:
            raise ValueError(f"Bank {bank_id} training subset contains no fraud examples.")
        pos_weight = torch.tensor(
            (len(ytr) - n_pos) / n_pos,
            device=device,
        )

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loader = make_loader(
        Xtr,
        ytr,
        cfg["batch_size"],
        balanced=mode in ("sampler", "both"),
        shuffle=True,
        seed=seed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr_max"],
        weight_decay=cfg["weight_decay"],
    )

    rng = np.random.default_rng(seed)
    subset = rng.choice(
        len(ytr),
        min(cfg["train_eval_subset"], len(ytr)),
        replace=False,
    )

    history = []
    best_pr = -1.0
    best_state = None
    best_epoch = -1
    bad_epochs = 0

    start_time = time.time()

    for epoch in range(cfg["epochs"]):
        lr = cosine_annealing(
            epoch,
            cfg["epochs"],
            cfg["lr_max"],
            cfg["lr_min"],
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        running_loss = 0.0
        batch_count = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["grad_clip"],
            )

            optimizer.step()
            running_loss += loss.item()
            batch_count += 1

        val_logits = predict_logits(model, Xval, device)
        val_probs = sigmoid(val_logits)
        val_pr = float(average_precision_score(yval, val_probs))

        train_logits = predict_logits(model, Xtr[subset], device)

        record = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss_sampled": running_loss / max(batch_count, 1),
            "train_loss_natural": bce(train_logits, ytr[subset]),
            "val_loss": bce(val_logits, yval),
            "val_pr_auc": val_pr,
        }
        history.append(record)

        print(
            f"[bank {bank_id} | seed {seed}] "
            f"epoch {epoch + 1}/{cfg['epochs']} "
            f"train={record['train_loss_natural']:.4f} "
            f"val={record['val_loss']:.4f} "
            f"val_PR-AUC={val_pr:.4f}"
        )

        if best_state is None or val_pr > best_pr:
            best_pr = val_pr
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    train_time = time.time() - start_time

    # Restore the best validation checkpoint.
    model.load_state_dict(best_state)

    # Choose the classification threshold using validation data only.
    val_probs = sigmoid(predict_logits(model, Xval, device))
    chosen_threshold = best_f1_threshold(val_probs, yval)

    # Evaluate only on this bank's held-out test data.
    test_probs = sigmoid(predict_logits(model, Xtest, device))
    results = compute_metrics(test_probs, ytest, cfg["thresholds"])

    results["at_val_best_f1_threshold"] = threshold_metrics(
        test_probs,
        ytest,
        chosen_threshold,
    )

    results.update({
        "bank_id": bank_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "val_pr_auc": best_pr,
        "chosen_threshold": chosen_threshold,
        "train_time_sec": train_time,
        "n_params": count_parameters(model),
        "imbalance": mode,
        "epochs_run": len(history),
        "train_rows": len(Xtr),
        "val_rows": len(Xval),
        "test_rows": len(Xtest),
        "device": str(device),
        "smoke_test_limit": limit_train_rows,
    })

    torch.save(best_state, out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    plot_results(
        history,
        results["at_val_best_f1_threshold"],
        bank_id,
        seed,
        out_dir,
    )

    print(
        f"Bank {bank_id} finished | "
        f"test PR-AUC={results['pr_auc']:.4f} | "
        f"test ROC-AUC={results['roc_auc']:.4f}"
    )

    return results


def aggregate(results):
    def mean_std(values):
        values = np.asarray(values, dtype=float)
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }

    return {
        "seeds": [r["seed"] for r in results],
        "pr_auc": mean_std([r["pr_auc"] for r in results]),
        "roc_auc": mean_std([r["roc_auc"] for r in results]),
        "f1_at_val_best_f1_threshold": mean_std([
            r["at_val_best_f1_threshold"]["f1"] for r in results
        ]),
        "recall_at_val_best_f1_threshold": mean_std([
            r["at_val_best_f1_threshold"]["recall"] for r in results
        ]),
        "precision_at_val_best_f1_threshold": mean_std([
            r["at_val_best_f1_threshold"]["precision"] for r in results
        ]),
        "train_time_sec": mean_std([r["train_time_sec"] for r in results]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="common/config_base.yaml")
    parser.add_argument("--config", default="local_only/configs/local_only.yaml")
    parser.add_argument("--banks", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train-rows", type=int, default=None)
    parser.add_argument("--imbalance", default=None)
    parser.add_argument("--scale-engineered", action="store_true")
    args = parser.parse_args()

    with open(args.base) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.config) as f:
        experiment_cfg = yaml.safe_load(f)

    cfg = {**base_cfg, **experiment_cfg}

    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.imbalance is not None:
        cfg["imbalance"] = args.imbalance

    seeds = args.seeds if args.seeds is not None else cfg["seeds"]

    print("Loading shared preprocessed dataset...")
    data = load_or_build(cfg["cache_path"], cfg["val_fraction"])

    if args.scale_engineered:
        data = scale_engineered_features(data)
        cfg["out_dir"] = "results/local_only_scaled"

    print("Dataset shapes:", {
        key: data[key].shape
        for key in ("X_train", "X_val", "X_test")
    })

    all_results = {}

    for bank_id in args.banks:
        bank_results = []

        for seed in seeds:
            output_dir = (
                Path(cfg["out_dir"])
                / f"bank_{bank_id}"
                / f"seed_{seed}"
            )

            result = run_bank_seed(
                cfg,
                data,
                bank_id,
                seed,
                output_dir,
                args.limit_train_rows,
            )
            bank_results.append(result)

        summary = aggregate(bank_results)
        all_results[str(bank_id)] = summary

        bank_dir = Path(cfg["out_dir"]) / f"bank_{bank_id}"
        bank_dir.mkdir(parents=True, exist_ok=True)
        (bank_dir / "summary.json").write_text(
            json.dumps({"config": cfg, **summary}, indent=2)
        )

        print(f"\nBank {bank_id} summary:")
        print(json.dumps(summary, indent=2))

    root = Path(cfg["out_dir"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary_all_banks.json").write_text(
        json.dumps({"config": cfg, "banks": all_results}, indent=2)
    )


if __name__ == "__main__":
    main()
