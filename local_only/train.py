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

from common.data import load_or_build, make_loader
from common.metrics import (
    best_f1_threshold,
    compute_metrics,
    sigmoid,
    threshold_metrics,
)
from common.model import Net, count_parameters
from common.seeding import set_seed
from local_only.experiment_utils import (
    apply_feature_policy,
    config_sha256,
    ensure_new_outputs,
    file_sha256,
    json_safe_config,
    validate_config,
    validate_data_dimension,
    validate_model_directory,
)


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
    """Compatibility helper for the scaled ablation's train-fit transform."""
    transformed, _ = apply_feature_policy(data, "scaled_engineered_15")
    return transformed

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
    expected_dim = int(cfg["feature_count"])
    validate_data_dimension(data, expected_dim, f"bank {bank_id}, seed {seed}")
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {out_dir}")
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
        expected_dim,
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
        "experiment_name": cfg["experiment_name"],
        "model_input_dim": int(model.net[0].in_features),
    })

    checkpoint_path = out_dir / "model.pt"
    torch.save(best_state, checkpoint_path)
    provenance_context = cfg["_provenance_context"]
    results["provenance"] = {
        "experiment_name": cfg["experiment_name"],
        "bank_id": bank_id,
        "seed": seed,
        "feature_count": expected_dim,
        "feature_names": list(cfg["feature_names"]),
        "feature_transform": provenance_context["feature_transform"],
        "configuration_path": provenance_context["configuration_path"],
        "configuration_sha256": provenance_context["configuration_sha256"],
        "base_configuration_path": provenance_context["base_configuration_path"],
        "base_configuration_sha256": provenance_context["base_configuration_sha256"],
        "configuration": json_safe_config({k: v for k, v in cfg.items() if not k.startswith("_")}),
        "training_parameters": {
            k: cfg.get(k) for k in (
                "epochs", "batch_size", "lr_max", "lr_min", "weight_decay",
                "grad_clip", "imbalance", "patience", "train_eval_subset",
            )
        } | {"limit_train_rows": limit_train_rows},
        "dataset_cache_path": provenance_context["dataset_cache_path"],
        "dataset_cache_sha256": provenance_context["dataset_cache_sha256"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "model_input_dim": int(model.net[0].in_features),
        "best_validation_metric": {"name": "pr_auc", "value": best_pr},
        "selected_classification_threshold": chosen_threshold,
        "test_metrics": {
            "pr_auc": results["pr_auc"],
            "roc_auc": results["roc_auc"],
            "at_validation_selected_threshold": results["at_val_best_f1_threshold"],
            "at_fixed_thresholds": results["thresholds"],
        },
        "training_completion_status": "in_progress",
    }
    results["training_completion_status"] = "in_progress"
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    plot_results(
        history,
        results["at_val_best_f1_threshold"],
        bank_id,
        seed,
        out_dir,
    )

    # Mark complete only after checkpoint, history, metrics, and plots all exist.
    results["provenance"]["training_completion_status"] = "completed"
    results["training_completion_status"] = "completed"
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

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
        "accuracy_at_val_best_f1_threshold": mean_std([
            r["at_val_best_f1_threshold"]["accuracy"] for r in results
        ]),
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


def main(default_config="local_only/configs/local_only.yaml", expected_experiment="flower15"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="common/config_base.yaml")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--banks", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train-rows", type=int, default=None)
    parser.add_argument("--imbalance", default=None)
    args = parser.parse_args()

    with open(args.base) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.config) as f:
        experiment_cfg = yaml.safe_load(f)

    cfg = {**base_cfg, **experiment_cfg}
    spec = validate_config(cfg, expected_experiment, args.config)
    output_root = Path(args.out_dir if args.out_dir is not None else spec["out_dir"]).resolve()
    validate_model_directory(output_root, expected_experiment)
    cfg["out_dir"] = str(output_root)

    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.imbalance is not None:
        cfg["imbalance"] = args.imbalance

    seeds = args.seeds if args.seeds is not None else cfg["seeds"]

    # Refuse collisions before loading data or beginning training.
    targets = [output_root, output_root / "summary_all_banks.json"]
    for bank_id in args.banks:
        bank_dir = output_root / f"bank_{bank_id}"
        targets.append(bank_dir / "summary.json")
        targets.extend(bank_dir / f"seed_{seed}" for seed in seeds)
        targets.extend(
            bank_dir / f"seed_{seed}" / name
            for seed in seeds
            for name in ("model.pt", "history.json", "results.json", "loss_curves.png", "confusion_matrix.png")
        )
    ensure_new_outputs(targets, cfg["experiment_name"])

    config_path = Path(args.config).resolve()
    print(
        f"Experiment: {spec['experiment_name']} | features: {cfg['feature_count']} | "
        f"config: {config_path} | output: {cfg['out_dir']}"
    )
    print("Loading shared preprocessed dataset...")
    data = load_or_build(cfg["cache_path"], cfg["val_fraction"])
    data, feature_transform = apply_feature_policy(data, cfg["feature_mode"])
    validate_data_dimension(data, cfg["feature_count"], cfg["experiment_name"])
    cache_digest = file_sha256(cfg["cache_path"])
    if cache_digest is None:
        raise RuntimeError(f"Dataset cache is unavailable for provenance: {cfg['cache_path']}")
    cfg["_provenance_context"] = {
        "feature_transform": feature_transform,
        "configuration_path": str(config_path),
        "configuration_sha256": config_sha256(config_path),
        "base_configuration_path": str(Path(args.base).resolve()),
        "base_configuration_sha256": config_sha256(args.base),
        "dataset_cache_path": str(Path(cfg["cache_path"]).resolve()),
        "dataset_cache_sha256": cache_digest,
    }

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
            json.dumps({
                "experiment_name": cfg["experiment_name"],
                "feature_count": cfg["feature_count"],
                "feature_names": cfg["feature_names"],
                "evaluation_scope": "bank-specific test split",
                "training_completion_status": "completed",
                "config": json_safe_config({k: v for k, v in cfg.items() if not k.startswith("_")}),
                **summary,
            }, indent=2)
        )

        print(f"\nBank {bank_id} summary:")
        print(json.dumps(summary, indent=2))

    root = Path(cfg["out_dir"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary_all_banks.json").write_text(
        json.dumps({
            "config": json_safe_config({k: v for k, v in cfg.items() if not k.startswith("_")}),
            "experiment_name": cfg["experiment_name"],
            "feature_count": cfg["feature_count"],
            "feature_names": cfg["feature_names"],
            "evaluation_scope": "bank-specific test split",
            "training_completion_status": "completed",
            "banks": all_results,
        }, indent=2)
    )


if __name__ == "__main__":
    main()
