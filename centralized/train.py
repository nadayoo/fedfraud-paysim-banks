"""Centralized MLP baseline: trains on the COMBINED training data of all five banks.

Run from the repo root:
    python -m centralized.train --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
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
from common.metrics import best_f1_threshold, compute_metrics, sigmoid, threshold_metrics
from common.model import Net, count_parameters
from common.seeding import set_seed


def cosine_annealing(cur: int, total: int, lr_max: float, lr_min: float) -> float:
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * cur / total))


@torch.no_grad()
def predict_logits(model, X, device, batch=65536) -> np.ndarray:
    model.eval()
    return torch.cat([model(torch.from_numpy(X[i:i + batch]).to(device)).cpu()
                      for i in range(0, len(X), batch)]).numpy()


def bce(logits: np.ndarray, y: np.ndarray) -> float:
    return float(F.binary_cross_entropy_with_logits(torch.from_numpy(logits), torch.from_numpy(y)))


def git_commit() -> str:
    """Short commit hash (+ '-dirty' if there are uncommitted changes), for the experiment record."""
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def run_seed(cfg: dict, data: dict, seed: int, out_dir: Path) -> dict:
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = cfg["model"]
    model = Net(INPUT_DIM, m["hidden_dim_1"], m["hidden_dim_2"], m["dropout"], m["use_layernorm"]).to(device)

    Xtr, ytr = data["X_train"], data["y_train"]
    mode = cfg["imbalance"]
    pos_weight = None
    if mode in ("pos_weight", "both"):
        n_pos = float(ytr.sum())
        pos_weight = torch.tensor((len(ytr) - n_pos) / n_pos, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = make_loader(Xtr, ytr, cfg["batch_size"], balanced=mode in ("sampler", "both"),
                         shuffle=True, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr_max"], weight_decay=cfg["weight_decay"])

    rng = np.random.default_rng(seed)
    sub = rng.choice(len(ytr), min(cfg["train_eval_subset"], len(ytr)), replace=False)

    history, best_pr, best_state, best_epoch, bad = [], -1.0, None, -1, 0
    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        lr = cosine_annealing(epoch, cfg["epochs"], cfg["lr_max"], cfg["lr_min"])
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        run, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            run += loss.item(); n += 1

        val_logits = predict_logits(model, data["X_val"], device)
        val_pr = float(average_precision_score(data["y_val"], sigmoid(val_logits)))
        rec = {"epoch": epoch + 1, "lr": lr, "train_loss_sampled": run / max(n, 1),
               "train_loss_natural": bce(predict_logits(model, Xtr[sub], device), ytr[sub]),
               "val_loss": bce(val_logits, data["y_val"]), "val_pr_auc": val_pr}
        history.append(rec)
        print(f"[seed {seed}] epoch {epoch+1}/{cfg['epochs']} lr={lr:.2e} "
              f"train={rec['train_loss_natural']:.4f} val={rec['val_loss']:.4f} val_pr_auc={val_pr:.4f}")
        if val_pr > best_pr:
            best_pr, best_state, best_epoch, bad = val_pr, copy.deepcopy(model.state_dict()), epoch + 1, 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"[seed {seed}] early stop at epoch {epoch+1}")
                break
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    val_probs = sigmoid(predict_logits(model, data["X_val"], device))
    thr = best_f1_threshold(val_probs, data["y_val"])        # chosen on VALIDATION only
    test_probs = sigmoid(predict_logits(model, data["X_test"], device))
    ytest = data["y_test"]
    res = compute_metrics(test_probs, ytest, cfg["thresholds"])
    res["at_val_best_f1_threshold"] = threshold_metrics(test_probs, ytest, thr)
    res["per_bank"] = {}
    for b in sorted(np.unique(data["bank_test"]).tolist()):
        mk = data["bank_test"] == b
        res["per_bank"][str(b)] = {"pr_auc": float(average_precision_score(ytest[mk], test_probs[mk])),
                                   **threshold_metrics(test_probs[mk], ytest[mk], thr)}
    res.update(seed=seed, best_epoch=best_epoch, val_pr_auc=best_pr, chosen_threshold=thr,
               train_time_sec=train_time, n_params=count_parameters(model),
               imbalance=mode, epochs_run=len(history), git_commit=git_commit())

    torch.save(best_state, out_dir / "model.pt")
    np.savez(out_dir / "scaler.npz", means=data["scaler_means"], stds=data["scaler_stds"])
    (out_dir / "config.json").write_text(json.dumps({**cfg, "seed": seed, "git_commit": git_commit()}, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    plot_seed(history, res["at_val_best_f1_threshold"], seed, out_dir)
    return res


def plot_seed(history, tm, seed, out_dir: Path) -> None:
    ep = [h["epoch"] for h in history]
    plt.figure(figsize=(5, 3.5))
    plt.plot(ep, [h["train_loss_natural"] for h in history], label="train (natural dist.)")
    plt.plot(ep, [h["val_loss"] for h in history], label="validation")
    plt.xlabel("epoch"); plt.ylabel("BCE loss"); plt.legend(); plt.title(f"Loss, seed {seed}")
    plt.tight_layout(); plt.savefig(out_dir / "loss_curves.png", dpi=150); plt.close()
    cm = np.array([[tm["tn"], tm["fp"]], [tm["fn"], tm["tp"]]])
    plt.figure(figsize=(3.8, 3.5))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    plt.xticks([0, 1], ["legit", "fraud"]); plt.yticks([0, 1], ["legit", "fraud"])
    plt.xlabel("predicted"); plt.ylabel("actual"); plt.title(f"Test, thr={tm['threshold']:.3f}")
    plt.tight_layout(); plt.savefig(out_dir / "confusion_matrix.png", dpi=150); plt.close()


def aggregate(results: list[dict]) -> dict:
    """mean and sample std (ddof=1) across seeds."""
    def ms(vals):
        v = np.array(vals, dtype=float)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    out = {"seeds": [r["seed"] for r in results],
           "pr_auc": ms([r["pr_auc"] for r in results]),
           "roc_auc": ms([r["roc_auc"] for r in results])}
    for k in ("accuracy", "precision", "recall", "f1", "fpr"):
        out[k + "_at_val_best_f1_threshold"] = ms([r["at_val_best_f1_threshold"][k] for r in results])
    out["train_time_sec"] = ms([r["train_time_sec"] for r in results])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="common/config_base.yaml")
    ap.add_argument("--config", default="centralized/configs/centralized.yaml")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--imbalance", default=None, help="override: none|sampler|pos_weight|both")
    ap.add_argument("--eng", default=None, help="override engineered-feature handling: none|standardize|signed_log")
    args = ap.parse_args()
    cfg = {**yaml.safe_load(open(args.base)), **yaml.safe_load(open(args.config))}
    if args.imbalance:
        cfg["imbalance"] = args.imbalance
    if args.eng:
        cfg["eng_transform"] = args.eng
    eng = cfg.get("eng_transform", "none")
    if eng != "none":   # separate cache + output folder per variant, nothing gets overwritten
        cfg["cache_path"] = cfg["cache_path"].replace(".npz", f"_{eng}.npz")
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    root = Path(cfg["out_dir"]) / (cfg["imbalance"] if eng == "none" else f"{cfg['imbalance']}_{eng}")
    data = load_or_build(cfg["cache_path"], cfg["val_fraction"], eng)
    print({k: v.shape for k, v in data.items() if k.startswith(("X_", "y_"))})
    results = [run_seed(cfg, data, s, root / f"seed_{s}") for s in seeds]
    summary = aggregate(results)
    (root / "summary.json").write_text(json.dumps({"git_commit": git_commit(), "config": cfg, **summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
