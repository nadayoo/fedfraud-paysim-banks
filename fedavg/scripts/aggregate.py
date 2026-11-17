"""Aggregate the 3 FedAvg seeds from the official app's result pickles.

Run from the folder where `flwr run` wrote result_fedavg_seed*.pkl (needs flwr installed):
    python scripts/aggregate.py            # default prefix: result_fedavg_seed
    python scripts/aggregate.py result_fedavg_seed . 0.5
"""

import glob
import json
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

prefix = sys.argv[1] if len(sys.argv) > 1 else "result_fedavg_seed"
root = sys.argv[2] if len(sys.argv) > 2 else "."
thr = sys.argv[3] if len(sys.argv) > 3 else "0.5"
PAPER = (56.40, 0.45)

files = sorted(f for f in glob.glob(f"{root}/**/{prefix}*.pkl", recursive=True))
assert files, f"no {prefix}*.pkl found under {root}"

runs = []
for f in files:
    r = pickle.load(open(f, "rb"))
    runs.append({int(k): dict(v) for k, v in r.evaluate_metrics_serverapp.items()})
print("seeds/files:", files)

last = max(runs[0])
rows = [f"Final round ({last}) at threshold {thr}, mean ± std over {len(runs)} seeds (%):"]
for m in ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]:
    key = f"{m}_at_{thr}"
    v = np.array([run[last][key] for run in runs]) * 100
    rows.append(f"  {m:10s} {v.mean():6.2f} ± {v.std():.2f}")
pr = np.array([run[last][f"pr_auc_at_{thr}"] for run in runs]) * 100
rows.append(f"\nPaper FedAvg PR-AUC {PAPER[0]:.2f} ± {PAPER[1]:.2f} | reproduced {pr.mean():.2f} ± {pr.std():.2f} "
            f"(gap {pr.mean() - PAPER[0]:+.2f})")
print("\n".join(rows))
open("fedavg_summary.txt", "w").write("\n".join(rows))

rounds = sorted(runs[0])
for m in ["pr_auc", "roc_auc", "f1"]:
    c = np.array([[run[k][f"{m}_at_{thr}"] for k in rounds] for run in runs]) * 100
    plt.figure(figsize=(5, 3.5))
    plt.plot(rounds, c.mean(0)); plt.fill_between(rounds, c.mean(0) - c.std(0), c.mean(0) + c.std(0), alpha=0.25)
    if m == "pr_auc":
        plt.axhline(PAPER[0], ls="--", c="k", label="paper 56.40"); plt.legend()
    plt.xlabel("round"); plt.ylabel(f"{m} (%)"); plt.tight_layout(); plt.savefig(f"fedavg_{m}_curve.png", dpi=150)

cm = np.mean([[run[last][f"{k}_at_{thr}"] for k in ("tn", "fp", "fn", "tp")] for run in runs], axis=0).reshape(2, 2)
print("\nConfusion matrix (mean, rows=true 0/1, cols=pred 0/1):\n", cm.round(0))

for f in glob.glob(f"{root}/**/{prefix}*_communication.json", recursive=True):
    print(f, json.load(open(f))["totals"])
