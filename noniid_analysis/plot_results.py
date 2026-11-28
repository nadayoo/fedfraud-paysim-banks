import csv
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("noniid_analysis/flwr_app")
OUT = BASE / "comparison_outputs"
OUT.mkdir(exist_ok=True)

EXPERIMENTS = {
    "Original (natural banks)": "result_original_seed0_r20.pkl",
    "IID": "result_iid_seed0_r20.pkl",
    "Fraud skew": "result_fraud_skew_seed0_r20.pkl",
    "Type skew (alpha=0.5)": "result_type_skew_a0.5_seed0_r20.pkl",
}

THRESHOLD = "0.5"

runs = {}
for label, filename in EXPERIMENTS.items():
    with open(BASE / filename, "rb") as f:
        result = pickle.load(f)

    metrics = {
        int(round_num): dict(values)
        for round_num, values in result.evaluate_metrics_serverapp.items()
    }
    runs[label] = metrics

# Confirm that all experiments have the same evaluation rounds.
round_sets = [set(metrics) for metrics in runs.values()]
if not all(rounds == round_sets[0] for rounds in round_sets):
    raise ValueError("Experiments do not contain identical evaluation rounds.")

rounds = sorted(round_sets[0])

# Export every round's metrics to CSV.
csv_path = OUT / "partition_comparison_all_rounds.csv"
fields = [
    "experiment", "round", "centralized_loss",
    "pr_auc", "roc_auc",
    f"precision_at_{THRESHOLD}",
    f"recall_at_{THRESHOLD}",
    f"f1_at_{THRESHOLD}",
    f"accuracy_at_{THRESHOLD}",
]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for label, metrics in runs.items():
        for r in rounds:
            m = metrics[r]
            writer.writerow({
                "experiment": label,
                "round": r,
                "centralized_loss": m["centralized_loss"],
                "pr_auc": m[f"pr_auc_at_{THRESHOLD}"],
                "roc_auc": m[f"roc_auc_at_{THRESHOLD}"],
                f"precision_at_{THRESHOLD}": m[f"precision_at_{THRESHOLD}"],
                f"recall_at_{THRESHOLD}": m[f"recall_at_{THRESHOLD}"],
                f"f1_at_{THRESHOLD}": m[f"f1_at_{THRESHOLD}"],
                f"accuracy_at_{THRESHOLD}": m[f"accuracy_at_{THRESHOLD}"],
            })

# Plot metrics across rounds.
plots = [
    ("pr_auc", "PR-AUC", "partition_pr_auc.png"),
    ("roc_auc", "ROC-AUC", "partition_roc_auc.png"),
    ("f1_at_" + THRESHOLD, f"F1 at threshold {THRESHOLD}", "partition_f1.png"),
    ("centralized_loss", "Centralized evaluation loss", "partition_loss.png"),
]

for metric, ylabel, filename in plots:
    plt.figure(figsize=(8, 5))

    for label, metrics in runs.items():
        values = [
            metrics[r].get(
                metric if metric == "centralized_loss"
                else metric + "_at_" + THRESHOLD
                if metric in ("pr_auc", "roc_auc")
                else metric
            )
            for r in rounds
        ]
        plt.plot(rounds, values, marker="o", markersize=3, label=label)

    plt.xlabel("Communication round")
    plt.ylabel(ylabel)
    plt.title(ylabel + " across communication rounds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=200)
    plt.close()

# Print final-round comparison.
last = rounds[-1]
print(f"\nFinal round: {last}")
print(f"F1 threshold: {THRESHOLD}\n")
print(f"{'Experiment':25s} {'PR-AUC':>10s} {'ROC-AUC':>10s} {'F1':>10s} {'Loss':>10s}")

for label, metrics in runs.items():
    m = metrics[last]
    print(
        f"{label:25s} "
        f"{m['pr_auc_at_0.5']:10.4f} "
        f"{m['roc_auc_at_0.5']:10.4f} "
        f"{m['f1_at_0.5']:10.4f} "
        f"{m['centralized_loss']:10.4f}"
    )

print("\nFiles saved in:", OUT.resolve())
print("CSV:", csv_path.name)
