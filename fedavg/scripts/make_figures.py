"""Report figures + tables from the FedAvg result files.

Usage (project folder, venv active):
    python scripts/make_figures.py                          # reads results/, writes figures/
    python scripts/make_figures.py result_fedavg_seed results 0.5 figures

Reads result_fedavg_seed*.json (written by run_fedavg_local.py) or *.pkl (written by `flwr run`).
Works with 1, 2 or 3 seeds (std is 0 for a single seed).
"""

import csv
import glob
import json
import pickle
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

prefix = sys.argv[1] if len(sys.argv) > 1 else "result_fedavg_seed"
root = sys.argv[2] if len(sys.argv) > 2 else "results"
thr = sys.argv[3] if len(sys.argv) > 3 else "0.5"
outdir = Path(sys.argv[4] if len(sys.argv) > 4 else "figures")
outdir.mkdir(exist_ok=True)
PAPER_PR_AUC, PAPER_STD = 56.40, 0.45
THRESHOLDS = ["0.05", "0.1", "0.2", "0.5", "0.8", "0.9", "0.95", "0.99"]

# ---- style: light surface, one accent, recessive grid, text in ink (never series colour)
INK, INK2, GRID = "#1f2933", "#52606d", "#e4e7eb"
BLUE, ORANGE, GREEN, PAPER_C = "#0072B2", "#D55E00", "#009E73", "#6b7280"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
})

# ---- load
files = sorted(glob.glob(f"{root}/**/{prefix}*.pkl", recursive=True)) + sorted(
    f for f in glob.glob(f"{root}/**/{prefix}*.json", recursive=True) if not f.endswith("_communication.json"))
assert files, f"no {prefix}*.json/.pkl under {root}"
runs = []
for f in files:
    if f.endswith(".json"):
        r = json.load(open(f))["evaluate_metrics_serverapp"]
    else:
        r = pickle.load(open(f, "rb")).evaluate_metrics_serverapp
    runs.append({int(k): dict(v) for k, v in r.items()})
n = len(runs)
rounds = sorted(set.intersection(*[set(r) for r in runs]))
last = rounds[-1]
print(f"{n} seed(s), rounds 0..{last}, threshold {thr}")


def series(metric, t=thr):
    return np.array([[run[k][f"{metric}_at_{t}"] for k in rounds] for run in runs]) * 100


def final(metric, t=thr):
    return np.array([run[last][f"{metric}_at_{t}"] for run in runs]) * 100


# ---- 1. convergence curves (small multiples, one y-axis each)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, (m, title) in zip(axes, [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("f1", f"F1 (threshold {thr})")]):
    c = series(m)
    mean, sd = c.mean(0), c.std(0)
    for row in c:
        ax.plot(rounds, row, color=BLUE, lw=0.8, alpha=0.25)
    ax.plot(rounds, mean, color=BLUE, lw=2, label="FedAvg, mean of seeds")
    ax.fill_between(rounds, mean - sd, mean + sd, color=BLUE, alpha=0.18, lw=0)
    if m == "pr_auc":
        ax.axhline(PAPER_PR_AUC, color=PAPER_C, ls="--", lw=1.4, label=f"Paper FedAvg {PAPER_PR_AUC:.2f}")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("%")
    ax.set_xlim(0, last)
    ax.annotate(f"{mean[-1]:.1f}", (rounds[-1], mean[-1]), textcoords="offset points", xytext=(0, 6),
                ha="right", fontsize=9, color=INK)
fig.suptitle(f"Convergence of FedAvg on the 5-bank PaySim split ({n} seed{'s' if n > 1 else ''}; band = ±1 std)",
             x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(outdir / "fig1_convergence.png", dpi=200)
plt.close(fig)

# ---- 2. confusion matrix (mean over seeds, final round)
cm = np.array([[np.mean([run[last][f"{k}_at_{thr}"] for run in runs]) for k in ("tn", "fp")],
               [np.mean([run[last][f"{k}_at_{thr}"] for run in runs]) for k in ("fn", "tp")]])
rowpct = cm / cm.sum(1, keepdims=True) * 100
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.grid(False)
im = ax.imshow(rowpct, cmap="Blues", vmin=0, vmax=100)
for i in range(2):
    for j in range(2):
        col = "white" if rowpct[i, j] > 55 else INK
        ax.text(j, i, f"{cm[i, j]:,.0f}\n({rowpct[i, j]:.1f}% of row)", ha="center", va="center",
                color=col, fontsize=11, fontweight="bold")
ax.set_xticks([0, 1], ["Predicted\nlegitimate", "Predicted\nfraud"])
ax.set_yticks([0, 1], ["Actual\nlegitimate", "Actual\nfraud"])
ax.spines[:].set_visible(False)
ax.tick_params(length=0)
ax.set_title(f"Confusion matrix, round {last}, threshold {thr}", pad=10)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("% of actual class")
fig.tight_layout()
fig.savefig(outdir / "fig2_confusion_matrix.png", dpi=200)
plt.close(fig)

# ---- 3. precision / recall / F1 vs decision threshold (final round)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
xs = np.arange(len(THRESHOLDS))
for m, colr, lab in [("precision", ORANGE, "Precision"), ("recall", BLUE, "Recall"), ("f1", GREEN, "F1")]:
    v = np.array([final(m, t).mean() for t in THRESHOLDS])
    ax.plot(xs, v, marker="o", ms=5, lw=2, color=colr, label=lab)
ax.set_xticks(xs, THRESHOLDS)
ax.set_xlabel("Decision threshold")
ax.set_ylabel("%")
ax.set_ylim(0, 102)
ax.set_title(f"Effect of the decision threshold (round {last}, mean of seeds)")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
fig.savefig(outdir / "fig3_threshold_sweep.png", dpi=200)
plt.close(fig)

# ---- 4. time and communication per round (two panels, no dual axis)
comms = []
for f in sorted(glob.glob(f"{root}/**/{prefix}*_communication.json", recursive=True)):
    comms.append(json.load(open(f)).get("train_rounds", {}))
if comms and all("round_time_sec" in next(iter(c.values()), {}) for c in comms if c):
    cr = sorted(int(k) for k in comms[0])
    t_min = np.array([[c[str(k)]["round_time_sec"] / 60 for k in cr] for c in comms])
    mb = np.array([[c[str(k)]["comm_bytes_total"] / 1e6 for k in cr] for c in comms]).mean(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(cr, t_min.mean(0), color=BLUE, width=0.7)
    axes[0].set_title("Wall-clock time per round")
    axes[0].set_xlabel("Communication round")
    axes[0].set_ylabel("minutes")
    axes[1].plot(cr, np.cumsum(mb), color=ORANGE, lw=2)
    axes[1].set_title("Cumulative communication (model up + down, 5 banks)")
    axes[1].set_xlabel("Communication round")
    axes[1].set_ylabel("MB")
    axes[1].annotate(f"{np.cumsum(mb)[-1]:.1f} MB", (cr[-1], np.cumsum(mb)[-1]), textcoords="offset points",
                     xytext=(0, 6), ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_time_communication.png", dpi=200)
    plt.close(fig)
    print(f"mean time/round {t_min.mean():.1f} min; total comm {np.cumsum(mb)[-1]:.2f} MB per run")
else:
    print("no per-round time/communication data found: fig4 skipped")

# ---- 5. final-metrics table (CSV + LaTeX + Markdown)
metrics = ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]
labels = ["Accuracy", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]
rows = [(lab, final(m).mean(), final(m).std()) for m, lab in zip(metrics, labels)]
with open(outdir / "results_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "mean_percent", "std_percent"])
    for lab, mu, sd in rows:
        w.writerow([lab, f"{mu:.2f}", f"{sd:.2f}"])
tex = ["\\begin{tabular}{lr}", "\\toprule", "Metric & Mean $\\pm$ std (\\%) \\\\", "\\midrule"]
tex += [f"{lab} & ${mu:.2f} \\pm {sd:.2f}$ \\\\" for lab, mu, sd in rows]
tex += ["\\midrule", f"Paper FedAvg PR-AUC & ${PAPER_PR_AUC:.2f} \\pm {PAPER_STD:.2f}$ \\\\", "\\bottomrule", "\\end{tabular}"]
(outdir / "results_table.tex").write_text("\n".join(tex), encoding="utf-8")
md = ["| Metric | Mean ± std (%) |", "|---|---|"] + [f"| {lab} | {mu:.2f} ± {sd:.2f} |" for lab, mu, sd in rows]
md.append(f"| Paper FedAvg PR-AUC | {PAPER_PR_AUC:.2f} ± {PAPER_STD:.2f} |")
(outdir / "results_table.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
print(f"\nFigures and tables written to {outdir.resolve()}")


# ---- 6. per-seed table (CSV + LaTeX + Markdown), with mean ± std row
seed_names = [(re.search(r"seed(\d+)", f) or [None, str(i)])[1] if re.search(r"seed(\d+)", f) else str(i)
              for i, f in enumerate(files)]
cols = ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]
hdr = ["Seed"] + labels + ["TP", "FP", "FN", "TN"]
body = []
for name, run in zip(seed_names, runs):
    body.append([name] + [f"{run[last][f'{m}_at_{thr}'] * 100:.2f}" for m in cols]
                + [f"{run[last][f'{k}_at_{thr}']:,.0f}" for k in ("tp", "fp", "fn", "tn")])
mean_row = ["Mean ± std"] + [f"{final(m).mean():.2f} ± {final(m).std():.2f}" for m in cols] \
    + [f"{np.mean([r[last][f'{k}_at_{thr}'] for r in runs]):,.0f}" for k in ("tp", "fp", "fn", "tn")]
with open(outdir / "per_seed_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(hdr)
    w.writerows(body + [mean_row])
md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
md += ["| " + " | ".join(r) + " |" for r in body + [mean_row]]
(outdir / "per_seed_table.md").write_text("\n".join(md), encoding="utf-8")
tex = ["\\begin{tabular}{l" + "r" * (len(hdr) - 1) + "}", "\\toprule", " & ".join(hdr) + " \\\\", "\\midrule"]
tex += [" & ".join(r).replace("±", "$\\pm$").replace(",", "{,}") + " \\\\" for r in body]
tex += ["\\midrule", " & ".join(mean_row).replace("±", "$\\pm$").replace(",", "{,}") + " \\\\", "\\bottomrule", "\\end{tabular}"]
(outdir / "per_seed_table.tex").write_text("\n".join(tex), encoding="utf-8")
print("\nPer-seed table (final round, threshold " + thr + ", %):")
print("\n".join(md))

# ---- 7. confusion matrix of every seed (one figure, one panel per seed)
fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.0), squeeze=False)
for ax, name, run in zip(axes[0], seed_names, runs):
    c = np.array([[run[last][f"{k}_at_{thr}"] for k in ("tn", "fp")],
                  [run[last][f"{k}_at_{thr}"] for k in ("fn", "tp")]])
    pct = c / c.sum(1, keepdims=True) * 100
    ax.grid(False)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{c[i, j]:,.0f}\n({pct[i, j]:.1f}%)", ha="center", va="center",
                    color="white" if pct[i, j] > 55 else INK, fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1], ["Pred.\nlegit.", "Pred.\nfraud"])
    ax.set_yticks([0, 1], ["Actual\nlegit.", "Actual\nfraud"])
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"Seed {name}")
fig.suptitle(f"Confusion matrices, round {last}, threshold {thr} (cell colour = % of actual class)",
             x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(outdir / "fig5_confusion_per_seed.png", dpi=200)
plt.close(fig)
print(f"\nAll files in {outdir.resolve()}")
