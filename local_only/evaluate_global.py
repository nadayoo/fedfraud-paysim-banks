"""Evaluate each independent bank model on the complete global test set."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from common.data import INPUT_DIM, load_or_build
from common.metrics import compute_metrics, threshold_metrics
from common.model import Net
from local_only.train import scale_engineered_features


def predict_probs(model, X, device, batch_size=65536):
    model.eval()
    predictions = []

    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size]).to(device)
            logits = model(xb)
            predictions.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(predictions)


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def aggregate(results):
    return {
        "seeds": [r["seed"] for r in results],
        "pr_auc": mean_std([r["pr_auc"] for r in results]),
        "roc_auc": mean_std([r["roc_auc"] for r in results]),
        "accuracy_at_val_threshold": mean_std([
            r["at_val_best_f1_threshold"]["accuracy"] for r in results
        ]),
        "precision_at_val_threshold": mean_std([
            r["at_val_best_f1_threshold"]["precision"] for r in results
        ]),
        "recall_at_val_threshold": mean_std([
            r["at_val_best_f1_threshold"]["recall"] for r in results
        ]),
        "f1_at_val_threshold": mean_std([
            r["at_val_best_f1_threshold"]["f1"] for r in results
        ]),
        "chosen_threshold": mean_std([
            r["chosen_threshold_from_validation"] for r in results
        ]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="common/config_base.yaml")
    parser.add_argument("--config", default="local_only/configs/local_only.yaml")
    parser.add_argument("--banks", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--out-dir", default="results/local_only_global_test")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--scale-engineered", action="store_true")
    args = parser.parse_args()

    with open(args.base) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.config) as f:
        experiment_cfg = yaml.safe_load(f)
    cfg = {**base_cfg, **experiment_cfg}

    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    output_root = Path(args.out_dir)
    model_root = Path(args.model_dir) if args.model_dir else Path(cfg["out_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading the original shared test dataset...")
    data = load_or_build(cfg["cache_path"], cfg["val_fraction"])
    if args.scale_engineered:
        data = scale_engineered_features(data)
    Xtest = data["X_test"]
    ytest = data["y_test"]
    bank_test = data["bank_test"]

    print(
        f"Global test set: {len(Xtest):,} rows, "
        f"{int(ytest.sum()):,} frauds, device={device}"
    )

    m = cfg["model"]
    all_summaries = {}

    for bank_id in args.banks:
        bank_results = []

        for seed in seeds:
            run_dir = (
                model_root
                / f"bank_{bank_id}"
                / f"seed_{seed}"
            )
            model_path = run_dir / "model.pt"
            original_results_path = run_dir / "results.json"

            if not model_path.exists() or not original_results_path.exists():
                raise FileNotFoundError(
                    f"Missing checkpoint or results for bank {bank_id}, "
                    f"seed {seed}: {run_dir}"
                )

            with open(original_results_path) as f:
                original_results = json.load(f)

            # Reuse the threshold selected on this model's validation set.
            threshold = float(original_results["chosen_threshold"])

            model = Net(
                INPUT_DIM,
                m["hidden_dim_1"],
                m["hidden_dim_2"],
                m["dropout"],
                m["use_layernorm"],
            ).to(device)

            state = torch.load(
                model_path,
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(state)

            # Evaluate this bank's model on ALL five banks' test rows.
            probs = predict_probs(model, Xtest, device)

            result = compute_metrics(probs, ytest, cfg["thresholds"])
            result["at_val_best_f1_threshold"] = threshold_metrics(
                probs, ytest, threshold
            )

            # Breakdown by destination bank, using the same threshold.
            result["per_bank"] = {}
            for test_bank in sorted(np.unique(bank_test).tolist()):
                mask = bank_test == test_bank
                result["per_bank"][str(int(test_bank))] = {
                    "pr_auc": float(
                        compute_metrics(
                            probs[mask], ytest[mask], cfg["thresholds"]
                        )["pr_auc"]
                    ),
                    **threshold_metrics(probs[mask], ytest[mask], threshold),
                }

            result.update({
                "model_bank_id": bank_id,
                "seed": seed,
                "chosen_threshold_from_validation": threshold,
                "global_test_rows": len(Xtest),
                "global_test_frauds": int(ytest.sum()),
            })

            seed_dir = output_root / f"bank_{bank_id}" / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "global_test_results.json").write_text(
                json.dumps(result, indent=2)
            )

            bank_results.append(result)

            print(
                f"Model bank {bank_id} | seed {seed} | "
                f"global PR-AUC={result['pr_auc']:.4f} | "
                f"ROC-AUC={result['roc_auc']:.4f} | "
                f"F1={result['at_val_best_f1_threshold']['f1']:.4f}"
            )

        summary = aggregate(bank_results)
        all_summaries[str(bank_id)] = summary

        bank_dir = output_root / f"bank_{bank_id}"
        (bank_dir / "summary.json").write_text(
            json.dumps(summary, indent=2)
        )

        print(f"\nBank {bank_id} global-test summary:")
        print(json.dumps(summary, indent=2))

    (output_root / "summary_all_banks.json").write_text(
        json.dumps({
            "evaluation": "Each local model evaluated on the complete global test set",
            "threshold_policy": "Each model's own validation-selected threshold",
            "banks": all_summaries,
        }, indent=2)
    )

    print(f"\nGlobal evaluation complete. Results saved in: {output_root}")


if __name__ == "__main__":
    main()
