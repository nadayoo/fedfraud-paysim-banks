"""Evaluate each independent bank model on the complete global test set."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from common.data import load_or_build
from common.metrics import compute_metrics, threshold_metrics
from common.model import Net
from local_only.experiment_utils import (
    apply_feature_policy,
    config_sha256,
    ensure_new_outputs,
    file_sha256,
    state_dict_input_dim,
    validate_config,
    validate_data_dimension,
    validate_evaluation_output,
    validate_model_directory,
)


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


def main(default_config="local_only/configs/local_only.yaml", expected_experiment="flower15"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="common/config_base.yaml")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--banks", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()

    with open(args.base) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.config) as f:
        experiment_cfg = yaml.safe_load(f)
    cfg = {**base_cfg, **experiment_cfg}
    spec = validate_config(cfg, expected_experiment, args.config)

    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    output_root = Path(args.out_dir).resolve() if args.out_dir else Path(cfg["global_test_out_dir"]).resolve()
    validate_evaluation_output(output_root, expected_experiment)
    model_root = Path(args.model_dir).resolve() if args.model_dir else Path(cfg["out_dir"]).resolve()
    validate_model_directory(model_root, expected_experiment)

    targets = [output_root, output_root / "summary_all_banks.json"]
    source_files = []
    for bank_id in args.banks:
        bank_dir = output_root / f"bank_{bank_id}"
        targets.append(bank_dir)
        targets.append(bank_dir / "summary.json")
        for seed in seeds:
            run_dir = model_root / f"bank_{bank_id}" / f"seed_{seed}"
            source_files.extend([run_dir / "model.pt", run_dir / "results.json"])
            targets.append(bank_dir / f"seed_{seed}")
            targets.append(bank_dir / f"seed_{seed}" / "global_test_results.json")
    ensure_new_outputs(targets, f"{spec['experiment_name']} global evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = Path(args.config).resolve()
    print(
        f"Experiment: {spec['experiment_name']} | features: {cfg['feature_count']} | "
        f"model directory: {model_root} | config: {config_path} | output: {output_root}"
    )
    for path in source_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {spec['experiment_name']} checkpoint/result: {path}")

    config_path = Path(args.config).resolve()
    base_path = Path(args.base).resolve()
    experiment_config_hash = config_sha256(config_path)
    base_config_hash = config_sha256(base_path)
    runs = {}
    reference_transform = None
    for bank_id in args.banks:
        for seed in seeds:
            run_dir = model_root / f"bank_{bank_id}" / f"seed_{seed}"
            model_path = run_dir / "model.pt"
            results_path = run_dir / "results.json"
            original_results = json.loads(results_path.read_text())
            provenance = original_results.get("provenance")
            if not provenance:
                raise ValueError(
                    f"Historical run has no verifiable provenance: {results_path}"
                )
            if (original_results.get("training_completion_status") != "completed"
                    or provenance.get("training_completion_status") != "completed"):
                raise ValueError(f"Training run is incomplete: {results_path}")
            if provenance.get("experiment_name") != spec["experiment_name"]:
                raise ValueError(f"Checkpoint experiment metadata mismatch: {results_path}")
            if provenance.get("bank_id") != bank_id or provenance.get("seed") != seed:
                raise ValueError(f"Checkpoint bank/seed metadata mismatch: {results_path}")
            if original_results.get("bank_id") != bank_id or original_results.get("seed") != seed:
                raise ValueError(f"Training result bank/seed mismatch: {results_path}")
            if (Path(provenance.get("configuration_path", "")).resolve() != config_path
                    or provenance.get("configuration_sha256") != experiment_config_hash):
                raise ValueError(f"Training configuration fingerprint/path mismatch: {results_path}")
            if (Path(provenance.get("base_configuration_path", "")).resolve() != base_path
                    or provenance.get("base_configuration_sha256") != base_config_hash):
                raise ValueError(f"Base configuration fingerprint/path mismatch: {results_path}")
            if provenance.get("feature_count") != cfg["feature_count"]:
                raise ValueError(f"Checkpoint feature-count metadata mismatch: {results_path}")
            if provenance.get("feature_names") != cfg["feature_names"]:
                raise ValueError(f"Checkpoint feature names/order mismatch: {results_path}")
            saved_config = provenance.get("configuration", {})
            saved_model_config = saved_config.get("model", {})
            if any(saved_model_config.get(k) != cfg["model"].get(k)
                   for k in ("hidden_dim_1", "hidden_dim_2", "dropout", "use_layernorm")):
                raise ValueError(f"Training model configuration mismatch: {results_path}")
            saved_out_dir = Path(saved_config.get("out_dir", "")).resolve()
            if saved_out_dir != model_root:
                raise ValueError(f"Checkpoint output-directory metadata mismatch: {results_path}")
            if Path(provenance.get("dataset_cache_path", "")).resolve() != Path(cfg["cache_path"]).resolve():
                raise ValueError(f"Dataset cache path mismatch: {results_path}")
            if provenance.get("feature_transform", {}).get("mode") != cfg["feature_mode"]:
                raise ValueError(f"Checkpoint feature transform mismatch: {results_path}")
            transform = provenance["feature_transform"]
            if (transform.get("feature_count") != cfg["feature_count"]
                    or transform.get("feature_names") != cfg["feature_names"]):
                raise ValueError(f"Feature-transform metadata mismatch: {results_path}")
            if reference_transform is None:
                reference_transform = transform
            elif transform != reference_transform:
                raise ValueError("Runs in one experiment do not share identical feature transformations")

            threshold = float(original_results.get("chosen_threshold", float("nan")))
            saved_threshold = float(provenance.get("selected_classification_threshold", float("nan")))
            if (not np.isfinite(threshold) or not 0 <= threshold <= 1
                    or threshold != saved_threshold):
                raise ValueError(f"Selected threshold is missing or inconsistent: {results_path}")
            best_val = provenance.get("best_validation_metric", {})
            if (best_val.get("name") != "pr_auc"
                    or float(best_val.get("value", float("nan"))) != float(original_results.get("val_pr_auc", float("nan")))):
                raise ValueError(f"Best validation metric metadata mismatch: {results_path}")

            state = torch.load(model_path, map_location="cpu", weights_only=True)
            checkpoint_dim = state_dict_input_dim(state)
            if (checkpoint_dim != cfg["feature_count"]
                    or provenance.get("model_input_dim") != checkpoint_dim
                    or original_results.get("model_input_dim") != checkpoint_dim):
                raise ValueError(f"Checkpoint input dimension metadata mismatch: {model_path}")
            if Path(provenance.get("checkpoint_path", "")).resolve() != model_path.resolve():
                raise ValueError(f"Checkpoint path does not match provenance: {model_path}")

            cache_hash = provenance.get("dataset_cache_sha256")
            if not isinstance(cache_hash, str) or len(cache_hash) != 64:
                raise ValueError(f"Dataset cache fingerprint missing or malformed: {results_path}")
            runs[(bank_id, seed)] = {
                "results": original_results,
                "provenance": provenance,
                "state": state,
                "threshold": threshold,
                "cache_hash": cache_hash,
            }

    cache_path = Path(cfg["cache_path"]).resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"Dataset cache required for verified evaluation: {cache_path}")
    current_cache_hash = file_sha256(cache_path)
    if any(run["cache_hash"] != current_cache_hash for run in runs.values()):
        raise ValueError(f"Dataset cache fingerprint differs from the training runs: {cache_path}")

    print("Loading the verified shared dataset cache and applying this experiment's feature policy...")
    data = load_or_build(cfg["cache_path"], cfg["val_fraction"])
    # Scaled evaluations apply the exact training-fitted transform persisted in
    # each run's provenance. The other variants do not transform these columns.
    data, transform_metadata = apply_feature_policy(
        data,
        cfg["feature_mode"],
        stored_transform=reference_transform if cfg["feature_mode"] == "scaled_engineered_15" else None,
    )
    validate_data_dimension(data, cfg["feature_count"], spec["experiment_name"])
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
            run = runs[(bank_id, seed)]
            original_results = run["results"]
            state = run["state"]
            # Reuse the threshold selected on this model's validation set.
            threshold = run["threshold"]

            model = Net(
                cfg["feature_count"],
                m["hidden_dim_1"],
                m["hidden_dim_2"],
                m["dropout"],
                m["use_layernorm"],
            ).to(device)

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
                    **{
                        metric: float(value)
                        for metric, value in compute_metrics(
                            probs[mask], ytest[mask], cfg["thresholds"]
                        ).items()
                        if metric in {"pr_auc", "roc_auc"}
                    },
                    **threshold_metrics(probs[mask], ytest[mask], threshold),
                }

            result.update({
                "model_bank_id": bank_id,
                "seed": seed,
                "chosen_threshold_from_validation": threshold,
                "global_test_rows": len(Xtest),
                "global_test_frauds": int(ytest.sum()),
                "experiment_name": spec["experiment_name"],
                "feature_count": cfg["feature_count"],
                "feature_names": cfg["feature_names"],
                "evaluation_scope": "global test with destination-bank breakdown",
                "training_completion_status": "completed",
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
            json.dumps({
                "experiment_name": spec["experiment_name"],
                "feature_count": cfg["feature_count"],
                "feature_names": cfg["feature_names"],
                "evaluation_scope": "global test, model trained on this bank",
                "training_completion_status": "completed",
                "standard_deviation_method": "sample standard deviation (ddof=1); 0 for one seed",
                **summary,
            }, indent=2)
        )

        print(f"\nBank {bank_id} global-test summary:")
        print(json.dumps(summary, indent=2))

    (output_root / "summary_all_banks.json").write_text(
        json.dumps({
            "experiment_name": spec["experiment_name"],
            "feature_count": cfg["feature_count"],
            "feature_names": cfg["feature_names"],
            "evaluation": "Each local model evaluated on the complete global test set",
            "threshold_policy": "Each model's own validation-selected threshold",
            "dataset_cache_sha256": current_cache_hash,
            "standard_deviation_method": "sample standard deviation (ddof=1); 0 for one seed",
            "banks": all_summaries,
        }, indent=2)
    )

    print(f"\nGlobal evaluation complete. Results saved in: {output_root}")


if __name__ == "__main__":
    main()
