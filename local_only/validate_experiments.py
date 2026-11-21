"""Fast, synthetic validation of Member 2 experiment separation.

Run from the repository root with:
    python -m local_only.validate_experiments

This does not load PaySim, train a model, or write experiment artifacts.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import yaml

from common.model import Net
from local_only.experiment_utils import (
    EXPERIMENTS,
    FEATURE_COLUMNS_7,
    apply_feature_policy,
    ensure_new_outputs,
    state_dict_input_dim,
    validate_config,
    validate_evaluation_output,
    validate_model_directory,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "common/config_base.yaml"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def wrapper_arguments(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {
            "train_main", "evaluate_main"
        }:
            return {
                kw.arg: ast.literal_eval(kw.value)
                for kw in node.keywords
                if kw.arg in {"default_config", "expected_experiment"}
            }
    return {}


def main_defaults(path: Path, function_name: str) -> dict[str, str]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            positional = node.args.args
            defaults = node.args.defaults
            return {
                arg.arg: ast.literal_eval(value)
                for arg, value in zip(positional[-len(defaults):], defaults)
            }
    return {}


def synthetic_arrays() -> dict:
    train = np.arange(60, dtype=np.float32).reshape(4, 15)
    val = (100 + np.arange(30, dtype=np.float32)).reshape(2, 15)
    test = (200 + np.arange(45, dtype=np.float32)).reshape(3, 15)
    return {"X_train": train, "X_val": val, "X_test": test}


def main() -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text())
    expected = {
        "flower15": (15, "local_only/train.py", "local_only/evaluate_global.py"),
        "balance_free_7": (
            7, "local_only/train_pretransaction.py", "local_only/evaluate_pretransaction_global.py"
        ),
        "scaled_engineered_15": (
            15, "local_only/train_scaled.py", "local_only/evaluate_scaled_global.py"
        ),
    }

    roots: list[Path] = []
    standard_train = main_defaults(ROOT / "local_only/train.py", "main")
    standard_eval = main_defaults(ROOT / "local_only/evaluate_global.py", "main")
    check(standard_train.get("default_config") == "local_only/configs/local_only.yaml"
          and standard_train.get("expected_experiment") == "flower15",
          "Standard trainer must default to the 15-feature configuration")
    check(standard_eval.get("default_config") == "local_only/configs/local_only.yaml"
          and standard_eval.get("expected_experiment") == "flower15",
          "Standard evaluator must default to the 15-feature configuration")

    for name, (dimension, train_entry, eval_entry) in expected.items():
        spec = EXPERIMENTS[name]
        config_path = ROOT / spec["config"]
        cfg = {**base, **yaml.safe_load(config_path.read_text())}
        validate_config(cfg, name, config_path)
        check(cfg["feature_count"] == dimension, f"{name}: configured feature count mismatch")
        check(len(cfg["feature_names"]) == dimension, f"{name}: feature order length mismatch")
        check(Net(dimension).net[0].in_features == dimension, f"{name}: model input mismatch")
        check(state_dict_input_dim(Net(dimension).state_dict()) == dimension,
              f"{name}: checkpoint dimension inspection mismatch")

        roots.extend([Path(cfg["out_dir"]).resolve(), Path(cfg["global_test_out_dir"]).resolve()])
        validate_evaluation_output(cfg["global_test_out_dir"], name)
        validate_model_directory(cfg["out_dir"], name)

        if train_entry.endswith("train_pretransaction.py"):
            args = wrapper_arguments(ROOT / train_entry)
            check(args == {
                "default_config": "local_only/configs/local_only_pretransaction.yaml",
                "expected_experiment": "balance_free_7",
            }, "Pretransaction trainer must select its own config and experiment")
        elif train_entry.endswith("train_scaled.py"):
            args = wrapper_arguments(ROOT / train_entry)
            check(args == {
                "default_config": "local_only/configs/local_only_scaled.yaml",
                "expected_experiment": "scaled_engineered_15",
            }, "Scaled trainer must select its own config and experiment")

        if eval_entry.endswith("evaluate_pretransaction_global.py"):
            args = wrapper_arguments(ROOT / eval_entry)
            check(args == {
                "default_config": "local_only/configs/local_only_pretransaction.yaml",
                "expected_experiment": "balance_free_7",
            }, "Pretransaction evaluator must select its own config and experiment")
        elif eval_entry.endswith("evaluate_scaled_global.py"):
            args = wrapper_arguments(ROOT / eval_entry)
            check(args == {
                "default_config": "local_only/configs/local_only_scaled.yaml",
                "expected_experiment": "scaled_engineered_15",
            }, "Scaled evaluator must select its own config and experiment")

        transformed, metadata = apply_feature_policy(synthetic_arrays(), cfg["feature_mode"])
        check(metadata["feature_count"] == dimension, f"{name}: transform metadata mismatch")
        for split in ("train", "val", "test"):
            check(transformed[f"X_{split}"].shape[1] == dimension,
                  f"{name}: inconsistent {split} feature dimension")

        if name == "balance_free_7":
            check(np.array_equal(transformed["X_test"], synthetic_arrays()["X_test"][:, FEATURE_COLUMNS_7]),
                  "Balance-free experiment selected unexpected feature columns")
        if name == "scaled_engineered_15":
            means = np.asarray(metadata["means"])
            stds = np.asarray(metadata["stds"])
            expected_val = (synthetic_arrays()["X_val"][:, 6:10] - means) / stds
            expected_test = (synthetic_arrays()["X_test"][:, 6:10] - means) / stds
            check(np.allclose(transformed["X_val"][:, 6:10], expected_val),
                  "Validation scaling differs from training-derived transform")
            check(np.allclose(transformed["X_test"][:, 6:10], expected_test),
                  "Test scaling differs from training-derived transform")
            check(np.allclose(transformed["X_train"][:, 6:10].mean(axis=0), 0, atol=1e-6),
                  "Scaled training engineered features should have zero mean")
            # Evaluation must consume the persisted training transform verbatim,
            # regardless of the synthetic validation/test distributions.
            reapplied, reapplied_meta = apply_feature_policy(
                synthetic_arrays(), cfg["feature_mode"], stored_transform=metadata
            )
            check(reapplied_meta == metadata, "Stored transform metadata changed on evaluation")
            for split in ("train", "val", "test"):
                check(np.array_equal(reapplied[f"X_{split}"], transformed[f"X_{split}"]),
                      f"{split} did not use the exact stored feature transform")

    check(len(set(roots)) == 6, "Experiment training/evaluation roots must all be unique")

    # Ensure a seven-feature entry point cannot accept the standard config and
    # the evaluator cannot route into another variant's checkpoint directory.
    standard = yaml.safe_load((ROOT / EXPERIMENTS["flower15"]["config"]).read_text())
    try:
        validate_config({**base, **standard}, "balance_free_7", "standard config")
    except ValueError:
        pass
    else:
        raise AssertionError("Pretransaction configuration silently accepted standard config")

    try:
        validate_model_directory(EXPERIMENTS["flower15"]["out_dir"], "balance_free_7")
    except ValueError:
        pass
    else:
        raise AssertionError("Pretransaction evaluator accepted standard model directory")

    # A custom path is allowed only below its own registered model root.
    validate_model_directory(
        Path(EXPERIMENTS["balance_free_7"]["out_dir"]) / "runs" / "synthetic",
        "balance_free_7",
    )
    for unsafe in (
        ROOT / "results" / "unrelated_models",
        Path(EXPERIMENTS["flower15"]["out_dir"]) / "nested_custom",
        Path(EXPERIMENTS["balance_free_7"]["global_test_out_dir"]) / "models",
    ):
        try:
            validate_model_directory(unsafe, "balance_free_7")
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsafe custom model path was accepted: {unsafe}")

    try:
        validate_evaluation_output(EXPERIMENTS["flower15"]["global_test_out_dir"], "balance_free_7")
    except ValueError:
        pass
    else:
        raise AssertionError("Pretransaction evaluator accepted standard evaluation output")

    validate_evaluation_output(
        Path(EXPERIMENTS["balance_free_7"]["global_test_out_dir"]) / "runs" / "synthetic",
        "balance_free_7",
    )
    try:
        validate_evaluation_output(ROOT / "results" / "unrelated_eval", "balance_free_7")
    except ValueError:
        pass
    else:
        raise AssertionError("Evaluator accepted an unrelated custom output path")

    try:
        ensure_new_outputs([Path(__file__)], "synthetic collision check")
    except FileExistsError:
        pass
    else:
        raise AssertionError("Output guard failed to reject an existing file")

    print("Local-only experiment validation passed (synthetic data only; no files written).")


if __name__ == "__main__":
    main()
