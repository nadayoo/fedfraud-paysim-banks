"""Experiment definitions and safeguards for Member 2's local-only runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


FEATURE_NAMES_15 = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "origin_delta", "dest_delta",
    "origin_error", "dest_error", "type_CASH_IN", "type_CASH_OUT",
    "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
]
FEATURE_COLUMNS_7 = [0, 1, 10, 11, 12, 13, 14]

EXPERIMENTS = {
    "flower15": {
        "experiment_name": "flower_style_15_feature",
        "feature_mode": "flower15",
        "feature_count": 15,
        "feature_names": FEATURE_NAMES_15,
        "out_dir": "results/local_only_flower15",
        "global_test_out_dir": "results/local_only_flower15_global_test",
        "config": "local_only/configs/local_only.yaml",
    },
    "balance_free_7": {
        "experiment_name": "balance_free_7_feature",
        "feature_mode": "balance_free_7",
        "feature_count": 7,
        "feature_names": [FEATURE_NAMES_15[i] for i in FEATURE_COLUMNS_7],
        "out_dir": "results/local_only_pretransaction",
        "global_test_out_dir": "results/local_only_pretransaction_global_test_v2",
        "config": "local_only/configs/local_only_pretransaction.yaml",
    },
    "scaled_engineered_15": {
        "experiment_name": "scaled_engineered_15_feature_ablation",
        "feature_mode": "scaled_engineered_15",
        "feature_count": 15,
        "feature_names": FEATURE_NAMES_15,
        "out_dir": "results/local_only_scaled_v2",
        "global_test_out_dir": "results/local_only_scaled_v2_global_test",
        "config": "local_only/configs/local_only_scaled.yaml",
    },
}


def _resolved(path: str | Path) -> Path:
    path = Path(path)
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _overlaps(path: Path, root: Path) -> bool:
    return _is_within(path, root) or _is_within(root, path)


def validate_config(cfg: dict, expected_experiment: str, config_path: str | Path) -> dict:
    """Reject mismatched experiment, feature count, or output-directory routing."""
    if expected_experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown local-only experiment: {expected_experiment}")
    spec = EXPERIMENTS[expected_experiment]
    if cfg.get("experiment_name") != spec["experiment_name"]:
        raise ValueError(
            f"Configuration {config_path} selects {cfg.get('experiment_name')!r}; "
            f"expected {spec['experiment_name']!r}. Refusing to fall back to another experiment."
        )
    if cfg.get("feature_mode") != spec["feature_mode"]:
        raise ValueError(f"{config_path}: feature_mode must be {spec['feature_mode']!r}")
    if cfg.get("feature_count") != spec["feature_count"]:
        raise ValueError(
            f"{config_path}: feature_count must be {spec['feature_count']}, "
            f"got {cfg.get('feature_count')!r}"
        )
    if cfg.get("feature_names") != spec["feature_names"]:
        raise ValueError(f"{config_path}: feature_names/order do not match the experiment definition")

    expected_train_root = _resolved(spec["out_dir"])
    expected_eval_root = _resolved(spec["global_test_out_dir"])
    output = _resolved(cfg["out_dir"])
    eval_output = _resolved(cfg["global_test_out_dir"])
    if output != expected_train_root or eval_output != expected_eval_root:
        raise ValueError(
            f"{config_path}: configured output roots must remain the registered roots for "
            f"{spec['experiment_name']} ({expected_train_root}, {expected_eval_root})"
        )
    if output == eval_output or output in eval_output.parents or eval_output in output.parents:
        raise ValueError("Training and global-evaluation output directories must be disjoint")
    for other_name, other in EXPERIMENTS.items():
        if other_name == expected_experiment:
            continue
        other_roots = (_resolved(other["out_dir"]), _resolved(other["global_test_out_dir"]))
        for other_root in other_roots:
            if _overlaps(output, other_root) or _overlaps(eval_output, other_root):
                raise ValueError(
                    f"Output directory for {expected_experiment} overlaps {other_name}: {other_root}"
                )
    return spec


def apply_feature_policy(
    data: dict,
    feature_mode: str,
    *,
    stored_transform: dict | None = None,
) -> tuple[dict, dict]:
    """Apply one feature policy consistently to train, validation, and test.

    For the scaled ablation, training fits the transform when ``stored_transform``
    is absent. Evaluation supplies the saved transform and applies those exact
    means and standard deviations to every split.
    """
    expected = 15
    for split in ("train", "val", "test"):
        x = data[f"X_{split}"]
        if x.ndim != 2 or x.shape[1] != expected:
            raise ValueError(
                f"Expected the shared 15-feature matrix before policy application; "
                f"X_{split} has shape {x.shape}"
            )

    result = dict(data)
    transform = {"mode": feature_mode}
    if feature_mode == "flower15":
        columns = list(range(15))
        names = FEATURE_NAMES_15
    elif feature_mode == "balance_free_7":
        columns = FEATURE_COLUMNS_7
        names = [FEATURE_NAMES_15[i] for i in columns]
    elif feature_mode == "scaled_engineered_15":
        columns = list(range(15))
        names = FEATURE_NAMES_15
        if stored_transform is None:
            means = data["X_train"][:, 6:10].mean(axis=0, dtype=np.float64)
            stds = data["X_train"][:, 6:10].std(axis=0, dtype=np.float64)
            stds[stds < 1e-12] = 1.0
        else:
            if stored_transform.get("mode") != feature_mode:
                raise ValueError("Stored feature transform belongs to another experiment")
            if stored_transform.get("columns") != [6, 7, 8, 9]:
                raise ValueError("Stored engineered-feature columns do not match [6, 7, 8, 9]")
            means = np.asarray(stored_transform.get("means"), dtype=np.float64)
            stds = np.asarray(stored_transform.get("stds"), dtype=np.float64)
            if (means.shape != (4,) or stds.shape != (4,)
                    or not np.isfinite(means).all() or not np.isfinite(stds).all()
                    or (stds <= 0).any()):
                raise ValueError("Stored scaling statistics must contain four finite means and positive stds")
        transform.update(means=means.tolist(), stds=stds.tolist(), columns=[6, 7, 8, 9])
        for split in ("train", "val", "test"):
            x = data[f"X_{split}"].copy()
            x[:, 6:10] = ((x[:, 6:10].astype(np.float64) - means) / stds).astype(np.float32)
            result[f"X_{split}"] = x
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode!r}")

    if feature_mode == "balance_free_7":
        for split in ("train", "val", "test"):
            result[f"X_{split}"] = data[f"X_{split}"][:, columns].copy()

    if any(result[f"X_{s}"].shape[1] != len(names) for s in ("train", "val", "test")):
        raise ValueError("Feature transformation produced inconsistent split dimensions")
    transform["feature_names"] = list(names)
    transform["feature_count"] = len(names)
    return result, transform


def validate_data_dimension(data: dict, feature_count: int, where: str) -> None:
    for split in ("train", "val", "test"):
        actual = data[f"X_{split}"].shape[1]
        if actual != feature_count:
            raise ValueError(f"{where}: X_{split} has {actual} features; expected {feature_count}")


def state_dict_input_dim(state: dict) -> int:
    """Read the first linear layer's input dimension without constructing a model."""
    key = "net.0.weight"
    if key not in state or state[key].ndim != 2:
        raise ValueError(f"Checkpoint is missing the expected first-layer weight {key!r}")
    return int(state[key].shape[1])


def file_sha256(path: str | Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_new_outputs(paths: list[Path], context: str) -> None:
    """Reject existing output roots, files, directories, and symlinks.

    Existing ancestors are allowed (for example, a new run directory below an
    experiment root); every requested destination itself must be unused.
    """
    collisions = [str(p) for p in paths if p.exists() or p.is_symlink()]
    if collisions:
        raise FileExistsError(
            f"Refusing to overwrite existing {context} output(s): " + ", ".join(collisions)
        )


def validate_evaluation_output(path: str | Path, expected_experiment: str) -> None:
    output = _resolved(path)
    spec = EXPERIMENTS[expected_experiment]
    expected_root = _resolved(spec["global_test_out_dir"])
    if not _is_within(output, expected_root):
        raise ValueError(
            f"Evaluation output must be {expected_root} or a new path beneath it, got {output}"
        )
    model_root = _resolved(spec["out_dir"])
    if _overlaps(output, model_root):
        raise ValueError("Global-evaluation output must be separate from model checkpoints")
    for other_name, other in EXPERIMENTS.items():
        if other_name == expected_experiment:
            continue
        for root in (_resolved(other["out_dir"]), _resolved(other["global_test_out_dir"])):
            if _overlaps(output, root):
                raise ValueError(f"Evaluation output overlaps {other_name}: {root}")


def validate_model_directory(path: str | Path, expected_experiment: str) -> None:
    model_root = _resolved(path)
    spec = EXPERIMENTS[expected_experiment]
    expected_root = _resolved(spec["out_dir"])
    if not _is_within(model_root, expected_root):
        raise ValueError(
            f"Model directory must be {expected_root} or a path beneath it for "
            f"{spec['experiment_name']}, got {model_root}"
        )
    for other_name, other in EXPERIMENTS.items():
        if other_name == expected_experiment:
            continue
        for root in (_resolved(other["out_dir"]), _resolved(other["global_test_out_dir"])):
            if _overlaps(model_root, root):
                raise ValueError(f"Model directory overlaps the {other_name} experiment: {root}")
    if _overlaps(model_root, _resolved(spec["global_test_out_dir"])):
        raise ValueError("Model directory cannot be this experiment's global-evaluation output")


def config_sha256(config_path: str | Path) -> str:
    return hashlib.sha256(Path(config_path).read_bytes()).hexdigest()


def json_safe_config(cfg: dict) -> dict:
    return json.loads(json.dumps(cfg, default=str))
