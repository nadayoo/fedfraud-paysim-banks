# Member 2: Local-Only Machine Learning

This module trains five separate PyTorch models. Each model trains on one bank's
training rows, selects its checkpoint and classification threshold from that
bank's validation rows, and is evaluated on both that bank's test rows and the
complete global test split.

## Experiment variants

| Variant | Features, in order | Config | Training output | Global-test output |
|---|---|---|---|---|
| Flower-style reproduction | `step`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `origin_delta`, `dest_delta`, `origin_error`, `dest_error`, then the five transaction-type one-hot columns | `configs/local_only.yaml` | `results/local_only_flower15/` | `results/local_only_flower15_global_test/` |
| Balance-free pretransaction | `step`, `amount`, `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` | `configs/local_only_pretransaction.yaml` | `results/local_only_pretransaction/` | `results/local_only_pretransaction_global_test_v2/` |
| Scaled engineered-feature ablation | Same 15 features as Flower-style; columns 6–9 standardized with `X_train` statistics | `configs/local_only_scaled.yaml` | `results/local_only_scaled_v2/` | `results/local_only_scaled_v2_global_test/` |

For the scaled ablation, only the training split is used to calculate means and
standard deviations. The saved transform is applied unchanged to validation
and test. The 15-feature reproduction does not apply that extra scaling.

## Training

Run commands from the repository root. The default feature policy is fixed by
each entry point; passing a config for another variant fails before training.

```bash
# Experiment A — defaults to the 15-feature config and fresh output path
python -m local_only.train --seeds 0 1 2

# Experiment B — pretransaction entry point defaults to its seven-feature config
python -m local_only.train_pretransaction --seeds 0 1 2 --out-dir results/local_only_pretransaction/runs/member2_run_001

# Experiment C — scaled 15-feature ablation
python -m local_only.train_scaled --seeds 0 1 2
```

Training refuses an existing output root or any existing requested run,
checkpoint, result, history, plot, or summary path. Custom output paths must be
new descendants of the corresponding variant's registered root; paths from
another experiment, arbitrary sibling directories, and model/evaluation root
overlaps are rejected. Use a new unused leaf for any rerun. Training prints
the selected experiment, feature count, configuration path, and output
directory before loading data.

Each run writes a checkpoint, history, plots, and `results.json`. The result
includes the config content and SHA-256, cache path/fingerprint when available,
feature names/order and transform statistics, model input dimension, training
parameters, best validation PR-AUC, validation-selected threshold, test metrics,
and completion status. Bank-level summaries are labeled as bank-specific test
results; seed-level result files retain the six metrics and threshold details.

## Global-test evaluation

The standard evaluator uses the standard configuration/checkpoint directory by
default. The wrappers for the other variants select their own config and require
matching experiment metadata and checkpoint dimensions.

```bash
python -m local_only.evaluate_global --seeds 0 1 2
python -m local_only.evaluate_pretransaction_global --seeds 0 1 2 --model-dir results/local_only_pretransaction/runs/member2_run_001 --out-dir results/local_only_pretransaction_global_test_v2/runs/member2_run_001
python -m local_only.evaluate_scaled_global --seeds 0 1 2
```

Global results include each local model's metrics on the complete test set and
the existing destination-bank breakdown, including PR-AUC, ROC-AUC and
threshold-based metrics. Evaluators accept checkpoints only inside their own
registered model root and write only to a fresh path beneath that variant's
global-test root. They require completed-run metadata, check configuration and
dataset-cache SHA-256 fingerprints, validate actual checkpoint dimensions
against metadata, and use the saved training-fitted transform for the scaled
ablation. Historical runs without verifiable provenance are rejected and
remain untouched.

## Lightweight validation

```bash
python -m local_only.validate_experiments
```

This uses synthetic arrays only. It checks all three feature counts, model input
dimensions, wrapper/config routing, output-directory isolation, seven-feature
column selection, and that scaled validation/test arrays use the training-fit
transform. It does not load PaySim or run training.

## Historical artifacts

Files already present under `results/local_only/`, `results/local_only_global_test/`,
`results/local_only_pretransaction/`, `results/local_only_pretransaction_global_test/`,
`results/local_only_scaled/`, and `results/local_only_scaled_global_test/` are
preserved. Their provenance was not established by this code change. They are
not rewritten or retroactively labeled as valid. The new evaluator requires
metadata created by a new training run before it will accept a checkpoint. The
shared `common.data` preprocessing derives numerical normalization statistics
from combined training partitions. Validation and test are not used to fit
those statistics, but local-only models therefore share training-derived
preprocessing statistics across banks.
