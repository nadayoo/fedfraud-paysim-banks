# Member 2: Local-Only Machine Learning

This module trains five separate PyTorch models. Each model trains on one bank's
training rows, selects its checkpoint and classification threshold from that
bank's validation rows, and is evaluated on both that bank's test rows and the
complete global test split.

## Experiment variants

| Variant | Features, in order | Config | Training output | Global-test output |
|---|---|---|---|---|
| Flower-style reproduction | `step`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `origin_delta`, `dest_delta`, `origin_error`, `dest_error`, then the five transaction-type one-hot columns | `configs/local_only.yaml` | `results/local_only_flower15/` | `results/local_only_flower15_global_test/` |
| Balance-free seven-feature experiment | `step`, `amount`, `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` | `configs/local_only_pretransaction.yaml` | `results/local_only_pretransaction/` | `results/local_only_pretransaction_global_test_v2/` |
| Balance-feature scaling ablation | Same 15 features as Flower-style; engineered balance features at columns 6–9 standardized with `X_train` statistics | `configs/local_only_scaled.yaml` | `results/local_only_scaled_v2/` | `results/local_only_scaled_v2_global_test/` |

For the scaled ablation, only the training split is used to calculate means and
standard deviations. The saved transform is applied unchanged to validation
and test. The 15-feature reproduction does not apply that extra scaling.

## Transaction-type mix and bank performance

In the cached training rows, transaction-type shares are very similar across
the five banks: CASH_IN 21.94–22.07%, CASH_OUT 35.05–35.25%, DEBIT
0.65–0.66%, PAYMENT 33.76–33.89%, and TRANSFER 8.31–8.41%. These spreads are
no greater than 0.20 percentage points. However, Experiment A global-test
PR-AUC by model bank ranges from 21.62% to 37.88%, while fresh Experiment B
ranges from 22.61% to 24.64%. Transaction-type proportions alone therefore do
not explain A’s bank-level variation. This is descriptive, not causal; bank
volume, fraud prevalence, numeric feature distributions, and interactions can
also affect results.

## Training

Run commands from the repository root. The default feature policy is fixed by
each entry point; passing a config for another variant fails before training.

```bash
# Experiment A — Flower-style 15-feature baseline; fresh run directory
python -m local_only.train --seeds 0 1 2 --out-dir results/local_only_flower15/runs/member2_submission_20261123

# Experiment B — pretransaction entry point defaults to its seven-feature config
CUDA_VISIBLE_DEVICES="" python -m local_only.train_pretransaction --seeds 0 1 2 --out-dir results/local_only_pretransaction/runs/member2_submission_20261123

# Experiment C — balance-feature scaling ablation; fresh run directory
python -m local_only.train_scaled --seeds 0 1 2 --out-dir results/local_only_scaled_v2/runs/member2_submission_20261123
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
python -m local_only.evaluate_global --seeds 0 1 2 --model-dir results/local_only_flower15/runs/member2_submission_20261123 --out-dir results/local_only_flower15_global_test/runs/member2_submission_20261123
python -m local_only.evaluate_pretransaction_global --seeds 0 1 2 --model-dir results/local_only_pretransaction/runs/member2_submission_20261123 --out-dir results/local_only_pretransaction_global_test_v2/runs/member2_submission_20261123
python -m local_only.evaluate_scaled_global --seeds 0 1 2 --model-dir results/local_only_scaled_v2/runs/member2_submission_20261123 --out-dir results/local_only_scaled_v2_global_test/runs/member2_submission_20261123
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

An additional complete B training run is preserved at
`results/local_only_pretransaction/runs/member2_final_20261122/`. Its 15 runs
match the selected `member2_final_20261122_run2` training metrics. The report uses
`member2_final_20261122_run2` because that directory has the paired global-test
evaluation as well.

Legacy outputs under the original result roots are preserved and are not
retroactively labeled as valid. The older B artifacts without provenance remain
separate from the new `runs/` outputs. Evaluators accept only completed runs with
matching provenance metadata. The
shared `common.data` preprocessing derives numerical normalization statistics
from combined training partitions. Validation and test are not used to fit
those statistics, but local-only models therefore share training-derived
preprocessing statistics across banks.

Experiment D is an unexecuted Flower Hub follow-on and is excluded from the reported local-only results.
