
## Member 5: dataset analysis and non-IID experiments (2026-11-28)

Design decisions for Parts A, B and C. Details are in `docs/dataset_analysis.md` and `docs/partitioning_methodology.md`.

### Repository and code organisation

- **Own branch.** Work is on branch `noniid-analysis`, created from `centralized-mlp` (the shared branch that holds `common/`). Only new files were added in `noniid_analysis/`, `results/noniid/` and `docs/`; nothing in `common/` or `centralized/` was changed.
- **Member 3's code is not modified.** `fedavg/` (branch `FedAvg`) is untouched. For the experiments a copy of the app lives in `noniid_analysis/flwr_app/`. The only change to Member 3's code is in the copy: one import line and a 3-line block in `fed_fraud/task.py` (`_make_partitioner`) that accepts partitioner names starting with `lookup_`.
- **Flower environment.** A separate Python 3.12 environment is used for Flower (Flower simulation did not run on Python 3.14 in our setup because Ray was not available). The analysis scripts use the project's own `.venv`.

### Partitions (Part B)

- **Only `BankID` changes.** Every partition reuses the same training rows, so preprocessing, validation split and scaler are identical across conditions. The test set is never partitioned.
- **IID is stratified and keeps the original bank sizes.** Rows are stratified by (`isFraud`, `type`) and split in the original size proportions. This differs from Flower's built-in `iid` partitioner (equal sizes, plain random), which was not used. Reason: isolate the effect of the data distribution from the effect of bank size.
- **Fraud skew.** Original bank sizes, fraud cases distributed 50/25/15/7/3% to banks 0 to 4. Reason: a single-factor change (fraud prevalence only) with every bank still holding fraud cases (smallest: 222).
- **Type skew.** Each transaction type is split across banks with Dirichlet(alpha = 0.5) proportions, redrawn until every bank has at least 100 fraud cases. Known limitation: because fraud only occurs in CASH_OUT and TRANSFER, this also changes bank sizes (8.1x) and fraud rates (40x), so its results are a combined effect.
- **Seed 0 for all partitions**, generated deterministically from the seed.
- **Rows are matched by content, not by position.** Flower Datasets shuffles before splitting, so a row-order assignment would put transactions in the wrong banks (verified). Each partition is stored as a lookup from a hash of the ten non-label columns (unique per row) to the bank, and a `LookupPartitioner` applies it. The `original` lookup reproduces the dataset's own banks exactly.
- **Hashing in chunks of 200,000 rows.** Hashing all rows at once with five parallel clients exhausted the 15 GB of RAM of the development laptop.

### Experiments (Part C)

- **Same configuration as Member 3's `pyproject.toml`** (FedAvg, 20 rounds, 5 clients all participating, 2 local epochs, batch size 1024), seed 42.
- **The original partition was rerun** through the same lookup mechanism (`lookup_original_seed0`) instead of reusing Member 3's runs, so all four conditions use identical code paths and are directly comparable.
- **The number of simulated clients must be set explicitly to 5.** The Flower default is 2. Two early runs (fraud skew, type skew) used 2 clients and were discarded; they were detected because their communication files showed 177,816 bytes per round instead of 444,540. `run_partitions.ps1` now sets 5 clients before every run.
- **One seed per condition.** The task asks for three seeds where computationally feasible; full runs take hours on a CPU laptop, so only seed 42 was run. No mean or standard deviation is reported, and differences of about 0.01 PR-AUC are not claimed as effects.
- **PR-AUC and ROC-AUC are the main comparison metrics.** F1 at threshold 0.5 is reported but is not meaningful on its own (precision at that threshold is about 0.025).
- **Open question.** Type skew collapsed to PR-AUC about 0.03 at round 11 and recovered. It is unknown whether this repeats; more seeds would be needed to say.
