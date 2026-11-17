"""Global evaluation for balance-free pre-transaction local models."""

import local_only.evaluate_global as evaluator

FEATURE_COLUMNS = [0, 1, 10, 11, 12, 13, 14]

original_load = evaluator.load_or_build

def load_pretransaction_data(*args, **kwargs):
    data = dict(original_load(*args, **kwargs))

    for split in ("train", "val", "test"):
        key = f"X_{split}"
        data[key] = data[key][:, FEATURE_COLUMNS].copy()

    return data

evaluator.load_or_build = load_pretransaction_data
evaluator.INPUT_DIM = len(FEATURE_COLUMNS)

if __name__ == "__main__":
    evaluator.main()
