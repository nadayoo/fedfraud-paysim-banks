"""Balance-free pre-transaction local-only experiment."""

import local_only.train as trainer

# Existing shared preprocessing:
# 0 = standardized step
# 1 = standardized amount
# 10-14 = one-hot transaction type
FEATURE_COLUMNS = [0, 1, 10, 11, 12, 13, 14]

original_load = trainer.load_or_build

def load_pretransaction_data(*args, **kwargs):
    data = dict(original_load(*args, **kwargs))

    for split in ("train", "val", "test"):
        key = f"X_{split}"
        data[key] = data[key][:, FEATURE_COLUMNS].copy()

    return data

trainer.load_or_build = load_pretransaction_data
trainer.INPUT_DIM = len(FEATURE_COLUMNS)

if __name__ == "__main__":
    trainer.main()
