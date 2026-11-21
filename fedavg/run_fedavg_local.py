"""
    python run_fedavg_local.py --smoke          # 1 round, checks everything
    python run_fedavg_local.py                  # seeds 42 43 44 x 20 rounds, then summary
    python run_fedavg_local.py --seeds 42       # one seed only
"""

import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from datasets import load_dataset 
from flwr_datasets import FederatedDataset  

from fed_fraud import task  
from fed_fraud.reproducibility import client_seed, seed_everything  

NUM_BANKS = 5
CACHE_DIR = HERE / "feature_cache"

# ---------------------------------------------------------------- feature cache

_names: dict[int, str] = {}
_mem: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_original_to_arrays = task._hf_dataset_to_arrays


def _cached_to_arrays(ds):
    name = _names[id(ds)]
    if name in _mem:
        return _mem[name]
    path = CACHE_DIR / f"{name}.npz"
    if path.exists():
        z = np.load(path)
        out = (z["x"], z["y"])
    else:
        print(f"[cache] computing features for {name} ({len(ds):,} rows), one-time cost ...", flush=True)
        t0 = time.time()
        out = _original_to_arrays(ds)
        CACHE_DIR.mkdir(exist_ok=True)
        np.savez(path, x=out[0], y=out[1])
        print(f"[cache] {name} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    _mem[name] = out
    return out


task._hf_dataset_to_arrays = _cached_to_arrays


def load_cfg() -> dict:
    with open(HERE / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["tool"]["flwr"]["app"]["config"]


def make_model(cfg: dict) -> torch.nn.Module:
    return task.Net(
        input_dim=task.get_input_dim(),
        hidden_dim_1=int(cfg["hidden-dim-1"]),
        hidden_dim_2=int(cfg["hidden-dim-2"]),
        dropout=float(cfg["dropout"]),
    )


def evaluate_global(state, cfg, device, test_loader) -> dict:
    model = make_model(cfg)
    model.load_state_dict(state)
    loss, metrics = task.test(model, test_loader, device)
    rec = {"centralized_loss": float(loss)}
    for thr, sub in metrics.items():
        for name, value in sub.items():
            rec[f"{name}_at_{thr}"] = float(value)
    return rec


def run_one(seed: int, rounds: int, name: str, cfg: dict, banks, test_loader, device, out: Path) -> None:
    bs = int(cfg["batch-size"])
    epochs = int(cfg["local-epochs"])
    lr_max, lr_min = float(cfg["learning-rate-max"]), float(cfg["learning-rate-min"])
    use_cw = str(cfg["use-class-weights"]).lower() == "true"

    seed_everything(seed)
    state = {k: v.clone() for k, v in make_model(cfg).state_dict().items()}
    payload = sum(t.numel() * t.element_size() for t in state.values())  # bytes of one model copy

    hist: dict[int, dict] = {}
    comm: dict[int, dict] = {}
    start = 1
    ckpt = out / f"ckpt_{name}.pt"
    if ckpt.exists():
        d = torch.load(ckpt, weights_only=False)
        if d["rounds"] == rounds and d["seed"] == seed:
            state, hist, comm, start = d["state"], d["hist"], d["comm"], d["next_round"]
            print(f"[{name}] resuming from round {start}", flush=True)
    if 0 not in hist:
        hist[0] = evaluate_global(state, cfg, device, test_loader)

    for rnd in range(start, rounds + 1):
        t_round = time.time()
        updates, client_times, losses = [], [], []
        lr = task.cosine_annealing(rnd, rounds, lr_max, lr_min)
        for i in range(NUM_BANKS):
            seed_everything(client_seed(seed, {"partition-id": i}, rnd))
            model = make_model(cfg)
            model.load_state_dict(state)
            loader = task._build_dataloader(banks[i], batch_size=bs, shuffle=True)
            t0 = time.perf_counter()
            loss = task.train(model, loader, epochs, float(lr), device, use_class_weights=use_cw)
            client_times.append(time.perf_counter() - t0)
            n = len(loader.dataset)
            updates.append(({k: v.detach().cpu() for k, v in model.state_dict().items()}, n))
            losses.append((float(loss), n))
            print(f"[{name}] round {rnd}/{rounds} bank {i}: n={n:,} loss={loss:.4f} "
                  f"time={client_times[-1]:.0f}s", flush=True)

        # FedAvg: weighted average by number of training examples
        total = float(sum(n for _, n in updates))
        state = {
            k: (sum(sd[k].double() * n for sd, n in updates) / total).to(state[k].dtype)
            for k in state
        }
        hist[rnd] = evaluate_global(state, cfg, device, test_loader)
        comm[rnd] = {
            "comm_bytes_total": int(payload * 2 * NUM_BANKS),  # model down + model up, 5 banks
            "round_time_sec": time.time() - t_round,
            "client_train_time_sec_sum": float(sum(client_times)),
            "train_loss_weighted": sum(l * n for l, n in losses) / total,
        }
        print(f"[{name}] round {rnd}/{rounds} done in {comm[rnd]['round_time_sec'] / 60:.1f} min | "
              f"pr_auc@0.5={hist[rnd]['pr_auc_at_0.5']:.4f}", flush=True)

        # save after EVERY round (results + checkpoint)
        write_results(out, name, hist, comm)
        torch.save({"state": state, "hist": hist, "comm": comm, "next_round": rnd + 1,
                    "rounds": rounds, "seed": seed}, ckpt)


def write_results(out: Path, name: str, hist: dict, comm: dict) -> None:
    with open(out / f"result_{name}.json", "w", encoding="utf-8") as f:
        json.dump({"evaluate_metrics_serverapp": {str(r): m for r, m in hist.items()}}, f, indent=1)
    summary = {
        "totals": {
            "total_comm_bytes": sum(c["comm_bytes_total"] for c in comm.values()),
            "total_time_sec": sum(c["round_time_sec"] for c in comm.values()),
            "total_client_train_time_sec": sum(c["client_train_time_sec_sum"] for c in comm.values()),
        },
        "train_rounds": {str(r): c for r, c in comm.items()},
    }
    with open(out / f"result_{name}_communication.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--rounds", type=int, default=None, help="default: num-server-rounds from pyproject.toml (20)")
    ap.add_argument("--smoke", action="store_true", help="1 round, seed 42, name 'smoke'")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--no-aggregate", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg()
    rounds = 1 if args.smoke else (args.rounds or int(cfg["num-server-rounds"]))
    jobs = [(42, 1, "smoke")] if args.smoke else [(s, rounds, f"fedavg_seed{s}") for s in args.seeds]
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device} banks={NUM_BANKS} rounds={rounds} jobs={[j[2] for j in jobs]}", flush=True)

    partitioner = task._make_partitioner(str(cfg["partitioner"]), NUM_BANKS)
    fds = FederatedDataset(dataset=task.HF_DATASET_ID, partitioners={"train": partitioner})
    banks = []
    for i in range(NUM_BANKS):
        ds = fds.load_partition(i, "train")
        _names[id(ds)] = f"bank{i}"
        banks.append(ds)
    test_ds = load_dataset(task.HF_DATASET_ID, split="test")
    _names[id(test_ds)] = "test"
    test_loader = task._build_dataloader(test_ds, batch_size=int(cfg["batch-size"]), shuffle=False)

    for seed, r, name in jobs:
        print(f"\n===== {name}: seed={seed}, rounds={r} =====", flush=True)
        t0 = time.time()
        run_one(seed, r, name, cfg, banks, test_loader, device, out)
        print(f"[{name}] finished in {(time.time() - t0) / 60:.1f} min -> {out / ('result_' + name + '.json')}",
              flush=True)

    if not args.smoke and not args.no_aggregate:
        subprocess.run([sys.executable, str(HERE / "scripts" / "aggregate.py"),
                        "result_fedavg_seed", str(out)], cwd=HERE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
