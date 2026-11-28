"""Teach fed_fraud.task to accept partition names starting with 'lookup_'."""
import os
import time
from pathlib import Path

with open(Path.home() / "fedfraud_patch.log", "a") as _f:
    _f.write(f"{time.strftime('%H:%M:%S')} patch imported in pid {os.getpid()}\n")

from fed_fraud import task
from fed_fraud_noniid.lookup_partitioner import LookupPartitioner

if not getattr(task, "_noniid_patched", False):
    _original = task._make_partitioner

    def _make_partitioner(partitioner_name: str, num_partitions: int):
        name = partitioner_name.strip().lower()
        if name.startswith("lookup_"):
            print(f"[noniid] using lookup partition {name}", flush=True)
            return LookupPartitioner(name, num_partitions)
        return _original(partitioner_name, num_partitions)

    task._make_partitioner = _make_partitioner
    task._noniid_patched = True
