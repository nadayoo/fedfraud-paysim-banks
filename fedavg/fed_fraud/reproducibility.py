"""Deterministic random-number setup for federated experiments."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import os
import random


MAX_SEED = 2**32 - 1


def validate_seed(seed: int) -> int:
    """Return a seed accepted by Python, NumPy, and PyTorch RNGs."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}, got {seed}")
    return seed


def derive_seed(base_seed: int, *components: object) -> int:
    """Derive a stable 32-bit seed without relying on Python's salted hash."""
    base_seed = validate_seed(base_seed)
    payload = "\0".join([str(base_seed), *(str(item) for item in components)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (MAX_SEED + 1)


def client_seed(
    base_seed: int,
    node_config: Mapping[str, object],
    server_round: int,
) -> int:
    """Derive an execution-order-independent seed for a client training round."""
    if "partition-id" in node_config:
        client_identity = f"partition:{node_config['partition-id']}"
    elif "data-path" in node_config:
        client_identity = f"data-path:{node_config['data-path']}"
    else:
        raise ValueError(
            "A deterministic client seed requires 'partition-id' or 'data-path' "
            "in node_config."
        )
    return derive_seed(base_seed, "client", client_identity, int(server_round))


def seed_everything(seed: int) -> int:
    """Seed all RNGs and require deterministic PyTorch kernels."""
    seed = validate_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)

    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return seed


class DeterministicGrid:
    """Order node discovery and replies while forwarding other Grid operations.

    A fixed node-ID-to-partition mapping is required across repeated runs.
    """

    def __init__(self, grid):
        self._grid = grid

    def get_node_ids(self) -> list[int]:
        """Give Flower's seeded sampler a stable population order."""
        return sorted(self._grid.get_node_ids())

    def send_and_receive(self, messages, *, timeout=None):
        """Aggregate replies in node order, independent of arrival timing."""
        return sorted(
            self._grid.send_and_receive(messages, timeout=timeout),
            key=lambda message: message.metadata.src_node_id,
        )

    def __getattr__(self, name):
        return getattr(self._grid, name)
