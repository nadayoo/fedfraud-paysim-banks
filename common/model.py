"""The MLP used by ALL experiments (centralized, local-only, federated).

Mirrors the Flower Hub fed-fin-fraud app (task.py, app version 1.0.0):
two hidden blocks of Linear -> LayerNorm -> ReLU -> Dropout, then one logit.
With input_dim=15, hidden 128/64 this has 10,753 parameters (paper: ~10.75K).
"""
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, input_dim: int, hidden_dim_1: int = 128, hidden_dim_2: int = 64,
                 dropout: float = 0.1, use_layernorm: bool = True):
        super().__init__()
        layers = []
        for d_in, d_out in ((input_dim, hidden_dim_1), (hidden_dim_1, hidden_dim_2)):
            layers.append(nn.Linear(d_in, d_out))
            if use_layernorm:
                layers.append(nn.LayerNorm(d_out))
            layers += [nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim_2, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # raw logit, use BCEWithLogitsLoss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
