"""Models: the hybrid quantum classifier and the classical baselines.

All models map a feature vector ``x`` (shape ``(B, n_features)``) to one logit per
sample (shape ``(B,)``) and are used with ``BCEWithLogitsLoss``.
"""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/model.py
# Purpose: wrapping an evolved circuit as a PyTorch classifier and defining the
#          classical baselines it is compared against.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import List, Sequence

import pennylane as qml
import torch
from torch import nn

from .circuit import make_qnode
from .genome import Gene, count_trainable

# Training everything in float64: the quantum simulator is double precision anyway.
DTYPE = torch.float64


class HybridQuantumModel(nn.Module):
    """features -> [fixed RY encoding -> evolved circuit] -> <Z_0> -> scale*<Z_0> + bias.

    The quantum part is wrapped with ``qml.qnn.TorchLayer`` so it behaves like any other
    PyTorch layer.  Only two classical parameters (``scale``, ``bias``) sit after it, so
    almost all of the model's learning capacity lives in the circuit structure itself.
    """

    def __init__(
        self,
        genes: Sequence[Gene],
        n_qubits: int,
        n_features: int,
        init_std: float = 0.3,
        init_scale: float = 2.0,
        device: str = "default.qubit",
        observable: str = "z0",
    ):
        super().__init__()
        # Storing the structure and building the QNode for it.
        self.genes: List[Gene] = list(genes)
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_trainable = count_trainable(self.genes)
        qnode = make_qnode(self.genes, n_qubits, n_features, device=device, observable=observable)
        # Keeping one dummy weight when the circuit has no trainable angle, because
        # TorchLayer cannot hold a zero-sized weight.
        self.qlayer = qml.qnn.TorchLayer(
            qnode,
            {"weights": (max(1, self.n_trainable),)},  # passing an explicit tuple: an int 1 would become a 0-d scalar
            init_method=lambda w: nn.init.normal_(w, mean=0.0, std=init_std),
        )
        # Adding the classical read-out: logit = scale * <Z0> + bias.
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.to(DTYPE)

    @property
    def n_parameters(self) -> int:
        """Number of *real* trainable parameters (circuit angles + scale + bias)."""
        return self.n_trainable + 2

    def gradient_parameters(self) -> list:
        # Returning only the circuit angles (used by the gradient-norm logging).
        return [p for p in self.qlayer.parameters() if p.requires_grad] if self.n_trainable else []

    def quantum_weights(self) -> torch.Tensor:
        # Copying the learned angles out, without the dummy weight if there is one.
        return self.qlayer.qnode_weights["weights"].detach().clone()[: self.n_trainable]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Running the circuit, flattening any extra axis, then applying scale and bias.
        z = self.qlayer(x)
        if z.ndim > 1:
            z = z.reshape(z.shape[0], -1)[:, 0]
        return self.scale * z + self.bias


class LinearBaseline(nn.Module):
    """Plain logistic regression on the classical features (``nn.Linear``)."""

    def __init__(self, n_features: int):
        super().__init__()
        # Using a single linear layer: the simplest possible classifier.
        self.fc = nn.Linear(n_features, 1)
        self.to(DTYPE)

    @property
    def n_parameters(self) -> int:
        # Counting every weight and bias in the network.
        return sum(p.numel() for p in self.parameters())

    def gradient_parameters(self) -> list:
        # Returning every trainable tensor (all of it is "the model" here).
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class MLPBaseline(nn.Module):
    """Tiny MLP whose parameter count is matched to a quantum model."""

    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        # Building features -> tanh hidden layer -> one logit.
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.to(DTYPE)

    @property
    def n_parameters(self) -> int:
        # Counting every weight and bias in the network.
        return sum(p.numel() for p in self.parameters())

    def gradient_parameters(self) -> list:
        # Returning every trainable tensor (all of it is "the model" here).
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mlp_hidden_for_params(n_features: int, target_params: int) -> int:
    """Hidden width whose parameter count ``(n_features + 2) * h + 1`` is closest to the target."""
    # Scanning widths and keeping the one whose size is nearest the target.
    best_h, best_gap = 2, 10**9
    for h in range(2, 512):  # starting at 2 hidden units: a single unit is degenerate and unstable
        gap = abs((n_features + 2) * h + 1 - target_params)
        if gap < best_gap:
            best_h, best_gap = h, gap
    return best_h
