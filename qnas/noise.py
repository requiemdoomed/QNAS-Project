"""Optional extension: how do trained circuits behave on *noisy* hardware?

The circuits are trained noise-free (as they would be on a simulator) and then **evaluated** on PennyLane's
``default.mixed`` density-matrix simulator with a depolarizing channel after every gate:

    single-qubit gate -> DepolarizingChannel(p)        two-qubit gate -> DepolarizingChannel(10 p) on each wire

(two-qubit gates are roughly an order of magnitude noisier than single-qubit gates on real devices).  Circuits
with fewer two-qubit gates should therefore degrade more gracefully - a direct, physical payoff of the
hardware-aware search objectives.
"""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/noise.py
# Purpose: re-scoring trained circuits on a noisy density-matrix simulator to see
#          how gracefully they degrade.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .circuit import make_qnode
from .genome import Gene, count_trainable

# Matching the precision used during training.
DTYPE = torch.float64


@torch.no_grad()
def circuit_accuracy(
    genes: Sequence[Gene],
    state: dict,
    x: np.ndarray,
    y: np.ndarray,
    n_qubits: int,
    n_features: int,
    p1: float = 0.0,
    p2: float = 0.0,
    observable: str = "z0",
) -> float:
    """Accuracy of a trained circuit (``state`` = weights/scale/bias) under depolarizing noise (p1, p2)."""
    # Switching to the mixed-state simulator only when noise is actually on.
    noise = (p1, p2) if (p1 > 0 or p2 > 0) else None
    device = "default.mixed" if noise else "default.qubit"
    qnode = make_qnode(genes, n_qubits, n_features, device=device, diff_method=None, observable=observable, noise=noise)
    # Loading the trained angles (padding the one dummy weight when there are none).
    n_w = max(1, count_trainable(genes))
    w = torch.zeros(n_w, dtype=DTYPE)
    vals = torch.as_tensor(state["weights"], dtype=DTYPE)
    w[: len(vals)] = vals
    # Applying the trained read-out and counting correct predictions.
    xt = torch.as_tensor(np.asarray(x), dtype=DTYPE)
    z = qnode(xt, w)
    logits = state["scale"] * z + state["bias"]
    pred = (logits > 0).numpy().astype(int)
    return float((pred == np.asarray(y)).mean())


def noise_sweep(
    models: Dict[str, Tuple[Sequence[Gene], dict]],
    x: np.ndarray,
    y: np.ndarray,
    n_qubits: int,
    n_features: int,
    p_values: Sequence[float] = (0.0, 0.001, 0.003, 0.01, 0.03),
    two_qubit_factor: float = 10.0,
    max_samples: int = 150,
    seed: int = 0,
    observable: str = "z0",
) -> List[dict]:
    """Accuracy of every trained circuit for each single-qubit error rate ``p`` (two-qubit rate = factor*p)."""
    # Using a fixed random subset of test images (density matrices are slow to simulate).
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(max_samples, len(x)), replace=False)
    xs, ys = np.asarray(x)[idx], np.asarray(y)[idx]
    rows = []
    for name, (genes, state) in models.items():
        for p in p_values:
            # Setting the two-qubit error rate to factor x p, capped at the fully-depolarizing 0.75.
            p2 = min(two_qubit_factor * p, 0.75)
            acc = circuit_accuracy(genes, state, xs, ys, n_qubits, n_features, p, p2, observable)
            rows.append({"circuit": name, "p_1q": p, "p_2q": p2, "accuracy": acc})
    return rows
