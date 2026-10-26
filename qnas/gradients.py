"""Gradient analysis for RQ3 (trainability / barren plateaus).

Barren plateau (McClean et al. 2018): for sufficiently random deep circuits the gradient of the cost with
respect to any parameter has mean 0 and a variance that shrinks *exponentially* with the number of qubits,
so training gets stuck on an exponentially flat landscape.  We measure this directly:

``barren_plateau_scan``   Var[dC/dtheta] versus number of qubits for template circuits (the classic plot).
``candidate_gradients``   gradient statistics of concrete circuits (evolved vs hand-crafted) at their native size,
                          at random initial parameters (``uniform`` in [0, 2pi) = the standard "random init"
                          where barren plateaus appear, and ``small`` = near-identity init).

All per-sample gradients come from ONE batched forward/backward pass: every row of the (B, P) weight matrix is
an independent random initialisation, and because rows do not interact, ``out.sum().backward()`` yields each
row's own gradient.
"""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/gradients.py
# Purpose: measuring gradient sizes to study trainability and barren plateaus (RQ3).
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .circuit import make_qnode
from .genome import Gene, count_trainable
from .templates import strongly_entangling

# Using double precision: the variances measured here can be as small as 1e-5.
DTYPE = torch.float64


def sample_gradients(
    genes: Sequence[Gene],
    n_qubits: int,
    n_features: int,
    inputs: np.ndarray,
    init: str = "uniform",
    observable: str = "z0",
    seed: int = 0,
    init_std: float = 0.3,
) -> np.ndarray:
    """Gradients ``d<O>/d weights`` for ``len(inputs)`` random initialisations -> array ``(B, P)``."""
    # Returning an empty matrix for circuits with nothing to train.
    p = count_trainable(genes)
    if p == 0:
        return np.zeros((len(inputs), 0))
    # Drawing one independent weight vector per row: uniform in [0, 2pi) or small normal.
    g = torch.Generator().manual_seed(seed)
    b = len(inputs)
    if init == "uniform":
        w = torch.rand(b, p, generator=g, dtype=DTYPE) * 2 * np.pi
    elif init == "small":
        w = torch.randn(b, p, generator=g, dtype=DTYPE) * init_std
    else:
        raise ValueError("init must be 'uniform' or 'small'")
    # Running one batched forward / backward pass; rows do not interact, so each
    # row of w.grad is that initialisation's own gradient.
    w.requires_grad_(True)
    x = torch.as_tensor(np.asarray(inputs), dtype=DTYPE)
    qnode = make_qnode(genes, n_qubits, n_features, observable=observable)
    out = qnode(x, w)
    out.sum().backward()
    return w.grad.detach().numpy()


def grad_stats(grads: np.ndarray) -> Dict[str, float]:
    # Summarising a (samples x parameters) gradient matrix; NaN when there are no parameters.
    if grads.shape[1] == 0:
        return {"n_params": 0, "var_mean": float("nan"), "var_first": float("nan"), "abs_mean": float("nan"),
                "norm_mean": float("nan"), "frac_small": float("nan")}
    return {
        "n_params": int(grads.shape[1]),
        "var_mean": float(grads.var(axis=0).mean()),   # averaging, over parameters, the variance across random inits
        "var_first": float(grads[:, 0].var()),         # tracking the first parameter, as McClean et al. plot it
        "abs_mean": float(np.abs(grads).mean()),
        "norm_mean": float(np.linalg.norm(grads, axis=1).mean()),
        "frac_small": float((np.abs(grads) < 1e-3).mean()),
    }


# ------------------------------------------------------------------------------
# 1. Reproducing the classic barren-plateau scaling experiment on templates
# ------------------------------------------------------------------------------
def default_scan_templates() -> Dict[str, Callable[[int], List[Gene]]]:
    # Comparing a fixed-depth template with one whose depth grows with the qubit count.
    return {
        "SEL, 2 layers (shallow)": lambda n: strongly_entangling(n, 2),
        "SEL, n layers (depth grows with qubits)": lambda n: strongly_entangling(n, n),
    }


def barren_plateau_scan(
    qubits: Sequence[int] = (2, 4, 6, 8),
    templates: Optional[Dict[str, Callable[[int], List[Gene]]]] = None,
    costs: Sequence[str] = ("z0", "global_z"),
    n_samples: int = 100,
    seed: int = 0,
    log: Callable[[str], None] = lambda s: None,
) -> List[dict]:
    # Looping over qubit counts, templates and cost functions, logging Var[grad] for each.
    templates = templates or default_scan_templates()
    rng = np.random.default_rng(seed)
    rows = []
    for n in qubits:
        inputs = rng.uniform(0, np.pi, size=(n_samples, n))
        for tname, make in templates.items():
            genes = make(n)
            for cost in costs:
                grads = sample_gradients(genes, n, n, inputs, "uniform", cost, seed=seed + n)
                st = grad_stats(grads)
                rows.append({"n_qubits": n, "template": tname, "cost": cost, **st})
                log(f"  n={n} {tname:<42} cost={cost:<8} Var[grad]={st['var_mean']:.3e}")
    return rows


# ------------------------------------------------------------------------------
# 2. Comparing gradient statistics of concrete circuits (evolved vs hand-crafted)
# ------------------------------------------------------------------------------
def candidate_gradients(
    circuits: Dict[str, Sequence[Gene]],
    n_qubits: int,
    n_features: int,
    x_pool: np.ndarray,
    n_samples: int = 200,
    seed: int = 0,
    observable: str = "z0",
    inits: Sequence[str] = ("uniform", "small"),
) -> List[dict]:
    # Sampling input vectors from the real data, then measuring every circuit under both inits.
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_pool), size=n_samples, replace=len(x_pool) < n_samples)
    inputs = np.asarray(x_pool)[idx]
    rows = []
    for name, genes in circuits.items():
        for init in inits:
            grads = sample_gradients(genes, n_qubits, n_features, inputs, init, observable, seed=seed)
            rows.append({"circuit": name, "init": init, **grad_stats(grads)})
    return rows
