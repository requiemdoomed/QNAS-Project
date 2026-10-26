"""Training / evaluation utilities shared by the search, the baselines and the final comparison."""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/train.py
# Purpose: training and scoring any model (quantum or classical) with one shared loop,
#          so every comparison in the project uses identical settings.
# -----------------------------------------------------------------------------

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .model import DTYPE


@dataclass(frozen=True)
class TrainConfig:
    # Holding the optimiser settings; frozen so a config can be hashed and reused safely.
    epochs: int = 5
    batch_size: int = 64
    lr: float = 0.1
    init_std: float = 0.3     # initialising circuit angles near zero with this std
    weight_decay: float = 0.0


def as_tensor(x) -> torch.Tensor:
    # Converting numpy arrays (or lists) to float64 tensors.
    return torch.as_tensor(np.asarray(x), dtype=DTYPE)


def grad_norm(params) -> float:
    # Summing squared gradients over all tensors, then taking the square root (L2 norm).
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float((p.grad.detach() ** 2).sum())
    return total ** 0.5


@torch.no_grad()
def evaluate_model(model: nn.Module, x, y) -> Dict[str, float]:
    """Accuracy, balanced accuracy and BCE loss of ``model`` on ``(x, y)``."""
    # Predicting class 1 whenever the logit is positive.
    xt, yt = as_tensor(x), as_tensor(y)
    logits = model(xt)
    loss = float(nn.functional.binary_cross_entropy_with_logits(logits, yt))
    pred = (logits > 0).to(DTYPE)
    acc = float((pred == yt).to(DTYPE).mean())
    # Averaging per-class recall (balanced accuracy) so class imbalance cannot flatter a model.
    recalls = [float((pred[yt == c] == c).to(DTYPE).mean()) for c in (0.0, 1.0) if bool((yt == c).any())]
    return {"acc": acc, "bal_acc": float(np.mean(recalls)), "loss": loss}


def fit(
    model: nn.Module,
    x_train,
    y_train,
    cfg: TrainConfig,
    seed: int,
    eval_sets: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    eval_every_epoch: bool = False,
) -> dict:
    """Mini-batch Adam training with BCE loss.

    Returns ``{"history": [per-epoch dicts], "final": {split: metrics}, "time_s": float}``.
    Each history row has ``loss``, ``grad_norm`` (mean over the epoch's mini-batches, measured on
    the circuit angles for quantum models) and optionally per-split metrics.
    """
    # Preparing tensors, the Adam optimiser and a private RNG for shuffling.
    t0 = time.time()
    xt, yt = as_tensor(x_train), as_tensor(y_train)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(int(seed))
    grad_params = model.gradient_parameters()
    history = []
    n = len(xt)
    for epoch in range(1, cfg.epochs + 1):
        # Shuffling the training set once per epoch with the seeded generator.
        perm = torch.randperm(n, generator=gen)
        losses, gnorms = [], []
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            # Taking one mini-batch step and logging the gradient norm before the update.
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(xt[idx]), yt[idx])
            loss.backward()
            gnorms.append(grad_norm(grad_params))
            opt.step()
            losses.append(float(loss.detach()))
        # Recording the epoch summary (and per-split scores when asked).
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "grad_norm": float(np.mean(gnorms))}
        if eval_sets and eval_every_epoch:
            for name, (xe, ye) in eval_sets.items():
                row[name] = evaluate_model(model, xe, ye)
        history.append(row)
    # Scoring the finished model on every requested split.
    final = {name: evaluate_model(model, xe, ye) for name, (xe, ye) in (eval_sets or {}).items()}
    return {"history": history, "final": final, "time_s": time.time() - t0}
