"""Evaluating a circuit: train the hybrid model briefly and compute the objectives.

Objectives (all computed from one short training run on the *training* split, scored on the
*validation* split - the test split is never touched during the search):

* ``acc``    validation accuracy                  (maximise)
* ``depth``  circuit depth of the evolved body    (minimise)
* ``n2q``    number of two-qubit gates            (minimise)
* ``params`` number of trainable parameters       (minimise, optional)

The scalar fitness used for the weighted mode is::

    F(C) = alpha * Accuracy(C) - beta * Depth(C) / D_max - gamma * N_2Q(C) / N_max
"""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/fitness.py
# Purpose: scoring one circuit: a short, seeded training run and the
#          objectives the search optimises.
# -----------------------------------------------------------------------------

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from .data import TaskData
from .genome import Gene, circuit_depth, count_two_qubit, genome_hash
from .model import HybridQuantumModel
from .train import TrainConfig, fit

# Giving each objective its direction for DEAP: +1 maximise, -1 minimise.
OBJECTIVE_WEIGHTS = {"acc": 1.0, "depth": -1.0, "n2q": -1.0, "params": -1.0}
# Linking each objective name to the metric key that holds its value.
_METRIC_OF = {"acc": "val_acc", "depth": "depth", "n2q": "n_2q", "params": "n_params"}


@dataclass(frozen=True)
class EvalSpec:
    """Everything (besides the genome and the data) that defines how a circuit is scored."""

    n_qubits: int = 4
    n_features: int = 4
    max_depth: int = 14
    max_genes: int = 32
    alpha: float = 1.0
    beta: float = 0.1
    gamma: float = 0.2
    train: TrainConfig = field(default_factory=TrainConfig)
    base_seed: int = 0
    observable: str = "z0"

    @property
    def n_max(self) -> int:  # bounding two-qubit gates by the gate budget
        return self.max_genes


def scalar_fitness(acc: float, depth: int, n2q: int, spec: EvalSpec) -> float:
    # Rewarding accuracy and charging for depth and two-qubit gates, each normalised.
    return spec.alpha * acc - spec.beta * depth / spec.max_depth - spec.gamma * n2q / spec.n_max


def eval_seed(base_seed: int, genes: Sequence[Gene]) -> int:
    """Deterministic seed per (run, genome) so a genome always gets the same score,
    no matter which worker process evaluates it or when."""
    # Hashing (run seed, genome) so the seed never depends on process or order.
    h = int(hashlib.sha1(f"{base_seed}|{genome_hash(genes)}".encode()).hexdigest()[:8], 16)
    return h % (2**31 - 1)


def objective_vector(metrics: Dict[str, float], names: Sequence[str]) -> Tuple[float, ...]:
    # Pulling the chosen objectives out of the metric dict, in order.
    return tuple(float(metrics[_METRIC_OF[n]]) for n in names)


def objective_weights(names: Sequence[str]) -> Tuple[float, ...]:
    # Listing the matching DEAP weights for those objectives.
    return tuple(OBJECTIVE_WEIGHTS[n] for n in names)


def evaluate_genome(genes: Sequence[Gene], task: TaskData, spec: EvalSpec) -> Dict[str, float]:
    """Train ``genes`` on ``task`` for ``spec.train.epochs`` epochs and return all metrics."""
    # Seeding torch from the genome so the same circuit always gets the same score.
    genes = list(genes)
    seed = eval_seed(spec.base_seed, genes)
    torch.manual_seed(seed)
    model = HybridQuantumModel(genes, spec.n_qubits, spec.n_features, init_std=spec.train.init_std,
                               observable=spec.observable)
    # Training on the training split and scoring on train + validation only
    # (the test split is never touched during the search).
    result = fit(model, task.x_train, task.y_train, spec.train, seed,
                 eval_sets={"train": (task.x_train, task.y_train), "val": (task.x_val, task.y_val)})
    # Gathering accuracy, structure and gradient statistics into one record.
    depth, n2q = circuit_depth(genes), count_two_qubit(genes)
    val, tr = result["final"]["val"], result["final"]["train"]
    grad_norms = [h["grad_norm"] for h in result["history"]]
    return {
        "val_acc": val["acc"],
        "val_bal_acc": val["bal_acc"],
        "val_loss": val["loss"],
        "train_acc": tr["acc"],
        "depth": depth,
        "n_2q": n2q,
        "n_gates": len(genes),
        "n_params": model.n_parameters,
        "grad_norm_first_epoch": grad_norms[0],
        "grad_norm_mean": float(np.mean(grad_norms)),
        "fitness": scalar_fitness(val["acc"], depth, n2q, spec),
        "time_s": result["time_s"],
        "seed": seed,
    }
