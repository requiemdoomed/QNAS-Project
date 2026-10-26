"""Evaluate the hand-crafted baseline circuits under the *same short protocol as the search*.

This puts SEL-L1..L3 and BEL-L1..L3 on the same (accuracy, depth, two-qubit-gate) axes as the evolved
circuits, so they can be drawn on the Pareto plot and used as warm-start seeds.
"""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/baselines.py
# Purpose: scoring the SEL / BEL templates with the same short protocol as the search.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict

from .data import TaskData
from .fitness import EvalSpec, evaluate_genome
from .templates import baseline_suite


def evaluate_baselines(task: TaskData, spec: EvalSpec, max_layers: int = 3) -> Dict[str, dict]:
    # Training each template exactly as a candidate circuit would be trained.
    out = {}
    for name, genes in baseline_suite(spec.n_qubits, max_layers).items():
        out[name] = {"metrics": evaluate_genome(genes, task, spec), "n_gates": len(genes)}
    return out
