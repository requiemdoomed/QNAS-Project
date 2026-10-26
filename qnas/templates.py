"""Hand-crafted baseline ansatzes, expressed as genomes.

Writing the templates as gene lists means the baselines go through *exactly* the same
circuit builder, trainer and metrics (depth, two-qubit count) as evolved circuits - so the
comparison is apples-to-apples.  ``tests/test_templates_circuit.py`` checks numerically that these
genomes reproduce PennyLane's own ``StronglyEntanglingLayers`` / ``BasicEntanglerLayers``.
"""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/templates.py
# Purpose: writing PennyLane's hand-crafted ansatz templates as genomes so they can be
#          trained and measured exactly like evolved circuits.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import List

from .genome import TRAIN, Gene, rot, two_qubit


def strongly_entangling(n_qubits: int, n_layers: int) -> List[Gene]:
    """PennyLane ``StronglyEntanglingLayers``: per layer, a trainable ``Rot = RZ RY RZ`` on every
    qubit followed by a ring of CNOTs with range ``r = (layer mod (n-1)) + 1``."""
    genes: List[Gene] = []
    for layer in range(n_layers):
        # Placing a general rotation Rot = RZ RY RZ on every qubit.
        for q in range(n_qubits):
            genes += [rot("RZ", q), rot("RY", q), rot("RZ", q)]
        if n_qubits > 1:
            # Closing the layer with a CNOT ring whose range grows with the layer index.
            r = (layer % (n_qubits - 1)) + 1
            for q in range(n_qubits):
                genes.append(two_qubit("CNOT", q, (q + r) % n_qubits))
    return genes


def basic_entangler(n_qubits: int, n_layers: int, rotation: str = "RX") -> List[Gene]:
    """PennyLane ``BasicEntanglerLayers``: per layer, one trainable rotation per qubit followed by
    a ring of CNOTs (a single CNOT when there are only 2 qubits)."""
    genes: List[Gene] = []
    for _ in range(n_layers):
        # Placing one trainable rotation per qubit, then a CNOT ring.
        for q in range(n_qubits):
            genes.append(rot(rotation, q, TRAIN))
        if n_qubits == 2:
            genes.append(two_qubit("CNOT", 0, 1))
        elif n_qubits > 2:
            for q in range(n_qubits):
                genes.append(two_qubit("CNOT", q, (q + 1) % n_qubits))
    return genes


def baseline_suite(n_qubits: int, max_layers: int = 3) -> dict:
    """Named baseline circuits: SEL-L1..Lk and BEL-L1..Lk."""
    # Naming the circuits by family and layer count, e.g. "SEL-L2".
    suite = {}
    for layers in range(1, max_layers + 1):
        suite[f"SEL-L{layers}"] = strongly_entangling(n_qubits, layers)
    for layers in range(1, max_layers + 1):
        suite[f"BEL-L{layers}"] = basic_entangler(n_qubits, layers)
    return suite
