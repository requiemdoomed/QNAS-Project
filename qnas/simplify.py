"""Post-processing of evolved circuits: remove bloat without (or with controlled) loss.

``simplify``            *lossless* rewrite rules - the circuit computes exactly the same function:
                        * two adjacent trainable rotations about the same axis on a qubit  -> one rotation
                          (RZ(a) RZ(b) = RZ(a+b), so the merged gate can represent the same family),
                        * two adjacent identical self-inverse gates (H H, CNOT CNOT, CZ CZ, SWAP SWAP) cancel.

``prune_by_ablation``   *evaluation-based* pruning: try deleting each gate (two-qubit gates first) and keep
                        the deletion if the multi-seed mean validation accuracy does not fall by more than
                        ``tolerance`` relative to the original circuit.  Seeds are fixed, so the decisions are
                        reproducible; averaging over seeds keeps one lucky/unlucky run from steering them.
"""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/simplify.py
# Purpose: cleaning up evolved circuits: lossless rewrites first, then a careful,
#          multi-seed ablation that removes gates the accuracy does not need.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .data import TaskData
from .fitness import EvalSpec
from .genome import ROTATIONS, SYMMETRIC_2Q, TRAIN, Gene, count_two_qubit, circuit_depth


def _next_touching(genes: List[Gene], i: int) -> Optional[int]:
    """Index of the next gene after ``i`` that touches any wire of ``genes[i]``."""
    # Scanning forward for the first later gate that shares a wire with gene i.
    wires = set(genes[i].wires)
    for j in range(i + 1, len(genes)):
        if wires & set(genes[j].wires):
            return j
    return None


def _same_op(a: Gene, b: Gene) -> bool:
    # Treating CZ / SWAP on (a, b) and (b, a) as the same gate; CNOT keeps its direction.
    if a.gate != b.gate:
        return False
    if a.gate in SYMMETRIC_2Q:
        return set(a.wires) == set(b.wires)
    return a.wires == b.wires


def simplify(genes: Sequence[Gene]) -> List[Gene]:
    """Apply the lossless rewrite rules until nothing changes. Never returns an empty circuit."""
    # Repeating passes until a full pass makes no change (each rule only shrinks the list).
    g = list(genes)
    changed = True
    while changed:
        changed = False
        for i in range(len(g)):
            j = _next_touching(g, i)
            if j is None:
                continue
            a, b = g[i], g[j]
            if a.is_two_qubit and _same_op(a, b):
                # Cancelling a repeated two-qubit gate only when nothing touches either
                # of its wires in between.
                other = [k for k in range(i + 1, j) if set(g[k].wires) & set(a.wires)]
                if not other:
                    del g[j], g[i]
                    changed = True
                    break
            elif (not a.is_two_qubit and not b.is_two_qubit and a.wires == b.wires):
                # Cancelling H H on the same wire.
                if a.gate == "H" and b.gate == "H":
                    del g[j], g[i]
                    changed = True
                    break
                if a.gate in ROTATIONS and a.gate == b.gate and a.mode == TRAIN and b.mode == TRAIN:
                    del g[j]  # merging two trainable rotations on the same axis into one
                    changed = True
                    break
    # Keeping the first gate if everything cancelled, so the circuit is never empty.
    return g if g else list(genes)[:1]


def prune_by_ablation(
    genes: Sequence[Gene],
    task: TaskData,
    spec: EvalSpec,
    n_seeds: int = 3,
    tolerance: float = 0.005,
    epochs: Optional[int] = None,
    log: Callable[[str], None] = lambda s: None,
) -> Tuple[List[Gene], dict]:
    """Greedy gate-ablation. Returns ``(pruned_genes, info)``."""
    from dataclasses import replace

    from .fitness import evaluate_genome

    # Shortening training (optionally) for the many ablation trials.
    train = spec.train if epochs is None else replace(spec.train, epochs=epochs)

    def score(gs: Sequence[Gene]) -> float:
        # Averaging validation accuracy over fixed seeds so one lucky run cannot decide.
        accs = []
        for s in range(n_seeds):
            sp = replace(spec, train=train, base_seed=10_000 + s)
            accs.append(evaluate_genome(list(gs), task, sp)["val_acc"])
        return float(np.mean(accs))

    # Scoring the simplified circuit once as the reference to beat.
    current = simplify(genes)
    ref = score(current)
    n_before = (len(current), count_two_qubit(current), circuit_depth(current))
    changed = True
    while changed and len(current) > 1:
        changed = False
        # Trying two-qubit gates first (the expensive ones), then the rest, back to front.
        order = sorted(range(len(current)), key=lambda i: (not current[i].is_two_qubit, -i))
        for i in order:
            trial = simplify(current[:i] + current[i + 1:])
            if len(trial) >= len(current):
                continue
            # Accepting the deletion when accuracy stays within the tolerance of the reference.
            acc = score(trial)
            if acc >= ref - tolerance:
                log(f"  pruned {current[i]}: acc {acc:.3f} (ref {ref:.3f}), {len(trial)} gates left")
                current = trial
                changed = True
                break
    # Reporting accuracy and size before and after pruning.
    info = {
        "ref_val_acc": ref,
        "final_val_acc": score(current),
        "before": {"gates": n_before[0], "n_2q": n_before[1], "depth": n_before[2]},
        "after": {"gates": len(current), "n_2q": count_two_qubit(current), "depth": circuit_depth(current)},
    }
    return current, info
