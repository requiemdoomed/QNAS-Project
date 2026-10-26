# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : tests/test_search_tools.py
# Purpose: checking dominance, the exact hypervolume and the lossless simplifier.
# -----------------------------------------------------------------------------

import numpy as np

from qnas.genome import count_trainable, rot, two_qubit
from qnas.pareto import dominates, hypervolume, non_dominated
from qnas.simplify import simplify
from qnas.templates import strongly_entangling
from qnas.circuit import make_qnode
import torch


def test_dominates_and_front():
    # Checking dominance on hand-made points and the resulting front.
    assert dominates((1, 1), (2, 2)) and not dominates((1, 2), (2, 1)) and not dominates((1, 1), (1, 1))
    assert non_dominated([(1, 3), (2, 2), (3, 1), (3, 3)]) == [0, 1, 2]


def test_hypervolume_known_values():
    # Comparing against hypervolumes that are easy to work out by hand.
    assert abs(hypervolume([(0.5, 0.5)], (1, 1)) - 0.25) < 1e-12
    assert abs(hypervolume([(0.0, 0.5), (0.5, 0.0)], (1, 1)) - 0.75) < 1e-12
    assert abs(hypervolume([(0.5, 0.5, 0.5)], (1, 1, 1)) - 0.125) < 1e-12
    assert hypervolume([(2, 2)], (1, 1)) == 0.0
    # Adding a dominated point must never change the hypervolume.
    a = hypervolume([(0.2, 0.7), (0.6, 0.3)], (1, 1))
    assert abs(a - hypervolume([(0.2, 0.7), (0.6, 0.3), (0.7, 0.8)], (1, 1))) < 1e-12


def _out(genes, x, w):
    # Evaluating a circuit once without gradients.
    return make_qnode(genes, 4, 4, diff_method=None)(x, w)


def test_simplify_is_lossless_and_shorter():
    # Building a circuit with a mergeable rotation pair and a cancelling CNOT pair.
    genes = [rot("RX", 0), rot("RX", 0), two_qubit("CNOT", 0, 1), two_qubit("CNOT", 0, 1), rot("RY", 1), rot("RZ", 2)]
    s = simplify(genes)
    assert len(s) < len(genes)
    rs = np.random.default_rng(0)
    x = torch.tensor(rs.uniform(0, np.pi, 4))
    # Giving the merged rotation the sum of the two original angles and comparing outputs.
    w_orig = torch.tensor(rs.normal(size=count_trainable(genes)))
    merged = torch.stack([w_orig[0] + w_orig[1], w_orig[2], w_orig[3]])
    assert abs(float(_out(genes, x, w_orig)) - float(_out(s, x, merged))) < 1e-8


def test_simplify_leaves_templates_alone():
    # Leaving a template untouched: nothing in it is redundant.
    g = strongly_entangling(4, 2)
    assert len(simplify(g)) == len(g)


def test_report_number_formatting():
    # Checking the small formatting helpers used in results.md.
    from qnas.report import _pfmt, _pts, _verdict
    assert _pfmt(float("nan")) == "n/a" and _pfmt(0.001) == "<0.01" and _pfmt(0.634) == "0.63"
    assert _pts(0.031) == "+3.1 pts"
    assert _verdict(0.05, 0.01) == "significantly higher" and _verdict(-0.003, 0.6) == "no significant difference"
