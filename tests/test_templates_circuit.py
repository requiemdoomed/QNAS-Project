# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : tests/test_templates_circuit.py
# Purpose: verifying the templates against PennyLane's own implementations and the
#          hybrid model, batching and noise behaviour.
# -----------------------------------------------------------------------------

import numpy as np
import pennylane as qml
import pytest
import torch

from qnas.circuit import make_qnode
from qnas.genome import count_trainable
from qnas.model import HybridQuantumModel, mlp_hidden_for_params
from qnas.templates import basic_entangler, strongly_entangling


def _reference(template, n, L, x, w):
    # Building the same circuit with the official PennyLane template.
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def c():
        for i in range(n):
            qml.RY(x[i], wires=i)
        template(w, wires=range(n))
        return qml.expval(qml.PauliZ(0))

    return float(c())


@pytest.mark.parametrize("L", [1, 2, 3])
def test_sel_matches_pennylane(L):
    n = 4
    # Matching our SEL genome to qml.StronglyEntanglingLayers to 1e-8.
    genes = strongly_entangling(n, L)
    rs = np.random.default_rng(L)
    x = rs.uniform(0, np.pi, n)
    w = rs.uniform(-2, 2, (L, n, 3))
    ours = make_qnode(genes, n, n, diff_method=None)(torch.tensor(x), torch.tensor(w.reshape(-1)))
    ref = _reference(qml.StronglyEntanglingLayers, n, L, x, w)
    assert abs(float(ours) - ref) < 1e-8


@pytest.mark.parametrize("L", [1, 2, 3])
def test_bel_matches_pennylane(L):
    n = 4
    # Matching our BEL genome to qml.BasicEntanglerLayers to 1e-8.
    genes = basic_entangler(n, L)
    rs = np.random.default_rng(10 + L)
    x = rs.uniform(0, np.pi, n)
    w = rs.uniform(-2, 2, (L, n))
    ours = make_qnode(genes, n, n, diff_method=None)(torch.tensor(x), torch.tensor(w.reshape(-1)))
    ref = _reference(qml.BasicEntanglerLayers, n, L, x, w)
    assert abs(float(ours) - ref) < 1e-8


def test_batched_equals_single():
    # Checking that a batch gives the same outputs as one sample at a time.
    genes = strongly_entangling(4, 1)
    q = make_qnode(genes, 4, 4, diff_method=None)
    rs = np.random.default_rng(0)
    xb = torch.tensor(rs.uniform(0, np.pi, (5, 4)))
    w = torch.tensor(rs.normal(size=count_trainable(genes)))
    batched = q(xb, w)
    single = torch.stack([q(xb[i], w) for i in range(5)])
    assert torch.allclose(batched, single, atol=1e-10)


def test_noise_at_zero_matches_ideal():
    # Checking zero noise matches the ideal result and real noise shrinks the expectation.
    genes = strongly_entangling(4, 1)
    rs = np.random.default_rng(1)
    x = torch.tensor(rs.uniform(0, np.pi, 4))
    w = torch.tensor(rs.normal(size=count_trainable(genes)))
    ideal = make_qnode(genes, 4, 4, diff_method=None)(x, w)
    noisy0 = make_qnode(genes, 4, 4, device="default.mixed", diff_method=None, noise=(0.0, 0.0))(x, w)
    assert abs(float(ideal) - float(noisy0)) < 1e-8
    noisy = make_qnode(genes, 4, 4, device="default.mixed", diff_method=None, noise=(0.05, 0.3))(x, w)
    assert abs(float(noisy)) <= abs(float(ideal)) + 1e-9  # depolarizing noise shrinks the expectation toward 0


def test_model_forward_and_param_count():
    # Running the model forward and backward and checking the parameter count.
    genes = strongly_entangling(4, 1)
    m = HybridQuantumModel(genes, 4, 4)
    out = m(torch.rand(7, 4, dtype=torch.float64))
    assert out.shape == (7,)
    assert m.n_parameters == 12 + 2
    out.sum().backward()
    assert m.qlayer.weights.grad is not None


def test_model_without_trainable_gates():
    # Handling a circuit with no trainable angle (only scale and bias learn).
    from qnas.genome import two_qubit
    m = HybridQuantumModel([two_qubit("CNOT", 0, 1)], 4, 4)
    assert m.n_parameters == 2
    assert m(torch.rand(3, 4, dtype=torch.float64)).shape == (3,)


def test_mlp_hidden_never_degenerate():
    # Making sure no MLP is built with a single hidden unit.
    assert mlp_hidden_for_params(4, 14) >= 2
    assert mlp_hidden_for_params(4, 3) >= 2
