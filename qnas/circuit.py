"""Compile a genome into a PennyLane ``QNode``.

The circuit always starts with a fixed *angle-embedding* layer ``RY(x_i)`` that loads
the classical features into the qubits.  The genes are then applied in order and the
circuit returns an expectation value (default: ``<Z_0>``).

The QNode signature ``circuit(inputs, weights)`` is what ``qml.qnn.TorchLayer`` expects:

* ``inputs``  - tensor of shape ``(batch, n_features)`` (or ``(n_features,)``)
* ``weights`` - tensor of shape ``(n_trainable,)`` or, for batched gradient analysis,
  ``(batch, n_trainable)``.  Both work because we index with ``weights[..., k]``.
"""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/circuit.py
# Purpose: turning a genome (a list of genes) into a runnable PennyLane circuit.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import pennylane as qml

from .genome import DATA, FIXED, TRAIN, Gene, count_trainable

# Mapping gate names in the genome to PennyLane operations.
_ROTATION_OPS = {"RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ}
_TWO_QUBIT_OPS = {"CNOT": qml.CNOT, "CZ": qml.CZ, "SWAP": qml.SWAP}


def _observable(kind: str, n_qubits: int):
    # Choosing between a local cost (one qubit) and a global cost (all qubits);
    # the barren-plateau study compares the two.
    if kind == "z0":  # measuring Z on the first qubit (local cost)
        return qml.PauliZ(0)
    if kind == "global_z":  # measuring Z (x) Z (x) ... (x) Z (global cost)
        obs = qml.PauliZ(0)
        for w in range(1, n_qubits):
            obs = obs @ qml.PauliZ(w)
        return obs
    raise ValueError(f"unknown observable {kind!r}")


def apply_circuit(
    genes: Sequence[Gene],
    inputs,
    weights,
    n_qubits: int,
    n_features: int,
    noise: Optional[Tuple[float, float]] = None,
) -> None:
    """Queue every operation of the circuit (must be called inside a QNode).

    ``noise=(p1, p2)`` adds a depolarizing channel after every gate (``p1`` after
    single-qubit gates, ``p2`` after two-qubit gates) - only valid on ``default.mixed``.
    """

    def _noisy(wires, p):
        # Inserting a depolarizing channel on each touched wire when noise is on.
        if noise is not None and p > 0:
            for w in wires:
                qml.DepolarizingChannel(p, wires=w)

    # Encoding the features with one RY per qubit (the fixed, non-evolved layer).
    for i in range(n_features):
        qml.RY(inputs[..., i], wires=i % n_qubits)
        _noisy((i % n_qubits,), noise[0] if noise else 0.0)

    # Applying the evolved genes in order; k walks through the trainable weights.
    k = 0
    for g in genes:
        if g.is_two_qubit:
            _TWO_QUBIT_OPS[g.gate](wires=list(g.wires))
            _noisy(g.wires, noise[1] if noise else 0.0)
        elif g.is_rotation:
            # Picking the angle source: a learned weight, an input feature or a constant.
            if g.mode == TRAIN:
                angle = weights[..., k]
                k += 1
            elif g.mode == DATA:
                angle = inputs[..., g.arg % n_features]
            elif g.mode == FIXED:
                angle = g.fixed_angle
            else:
                raise ValueError(f"bad rotation mode {g.mode!r}")
            _ROTATION_OPS[g.gate](angle, wires=g.wires[0])
            _noisy(g.wires, noise[0] if noise else 0.0)
        elif g.gate == "H":
            qml.Hadamard(wires=g.wires[0])
            _noisy(g.wires, noise[0] if noise else 0.0)
        else:
            raise ValueError(f"unknown gate {g.gate!r}")


def make_qnode(
    genes: Sequence[Gene],
    n_qubits: int,
    n_features: int,
    device: str = "default.qubit",
    diff_method: Optional[str] = "backprop",
    observable: str = "z0",
    noise: Optional[Tuple[float, float]] = None,
):
    """Build the QNode ``circuit(inputs, weights) -> expval``."""
    # Freezing the gene list so later edits to the caller's list cannot leak in.
    genes = list(genes)
    dev = qml.device(device, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        apply_circuit(genes, inputs, weights, n_qubits, n_features, noise)
        return qml.expval(_observable(observable, n_qubits))

    return circuit


def n_weights(genes: Sequence[Gene]) -> int:
    # Counting the trainable angles the circuit expects in ``weights``.
    return count_trainable(genes)
