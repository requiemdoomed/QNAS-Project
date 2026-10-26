"""Genotype of a quantum circuit: a variable-length list of genes.

A *gene* is one gate application::

    Gene(gate, wires, mode, arg)

* ``gate``  - one of RX, RY, RZ, H, CNOT, CZ, SWAP
* ``wires`` - the qubit(s) the gate acts on (1 for single-qubit, 2 for two-qubit)
* ``mode``  - the *parametric status* of a rotation gate:
    - ``"train"`` : the angle is a trainable weight (learned by gradient descent)
    - ``"data"``  : the angle is one of the input features (data re-uploading)
    - ``"fixed"`` : the angle is a constant picked from ``FIXED_ANGLES``
  Non-rotation gates always use ``mode="none"``.
* ``arg``   - feature index (mode ``"data"``) or index into ``FIXED_ANGLES``
  (mode ``"fixed"``); ``0`` otherwise.

A *genome* is simply ``list[Gene]``.  Everything in this file is plain Python
(no PennyLane / PyTorch), so it is cheap to import, hash and pickle.
"""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/genome.py
# Purpose: describing a circuit as a plain list of genes, plus the small
#          helpers (depth, gate counts, hashing, JSON) the rest of the code uses.
# -----------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

# Listing the gate alphabet the search is allowed to use.
ROTATIONS = ("RX", "RY", "RZ")
SINGLE_FIXED = ("H",)
ONE_QUBIT_GATES = ROTATIONS + SINGLE_FIXED
TWO_QUBIT_GATES = ("CNOT", "CZ", "SWAP")
ALL_GATES = ONE_QUBIT_GATES + TWO_QUBIT_GATES

# Marking CZ and SWAP as symmetric in their two wires (CNOT is not), which the
# simplifier relies on when it cancels neighbouring pairs.
SYMMETRIC_2Q = ("CZ", "SWAP")

# Naming the four parametric modes and the menu of constant angles.
TRAIN, DATA, FIXED, NONE = "train", "data", "fixed", "none"
FIXED_ANGLES = (-math.pi / 2, -math.pi / 4, math.pi / 4, math.pi / 2, math.pi)
_FIXED_LABELS = ("-π/2", "-π/4", "π/4", "π/2", "π")


@dataclass(frozen=True)
class Gene:
    """One gate of the circuit (immutable and hashable)."""

    gate: str
    wires: tuple
    mode: str = NONE
    arg: int = 0

    # ---- convenient properties -------------------------------------------------
    @property
    def is_rotation(self) -> bool:
        # Checking whether the gate carries an angle at all.
        return self.gate in ROTATIONS

    @property
    def is_two_qubit(self) -> bool:
        # Checking whether the gate entangles two wires (the costly kind on hardware).
        return self.gate in TWO_QUBIT_GATES

    @property
    def fixed_angle(self) -> float:
        # Looking up the constant angle this gene points at.
        return FIXED_ANGLES[self.arg]

    # ---- (de)serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        # Packing the gene into short JSON keys so result files stay small.
        return {"g": self.gate, "w": list(self.wires), "m": self.mode, "a": int(self.arg)}

    @staticmethod
    def from_dict(d: dict) -> "Gene":
        # Rebuilding a gene from JSON, falling back to defaults for older files.
        return Gene(d["g"], tuple(int(w) for w in d["w"]), d.get("m", NONE), int(d.get("a", 0)))

    def __str__(self) -> str:  # printing a readable form, e.g. "RY(x2)[1]"
        # Writing the wires as "0,1" and the angle as θ, x<i> or a fixed label.
        w = ",".join(str(i) for i in self.wires)
        if self.is_rotation:
            if self.mode == TRAIN:
                p = "θ"
            elif self.mode == DATA:
                p = f"x{self.arg}"
            else:
                p = _FIXED_LABELS[self.arg]
            return f"{self.gate}({p})[{w}]"
        return f"{self.gate}[{w}]"


Genome = Sequence[Gene]


# ------------------------------------------------------------------------------
# Constructing genes (used by templates, tests and examples)
# ------------------------------------------------------------------------------
def rot(gate: str, wire: int, mode: str = TRAIN, arg: int = 0) -> Gene:
    # Refusing anything that is not RX / RY / RZ, so typos fail loudly.
    if gate not in ROTATIONS:
        raise ValueError(f"{gate!r} is not a rotation gate")
    return Gene(gate, (int(wire),), mode, int(arg))


def hadamard(wire: int) -> Gene:
    # Building a Hadamard gene (no angle, so the mode is always "none").
    return Gene("H", (int(wire),), NONE, 0)


def two_qubit(gate: str, a: int, b: int) -> Gene:
    # Refusing single-qubit gates here for the same reason as in rot().
    if gate not in TWO_QUBIT_GATES:
        raise ValueError(f"{gate!r} is not a two-qubit gate")
    return Gene(gate, (int(a), int(b)), NONE, 0)


# ------------------------------------------------------------------------------
# Measuring circuit structure
# ------------------------------------------------------------------------------
def circuit_depth(genes: Iterable[Gene]) -> int:
    """ASAP circuit depth of the gate list (length of the longest dependency chain)."""
    # Tracking, per wire, the layer of the last gate that touched it; a new gate
    # lands one layer after the busiest of its wires.
    level: dict = {}
    for g in genes:
        d = max(level.get(w, 0) for w in g.wires) + 1
        for w in g.wires:
            level[w] = d
    return max(level.values(), default=0)


def count_two_qubit(genes: Iterable[Gene]) -> int:
    # Counting entangling gates - the main hardware-cost objective.
    return sum(1 for g in genes if g.is_two_qubit)


def count_trainable(genes: Iterable[Gene]) -> int:
    # Counting rotations whose angle is learned.
    return sum(1 for g in genes if g.is_rotation and g.mode == TRAIN)


def count_data_gates(genes: Iterable[Gene]) -> int:
    # Counting rotations that re-upload an input feature.
    return sum(1 for g in genes if g.is_rotation and g.mode == DATA)


def genome_key(genes: Iterable[Gene]) -> tuple:
    """Hashable identity of a genome (used for caching evaluations)."""
    return tuple(genes)


def genome_hash(genes: Iterable[Gene]) -> str:
    """Short stable hash of a genome (stable across processes / runs)."""
    # Using SHA-1 of a text form instead of Python's hash(), which changes
    # between interpreter runs and would break seeding and caching.
    text = "|".join(f"{g.gate}:{','.join(map(str, g.wires))}:{g.mode}:{g.arg}" for g in genes)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def describe(genes: Iterable[Gene]) -> str:
    # Joining the genes into one readable line, e.g. "RY(θ)[0] → CNOT[0,1]".
    genes = list(genes)
    return " → ".join(str(g) for g in genes) if genes else "(empty circuit)"


def genome_to_json(genes: Iterable[Gene]) -> list:
    # Converting a genome into a JSON-friendly list of dicts.
    return [g.to_dict() for g in genes]


def genome_from_json(items: Iterable[dict]) -> list:
    # Reading a genome back from its JSON list.
    return [Gene.from_dict(d) for d in items]


def structure_summary(genes: Iterable[Gene]) -> dict:
    # Collecting every structural number about a circuit in one dict.
    genes = list(genes)
    return {
        "n_gates": len(genes),
        "depth": circuit_depth(genes),
        "n_2q": count_two_qubit(genes),
        "n_trainable": count_trainable(genes),
        "n_data": count_data_gates(genes),
    }
