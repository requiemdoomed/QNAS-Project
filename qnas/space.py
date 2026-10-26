"""The search space: what genomes are legal and how to sample / vary them.

``GeneSpace`` owns every random decision about circuit structure:

* random gene / genome generation (biased toward *shallow* circuits)
* ``repair``  - makes any gene list legal (wires, connectivity, length, depth)
* ``mutate``  - gate-level point mutations + insert / delete
* ``crossover`` - two-point *sub-circuit* swap between two parents

All methods take an explicit ``random.Random`` so runs are reproducible.
"""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/space.py
# Purpose: defining which circuits are legal and how the genetic operators
#          (sampling, repair, mutation, crossover) change them.
# -----------------------------------------------------------------------------

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from typing import List, Sequence

from .genome import (
    DATA,
    FIXED,
    FIXED_ANGLES,
    NONE,
    ONE_QUBIT_GATES,
    ROTATIONS,
    SYMMETRIC_2Q,
    TRAIN,
    TWO_QUBIT_GATES,
    Gene,
    circuit_depth,
)


@lru_cache(maxsize=None)
def _wire_pairs(n: int, connectivity: str, symmetric: bool) -> tuple:
    """All ordered wire pairs allowed by a connectivity graph (cached)."""
    # Building the edge list once per (size, topology) instead of on every
    # validity check - this function is called thousands of times per run.
    if connectivity == "all":
        base = list(permutations(range(n), 2))
    elif connectivity == "linear":
        base = [(i, i + 1) for i in range(n - 1)] + [(i + 1, i) for i in range(n - 1)]
    else:  # ring
        base = [(i, (i + 1) % n) for i in range(n)] + [((i + 1) % n, i) for i in range(n)]
    if symmetric:  # keeping one ordering only, since (a, b) == (b, a)
        base = [p for p in base if p[0] < p[1]]
    return tuple(sorted(set(base)))


@dataclass(frozen=True)
class GeneSpace:
    n_qubits: int = 4
    n_features: int = 4
    max_depth: int = 14          # capping circuit depth of the evolved body (D_max)
    max_genes: int = 32          # capping the number of gates
    init_max_genes: int = 14     # keeping generation-0 circuits short (2..init_max_genes gates)
    connectivity: str = "all"    # choosing "all", "linear" or "ring" wiring
    gate_set: tuple = ("RX", "RY", "RZ", "H", "CNOT", "CZ", "SWAP")
    allow_data: bool = False     # allowing rotations whose angle is an input feature (re-uploading)
    allow_fixed: bool = True     # allowing constant-angle rotations
    p_two_qubit: float = 0.35    # drawing a 2-qubit gate with this probability
    p_hadamard: float = 0.08     # drawing H with this probability among single-qubit gates

    def __post_init__(self):
        # Validating the settings early so a bad config fails before any training.
        if self.n_qubits < 2:
            raise ValueError("need at least 2 qubits")
        if self.connectivity not in ("all", "linear", "ring"):
            raise ValueError("connectivity must be 'all', 'linear' or 'ring'")
        bad = [g for g in self.gate_set if g not in ONE_QUBIT_GATES + TWO_QUBIT_GATES]
        if bad:
            raise ValueError(f"unknown gates in gate_set: {bad}")

    # ------------------------------------------------------------------ helpers
    @property
    def rotations(self) -> tuple:
        # Filtering the gate set down to the rotation gates.
        return tuple(g for g in self.gate_set if g in ROTATIONS)

    @property
    def two_qubit_gates(self) -> tuple:
        # Filtering the gate set down to the entangling gates.
        return tuple(g for g in self.gate_set if g in TWO_QUBIT_GATES)

    @property
    def has_hadamard(self) -> bool:
        return "H" in self.gate_set

    def allowed_pairs(self, gate: str) -> List[tuple]:
        """All (a, b) wire pairs the given two-qubit gate may use."""
        return list(_wire_pairs(self.n_qubits, self.connectivity, gate in SYMMETRIC_2Q))

    def gene_is_valid(self, g: Gene) -> bool:
        # Rejecting gates outside the gate set or wires outside the register.
        if g.gate not in self.gate_set:
            return False
        if any((w < 0 or w >= self.n_qubits) for w in g.wires):
            return False
        # Checking two-qubit gates against the connectivity graph.
        if g.is_two_qubit:
            return len(g.wires) == 2 and tuple(g.wires) in _wire_pairs(
                self.n_qubits, self.connectivity, g.gate in SYMMETRIC_2Q)
        if len(g.wires) != 1:
            return False
        # Checking that a rotation's mode is switched on and its argument is in range.
        if g.is_rotation:
            if g.mode == TRAIN:
                return True
            if g.mode == DATA:
                return self.allow_data and 0 <= g.arg < self.n_features
            if g.mode == FIXED:
                return self.allow_fixed and 0 <= g.arg < len(FIXED_ANGLES)
            return False
        return g.mode == NONE

    def is_valid(self, genes: Sequence[Gene]) -> bool:
        # Requiring a non-empty genome within the gate and depth budgets.
        return (
            1 <= len(genes) <= self.max_genes
            and circuit_depth(genes) <= self.max_depth
            and all(self.gene_is_valid(g) for g in genes)
        )

    # ------------------------------------------------------------------ sampling
    def _random_rotation_params(self, rng: random.Random):
        """Pick (mode, arg) for a rotation: mostly trainable, some data / fixed."""
        # Weighting the draw 70 / 20 / 10 so most rotations stay trainable.
        choices, weights = [TRAIN], [0.70]
        if self.allow_data:
            choices.append(DATA)
            weights.append(0.20)
        if self.allow_fixed:
            choices.append(FIXED)
            weights.append(0.10)
        mode = rng.choices(choices, weights)[0]
        if mode == DATA:
            return DATA, rng.randrange(self.n_features)
        if mode == FIXED:
            return FIXED, rng.randrange(len(FIXED_ANGLES))
        return TRAIN, 0

    def random_gene(self, rng: random.Random) -> Gene:
        # Deciding first between a two-qubit and a single-qubit gate.
        two_ok = bool(self.two_qubit_gates)
        one_ok = bool(self.rotations) or self.has_hadamard
        if two_ok and (not one_ok or rng.random() < self.p_two_qubit):
            gate = rng.choice(self.two_qubit_gates)
            a, b = rng.choice(self.allowed_pairs(gate))
            return Gene(gate, (a, b), NONE, 0)
        # Picking a wire, then either a Hadamard or a rotation with its mode.
        wire = rng.randrange(self.n_qubits)
        if self.has_hadamard and (not self.rotations or rng.random() < self.p_hadamard):
            return Gene("H", (wire,), NONE, 0)
        gate = rng.choice(self.rotations)
        mode, arg = self._random_rotation_params(rng)
        return Gene(gate, (wire,), mode, arg)

    def random_genome(self, rng: random.Random, max_genes: int | None = None) -> List[Gene]:
        # Drawing a short-to-medium length so generation 0 is not full of deep circuits.
        hi = max(2, min(max_genes or self.init_max_genes, self.max_genes))
        length = rng.randint(2, hi)
        return self.repair([self.random_gene(rng) for _ in range(length)], rng)

    # ------------------------------------------------------------------ repair
    def repair(self, genes: Sequence[Gene], rng: random.Random) -> List[Gene]:
        """Return a legal genome that is as close as possible to ``genes``."""
        # Truncating to the gate budget and swapping illegal genes for random legal ones.
        out: List[Gene] = []
        for g in list(genes)[: self.max_genes]:
            out.append(g if self.gene_is_valid(g) else self.random_gene(rng))
        if not out:
            out = [self.random_gene(rng)]
        while len(out) > 1 and circuit_depth(out) > self.max_depth:
            out.pop()  # dropping trailing gates until the depth budget is respected
        return out

    # ------------------------------------------------------------------ variation
    def mutate_gene(self, g: Gene, rng: random.Random) -> Gene:
        # Choosing one of three point mutations with equal odds.
        op = rng.choice(("retype", "rewire", "reparam"))
        if op == "retype":
            return self.random_gene(rng)
        if op == "rewire":
            if g.is_two_qubit:
                a, b = rng.choice(self.allowed_pairs(g.gate))
                return Gene(g.gate, (a, b), NONE, 0)
            return Gene(g.gate, (rng.randrange(self.n_qubits),), g.mode, g.arg)
        # Re-drawing the angle mode of a rotation (other gates get retyped instead).
        if g.is_rotation:
            mode, arg = self._random_rotation_params(rng)
            return Gene(g.gate, g.wires, mode, arg)
        return self.random_gene(rng)

    def mutate(
        self,
        genes: Sequence[Gene],
        rng: random.Random,
        p_gene: float = 0.2,
        p_insert: float = 0.25,
        p_delete: float = 0.2,
    ) -> List[Gene]:
        """Gate-level mutation: each gene mutates with prob ``p_gene`` (P_m = 0.2);
        additionally one gate may be inserted and/or one deleted."""
        # Mutating each gene independently, then maybe growing or shrinking the circuit.
        out = list(genes)
        for i in range(len(out)):
            if rng.random() < p_gene:
                out[i] = self.mutate_gene(out[i], rng)
        if rng.random() < p_insert and len(out) < self.max_genes:
            out.insert(rng.randint(0, len(out)), self.random_gene(rng))
        if rng.random() < p_delete and len(out) > 1:
            out.pop(rng.randrange(len(out)))
        return self.repair(out, rng)

    def crossover(self, a: Sequence[Gene], b: Sequence[Gene], rng: random.Random):
        """Two-point crossover that swaps a *sub-circuit* (a slice of genes) between parents."""
        # Cutting each parent at two random points and exchanging the middle slices.
        a, b = list(a), list(b)
        i1, i2 = sorted((rng.randint(0, len(a)), rng.randint(0, len(a))))
        j1, j2 = sorted((rng.randint(0, len(b)), rng.randint(0, len(b))))
        c1 = a[:i1] + b[j1:j2] + a[i2:]
        c2 = b[:j1] + a[i1:i2] + b[j2:]
        return self.repair(c1, rng), self.repair(c2, rng)
