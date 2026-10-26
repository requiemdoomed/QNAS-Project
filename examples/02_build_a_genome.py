"""Example 2 - a circuit is a list of genes.  Run:  python examples/02_build_a_genome.py

Builds a tiny genome by hand, measures its cost (depth, two-qubit gates), runs it, then lets the
genetic operators (mutation, crossover) change it and checks that the results stay legal.
"""
# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : examples/02_build_a_genome.py
# Purpose: building a genome by hand, running it, and letting the genetic operators change it.
# -----------------------------------------------------------------------------

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # running without installing the package

import random

import torch

from qnas.circuit import make_qnode
from qnas.genome import circuit_depth, count_trainable, count_two_qubit, describe, rot, two_qubit
from qnas.space import GeneSpace

# Writing a six-gate circuit as a list of genes and measuring its cost.
genome = [rot("RY", 0), rot("RY", 1), two_qubit("CNOT", 0, 1), rot("RZ", 1), two_qubit("CNOT", 1, 2), rot("RX", 2)]
print("genome        :", describe(genome))
print("depth         :", circuit_depth(genome))
print("two-qubit gates:", count_two_qubit(genome))
print("parameters    :", count_trainable(genome))

# Compiling it into a PennyLane circuit and running it on one input (fixed seed for repeatable output).
torch.manual_seed(0)
qnode = make_qnode(genome, n_qubits=4, n_features=4, diff_method=None)
x = torch.tensor([0.3, 1.2, 2.0, 0.5], dtype=torch.float64)
w = torch.randn(count_trainable(genome), dtype=torch.float64)
print("<Z0> output   :", float(qnode(x, w)))

# Recombining with a random partner and mutating the child, then checking legality.
rng = random.Random(1)
space = GeneSpace(n_qubits=4)
other = space.random_genome(rng)
child1, child2 = space.crossover(genome, other, rng)
mutant = space.mutate(child1, rng)
print("\nrandom partner:", describe(other))
print("child (crossover):", describe(child1))
print("mutant        :", describe(mutant))
print("all legal     :", all(space.is_valid(g) for g in (child1, child2, mutant)))
