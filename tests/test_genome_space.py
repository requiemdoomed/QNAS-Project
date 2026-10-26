# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : tests/test_genome_space.py
# Purpose: checking gate counting, depth, hashing and that the genetic operators
#          always produce legal circuits.
# -----------------------------------------------------------------------------

import pytest

from qnas.genome import (circuit_depth, count_trainable, count_two_qubit, genome_from_json, genome_hash,
                         genome_to_json, rot, two_qubit)
from qnas.space import GeneSpace
from qnas.templates import basic_entangler, strongly_entangling


def test_depth_and_counts_of_templates():
    # Checking the known sizes of the SEL and BEL templates on 4 qubits.
    sel = {L: strongly_entangling(4, L) for L in (1, 2, 3)}
    assert [count_two_qubit(sel[L]) for L in (1, 2, 3)] == [4, 8, 12]
    assert [circuit_depth(sel[L]) for L in (1, 2, 3)] == [7, 12, 19]
    assert count_trainable(sel[2]) == 24
    bel = basic_entangler(4, 2)
    assert count_trainable(bel) == 8 and count_two_qubit(bel) == 8


def test_depth_asap_parallel_gates():
    # Placing three parallel rotations in one layer and the CNOT in the next.
    genes = [rot("RX", 0), rot("RX", 1), rot("RX", 2), two_qubit("CNOT", 0, 1)]
    assert circuit_depth(genes) == 2
    assert circuit_depth([]) == 0


def test_json_roundtrip_and_hash():
    # Round-tripping through JSON and checking the hash is stable but discriminating.
    g = strongly_entangling(4, 2)
    assert genome_from_json(genome_to_json(g)) == g
    assert genome_hash(g) == genome_hash(list(g))
    assert genome_hash(g) != genome_hash(g[:-1])


def test_random_genomes_are_valid(rng):
    # Sampling many random genomes and requiring every one to be legal.
    space = GeneSpace()
    for _ in range(100):
        assert space.is_valid(space.random_genome(rng))


@pytest.mark.parametrize("conn", ["all", "linear", "ring"])
def test_connectivity_respected(conn, rng):
    space = GeneSpace(connectivity=conn)
    n = space.n_qubits

    # Defining which wire pairs count as neighbours on each topology.
    def adjacent(a, b):
        d = abs(a - b)
        return d == 1 if conn == "linear" else (d == 1 or d == n - 1 if conn == "ring" else True)

    seen_two_qubit = 0
    for _ in range(80):
        for g in space.random_genome(rng):
            if g.is_two_qubit:
                seen_two_qubit += 1
                assert adjacent(*g.wires), (conn, g)
    assert seen_two_qubit > 0
    # Rejecting a gate between non-neighbours on a restricted topology.
    if conn == "linear":
        assert not space.gene_is_valid(two_qubit("CNOT", 0, 2))


def test_mutation_and_crossover_keep_validity(rng):
    # Breeding for many rounds and checking every child and mutant stays legal.
    space = GeneSpace()
    pop = [space.random_genome(rng) for _ in range(20)]
    for _ in range(300):
        a, b = rng.sample(pop, 2)
        c1, c2 = space.crossover(a, b, rng)
        assert space.is_valid(c1) and space.is_valid(c2)
        m = space.mutate(c1, rng)
        assert space.is_valid(m)
        pop[rng.randrange(len(pop))] = m


def test_repair_enforces_limits(rng):
    # Feeding repair an over-long circuit and checking both limits are enforced.
    space = GeneSpace(max_depth=6, max_genes=10)
    too_long = [rot("RX", i % 4) for i in range(40)]
    fixed = space.repair(too_long, rng)
    assert len(fixed) <= 10 and circuit_depth(fixed) <= 6 and space.is_valid(fixed)


def test_data_genes_only_when_allowed(rng):
    # Making sure data re-uploading genes only appear when switched on.
    space = GeneSpace(allow_data=False)
    for _ in range(100):
        assert all(g.mode != "data" for g in space.random_genome(rng))
    space = GeneSpace(allow_data=True)
    assert any(g.mode == "data" for _ in range(200) for g in space.random_genome(rng))


def test_connectivity_pair_counts():
    # Counting allowed CNOT pairs: all-to-all 4x3, a line 2x3, a ring 2x4; CZ keeps one direction only.
    assert len(GeneSpace(connectivity="all").allowed_pairs("CNOT")) == 12
    assert len(GeneSpace(connectivity="linear").allowed_pairs("CNOT")) == 6
    assert len(GeneSpace(connectivity="ring").allowed_pairs("CNOT")) == 8
    assert len(GeneSpace(connectivity="ring").allowed_pairs("CZ")) == 4
