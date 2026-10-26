"""Example 4 - a miniature evolutionary search (about 30 seconds).  Run:  python examples/04_tiny_search.py"""
# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : examples/04_tiny_search.py
# Purpose: running a miniature evolutionary search and printing its Pareto front.
# -----------------------------------------------------------------------------

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # running without installing the package

import torch

from qnas.data import prepare_task
from qnas.evolution import QNASSearch, SearchConfig
from qnas.fitness import EvalSpec
from qnas.genome import describe
from qnas.space import GeneSpace
from qnas.templates import strongly_entangling
from qnas.train import TrainConfig

# Preparing the data, a small search space and a short scoring protocol.
torch.set_num_threads(1)
task = prepare_task("digits", ((0, 2, 4, 6, 8), (1, 3, 5, 7, 9)), n_features=4, extractor="pca", seed=0)
space = GeneSpace(n_qubits=4, max_depth=14, max_genes=32, init_max_genes=12)
spec = EvalSpec(n_qubits=4, n_features=4, max_depth=14, max_genes=32, train=TrainConfig(epochs=3, batch_size=64))
search = QNASSearch(task, space, spec, SearchConfig(pop_size=12, generations=6, seed=0), out_dir=None,
                      # warm-starting: random circuits alone rarely beat tiny ones in so short a run
                      initial_genomes=[strongly_entangling(4, 1), strongly_entangling(4, 2)])
search.run()
# Listing the non-dominated circuits from cheapest to most entangling.
print("\nPareto front (accuracy / depth / two-qubit gates):")
for r in sorted(search.pareto_front(), key=lambda r: r["metrics"]["n_2q"]):
    m = r["metrics"]
    print(f"  val acc {m['val_acc']:.3f}  depth {m['depth']:2.0f}  2q {m['n_2q']:2.0f}  {describe(r['genes'])[:70]}")
