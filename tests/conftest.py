# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : tests/conftest.py
# Purpose: sharing fixtures (a seeded RNG and a tiny toy task) across the test suite.
# -----------------------------------------------------------------------------

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Making the package importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qnas.data import TaskData  # noqa: E402


@pytest.fixture(autouse=True)
def _single_thread():
    # Keeping torch on one thread so tests are fast and deterministic.
    torch.set_num_threads(1)


@pytest.fixture
def rng():
    # Giving every test the same seeded random generator.
    return random.Random(0)


@pytest.fixture(scope="session")
def toy_task() -> TaskData:
    """Small separable 4-feature task in [0, pi] (no dataset needed)."""
    # Labelling points by a simple linear rule so any decent model can learn it.
    g = np.random.default_rng(0)

    def make(n):
        x = g.uniform(0, np.pi, size=(n, 4))
        y = (x[:, 0] + 0.5 * x[:, 1] > 1.5 * np.pi / 2 + 0.2).astype(np.int64)
        return x, y

    xs, ys = zip(make(120), make(60), make(60))
    return TaskData(xs[0], ys[0], xs[1], ys[1], xs[2], ys[2], n_features=4, info={"toy": True})
