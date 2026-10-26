# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : tests/test_pipeline_pieces.py
# Purpose: exercising the data loaders, evaluation, gradients, noise, search with
#          resume, and the command line on small inputs.
# -----------------------------------------------------------------------------

import json

import numpy as np
import pytest

from qnas.data import DatasetUnavailable, load_medmnist_raw, prepare_task
from qnas.evolution import QNASSearch, SearchConfig
from qnas.fitness import EvalSpec, evaluate_genome
from qnas.gradients import grad_stats, sample_gradients
from qnas.noise import circuit_accuracy
from qnas.space import GeneSpace
from qnas.templates import strongly_entangling
from qnas.train import TrainConfig


# Using a very short training protocol so the whole suite runs in seconds.
SPEC = EvalSpec(n_qubits=4, n_features=4, train=TrainConfig(epochs=2, batch_size=32), base_seed=3)


def test_evaluation_is_deterministic(toy_task):
    # Scoring the same circuit twice and expecting identical numbers.
    g = strongly_entangling(4, 1)
    a, b = evaluate_genome(g, toy_task, SPEC), evaluate_genome(g, toy_task, SPEC)
    assert a["val_acc"] == b["val_acc"] and a["fitness"] == b["fitness"]
    assert a["n_2q"] == 4 and a["depth"] == 7


def test_prepare_task_digits_shapes_and_range():
    # Building the even/odd digits task with PCA and checking shapes and angle range.
    t = prepare_task("digits", ((0, 2, 4, 6, 8), (1, 3, 5, 7, 9)), 4, "pca", seed=0)
    assert t.x_train.shape[1] == 4 and set(np.unique(t.y_train)) == {0, 1}
    assert t.x_train.min() >= 0 and t.x_train.max() <= np.pi + 1e-9
    assert len(t.y_test) > 100


def test_fake_medmnist_npz_and_missing_file(tmp_path):
    # Writing a fake MedMNIST file to test the loader and balancing without a download.
    g = np.random.default_rng(0)
    arrs = {}
    for split, n in (("train", 80), ("val", 20), ("test", 30)):
        arrs[f"{split}_images"] = g.integers(0, 255, (n, 28, 28), dtype=np.uint8)
        arrs[f"{split}_labels"] = g.integers(0, 2, (n, 1))
    np.savez(tmp_path / "fakemnist.npz", **arrs)
    raw = load_medmnist_raw("fakemnist", str(tmp_path))
    assert raw.image_shape == (28, 28)
    assert abs(raw.y_train.mean() - 0.5) < 1e-9  # checking the classes were balanced
    # Expecting a friendly error (not a crash) when the file is missing.
    with pytest.raises(DatasetUnavailable):
        load_medmnist_raw("pneumoniamnist", str(tmp_path / "nowhere"))


def test_gradient_stats_and_barren_trend():
    # Checking the gradient matrix shape and that the statistics are positive.
    g = strongly_entangling(4, 2)
    inputs = np.random.default_rng(0).uniform(0, np.pi, (24, 4))
    grads = sample_gradients(g, 4, 4, inputs, seed=0)
    assert grads.shape == (24, 24)
    st = grad_stats(grads)
    assert st["var_mean"] > 0 and st["n_params"] == 24


def test_barren_plateau_trend():
    from qnas.gradients import barren_plateau_scan
    # Reproducing the barren-plateau trend: variance falls from 2 to 6 qubits.
    rows = barren_plateau_scan(qubits=(2, 6), n_samples=60, costs=("z0",), seed=0)
    for t in {r["template"] for r in rows}:
        v = {r["n_qubits"]: r["var_mean"] for r in rows if r["template"] == t}
        assert v[6] < v[2]  # expecting variance to shrink as qubits are added


def test_noise_zero_equals_ideal_accuracy(toy_task):
    from qnas.model import HybridQuantumModel
    # Comparing the ideal simulator with a practically noise-free mixed-state run.
    g = strongly_entangling(4, 1)
    m = HybridQuantumModel(g, 4, 4)
    state = {"weights": m.quantum_weights().tolist(), "scale": float(m.scale.detach()), "bias": float(m.bias.detach())}
    a = circuit_accuracy(g, state, toy_task.x_val[:20], toy_task.y_val[:20], 4, 4, 0.0, 0.0)
    b = circuit_accuracy(g, state, toy_task.x_val[:20], toy_task.y_val[:20], 4, 4, 1e-12, 1e-12)
    assert a == b


def test_search_runs_and_resumes(toy_task, tmp_path):
    # Running a tiny search, then resuming it for one more generation from the checkpoint.
    space = GeneSpace(max_genes=16, init_max_genes=8)
    cfg = SearchConfig(pop_size=6, generations=2, seed=1)
    out = tmp_path / "s"
    QNASSearch(toy_task, space, SPEC, cfg, out_dir=str(out), log=lambda *_: None).run()
    assert (out / "pareto.json").exists() and (out / "history.csv").exists() and (out / "checkpoint.pkl").exists()
    front = json.loads((out / "pareto.json").read_text())
    assert len(front) >= 1
    cfg2 = SearchConfig(pop_size=6, generations=3, seed=1)
    s2 = QNASSearch(toy_task, space, SPEC, cfg2, out_dir=str(out), log=lambda *_: None)
    s2.run(resume=True)
    assert s2.gen == 3
    rows = (out / "history.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + 4  # counting header + generations 0..3: resumed, not restarted


def test_cli_info_and_help(capsys):
    # Calling the CLI directly and checking info and --help both work.
    from qnas.cli import main
    assert main(["info"]) == 0
    assert "pennylane" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--help"])
