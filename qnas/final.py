"""Final, fair comparison of evolved circuits against the baselines.

Every model is re-trained **from scratch** for ``epochs`` epochs with ``n_seeds`` different seeds on the
training split and then scored on the *test* split, which the search never saw.  Results are reported as
mean +- std over seeds together with the circuit cost (depth, two-qubit gates, parameters), plus Welch
t-tests of the reference QNAS circuit against each baseline.

Models
------
* quantum circuits : evolved candidates (QNAS-*) and the hand-crafted SEL-L* / BEL-L* templates
* ``Linear``       : logistic regression (``nn.Linear``) on the same classical features
* ``MLP-matched``  : MLP whose parameter count is matched to the reference QNAS circuit (at least 2 tanh hidden units)
* ``MLP-SEL2``     : MLP matched to the parameter count of the SEL-L2 baseline
"""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/final.py
# Purpose: retraining every model from scratch on several seeds and comparing them
#          once on the untouched test split, with Welch t-tests.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import multiprocessing as mp
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

from .data import TaskData
from .fitness import EvalSpec
from .genome import Gene, circuit_depth, count_two_qubit, genome_to_json, genome_from_json
from .model import HybridQuantumModel, LinearBaseline, MLPBaseline, mlp_hidden_for_params
from .train import TrainConfig, fit

SEED_OFFSET = 1000  # keeping final-training seeds disjoint from the seeds used during the search


@dataclass
class FinalConfig:
    # Setting the length of final training, the number of seeds and the worker count.
    epochs: int = 30
    n_seeds: int = 5
    workers: int = 1


# ------------------------------------------------------------------------------
# Running one training job (module level, so multiprocessing can pickle it)
# ------------------------------------------------------------------------------
# Holding the shared task / spec / config inside each worker process.
_FINAL: dict = {}


def _init_final(task: TaskData, spec: EvalSpec, cfg: FinalConfig) -> None:
    import torch

    # Pinning one thread per worker and storing the shared objects once.
    torch.set_num_threads(1)
    _FINAL.update(task=task, spec=spec, cfg=cfg)


def run_job(task: TaskData, spec: EvalSpec, cfg: FinalConfig, job: dict) -> dict:
    import torch

    # Seeding this run and rebuilding the same training settings as the search, but longer.
    seed = SEED_OFFSET + job["seed"]
    torch.manual_seed(seed)
    train = TrainConfig(epochs=cfg.epochs, batch_size=spec.train.batch_size, lr=spec.train.lr, init_std=spec.train.init_std)
    # Building the requested model: a circuit, logistic regression or a small MLP.
    if job["kind"] == "quantum":
        genes = genome_from_json(job["genes"])
        model = HybridQuantumModel(genes, spec.n_qubits, spec.n_features, init_std=train.init_std, observable=spec.observable)
        depth, n2q, n_gates = circuit_depth(genes), count_two_qubit(genes), len(genes)
    elif job["kind"] == "linear":
        model, depth, n2q, n_gates = LinearBaseline(spec.n_features), 0, 0, 0
    elif job["kind"] == "mlp":
        model, depth, n2q, n_gates = MLPBaseline(spec.n_features, job["hidden"]), 0, 0, 0
    else:
        raise ValueError(job["kind"])
    # Training on the training split; validation and test are only evaluated, never trained on
    # (per-epoch test scores are logged for the learning-curve figure, not used for any choice).
    res = fit(model, task.x_train, task.y_train, train, seed,
              eval_sets={"val": (task.x_val, task.y_val), "test": (task.x_test, task.y_test)}, eval_every_epoch=True)
    out = {
        "name": job["name"], "kind": job["kind"], "seed": job["seed"],
        "val_acc": res["final"]["val"]["acc"], "test_acc": res["final"]["test"]["acc"],
        "test_bal_acc": res["final"]["test"]["bal_acc"], "test_loss": res["final"]["test"]["loss"],
        "depth": depth, "n_2q": n2q, "n_gates": n_gates, "n_params": model.n_parameters,
        "curve": [{"epoch": h["epoch"], "loss": h["loss"], "grad_norm": h["grad_norm"],
                   "val_acc": h["val"]["acc"], "test_acc": h["test"]["acc"]} for h in res["history"]],
        "time_s": res["time_s"],
    }
    # Keeping the learned angles so a trained circuit can be inspected later.
    if job["kind"] == "quantum":
        out["state"] = {"weights": model.quantum_weights().tolist(), "scale": float(model.scale.detach()), "bias": float(model.bias.detach())}
    return out


def _job_worker(job: dict) -> dict:
    # Running one job inside a worker process.
    return run_job(_FINAL["task"], _FINAL["spec"], _FINAL["cfg"], job)


# ------------------------------------------------------------------------------
# Orchestrating all jobs
# ------------------------------------------------------------------------------
def build_jobs(quantum_models: Dict[str, Sequence[Gene]], reference: str, spec: EvalSpec, cfg: FinalConfig,
               include_classical: bool = True) -> List[dict]:
    # Creating one job per (model, seed) for every quantum circuit.
    jobs = []
    for name, genes in quantum_models.items():
        for s in range(cfg.n_seeds):
            jobs.append({"name": name, "kind": "quantum", "genes": genome_to_json(genes), "seed": s})
    # Adding the classical baselines, with MLP sizes matched to the reference and to SEL-L2.
    if include_classical:
        ref_params = sum(1 for g in quantum_models[reference] if g.is_rotation and g.mode == "train") + 2
        hidden = mlp_hidden_for_params(spec.n_features, ref_params)
        for s in range(cfg.n_seeds):
            jobs.append({"name": "Linear", "kind": "linear", "seed": s})
            jobs.append({"name": "MLP-matched", "kind": "mlp", "hidden": hidden, "seed": s})
        if "SEL-L2" in quantum_models:
            sel_params = sum(1 for g in quantum_models["SEL-L2"] if g.is_rotation and g.mode == "train") + 2
            hidden2 = mlp_hidden_for_params(spec.n_features, sel_params)
            for s in range(cfg.n_seeds):
                jobs.append({"name": "MLP-SEL2", "kind": "mlp", "hidden": hidden2, "seed": s})
    return jobs


def run_final(task: TaskData, spec: EvalSpec, cfg: FinalConfig, jobs: List[dict],
              log: Callable[[str], None] = print) -> List[dict]:
    log(f"final evaluation: {len(jobs)} training runs ({cfg.epochs} epochs each) on {cfg.workers} worker(s)")
    # Spreading the jobs over a process pool, or running them one by one.
    if cfg.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(cfg.workers, initializer=_init_final, initargs=(task, spec, cfg)) as pool:
            results = []
            for i, r in enumerate(pool.imap(_job_worker, jobs, chunksize=1), 1):
                results.append(r)
                if i % max(1, len(jobs) // 10) == 0 or i == len(jobs):
                    log(f"  {i}/{len(jobs)} runs done")
            return results
    results = []
    for i, job in enumerate(jobs, 1):
        results.append(run_job(task, spec, cfg, job))
        if i % max(1, len(jobs) // 10) == 0 or i == len(jobs):
            log(f"  {i}/{len(jobs)} runs done")
    return results


def summarize(results: List[dict]) -> Dict[str, dict]:
    """Group per-seed results by model name -> mean/std statistics and mean learning curves."""
    # Grouping the per-seed runs by model name.
    by_name: Dict[str, List[dict]] = {}
    for r in results:
        by_name.setdefault(r["name"], []).append(r)
    out = {}
    for name, rs in by_name.items():
        # Computing mean / sample-std test accuracy and the seed-averaged learning curves.
        test = np.array([r["test_acc"] for r in rs])
        n_epochs = len(rs[0]["curve"])
        curve = {k: np.mean([[c[k] for c in r["curve"]] for r in rs], axis=0).tolist()
                 for k in ("loss", "grad_norm", "val_acc", "test_acc")}
        out[name] = {
            "kind": rs[0]["kind"], "n_seeds": len(rs),
            "test_acc_mean": float(test.mean()), "test_acc_std": float(test.std(ddof=1)) if len(rs) > 1 else 0.0,
            "test_acc_runs": test.tolist(),
            "val_acc_mean": float(np.mean([r["val_acc"] for r in rs])),
            "test_bal_acc_mean": float(np.mean([r["test_bal_acc"] for r in rs])),
            "depth": rs[0]["depth"], "n_2q": rs[0]["n_2q"], "n_gates": rs[0]["n_gates"], "n_params": rs[0]["n_params"],
            "curve": {"epoch": list(range(1, n_epochs + 1)), **curve},
        }
    return out


def compare_to_reference(summary: Dict[str, dict], reference: str) -> List[dict]:
    """Welch t-test + effect sizes of ``reference`` vs every other model (test accuracy over seeds)."""
    from scipy import stats

    ref = summary[reference]
    rows = []
    for name, s in summary.items():
        if name == reference:
            continue
        # Running a two-sided Welch t-test (unequal variances) on the per-seed accuracies.
        a, b = np.array(ref["test_acc_runs"]), np.array(s["test_acc_runs"])
        if len(a) > 1 and len(b) > 1 and (a.std() > 0 or b.std() > 0):
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        else:
            p = 1.0 if np.isclose(a.mean(), b.mean()) else float("nan")
        # Reporting the accuracy gap next to the depth and two-qubit savings.
        rows.append({
            "model": name, "acc_diff": float(a.mean() - b.mean()), "p_value": p,
            "n_2q_ref": ref["n_2q"], "n_2q_other": s["n_2q"],
            "n_2q_reduction": (1 - ref["n_2q"] / s["n_2q"]) if s["n_2q"] else float("nan"),
            "depth_ref": ref["depth"], "depth_other": s["depth"],
            "depth_reduction": (1 - ref["depth"] / s["depth"]) if s["depth"] else float("nan"),
        })
    return rows


def save_final(out_dir: Path, results: List[dict], summary: Dict[str, dict], comparison: List[dict],
               candidates: Dict[str, Sequence[Gene]]) -> None:
    # Writing raw runs, summaries, comparisons and the circuits used.
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_runs.json").write_text(json.dumps(results))
    (out_dir / "final_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "final_comparison.json").write_text(json.dumps(comparison, indent=2))
    (out_dir / "candidates.json").write_text(json.dumps({k: genome_to_json(v) for k, v in candidates.items()}, indent=2))
    import csv

    # Writing a CSV table sorted by test accuracy for quick reading.
    with open(out_dir / "final_summary.csv", "w", newline="") as f:
        cols = ["model", "kind", "test_acc_mean", "test_acc_std", "val_acc_mean", "test_bal_acc_mean", "n_params", "n_2q", "depth", "n_gates"]
        w = csv.writer(f)
        w.writerow(cols)
        for name, s in sorted(summary.items(), key=lambda kv: -kv[1]["test_acc_mean"]):
            w.writerow([name, s["kind"], f"{s['test_acc_mean']:.4f}", f"{s['test_acc_std']:.4f}", f"{s['val_acc_mean']:.4f}",
                        f"{s['test_bal_acc_mean']:.4f}", s["n_params"], s["n_2q"], s["depth"], s["n_gates"]])
