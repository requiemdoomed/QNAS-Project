"""End-to-end pipeline: one ``Run`` = one directory of reproducible artifacts.

Stages (each can be run on its own and resumes from the files written by the previous ones)::

    search      evolutionary architecture search            -> search/*
    baselines   hand-crafted circuits under the same protocol -> baselines.json
    candidates  pick QNAS-bestF / -accurate / -compact, simplify + prune -> candidates.json
    final       multi-seed retraining, test-set comparison    -> final/*
    gradients   barren-plateau scan + per-circuit gradients   -> gradients/*
    noise       accuracy under depolarizing noise (optional)  -> noise/*
    report      figures + results.md                          -> figures/*, results.md
"""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/pipeline.py
# Purpose: chaining every stage (search -> baselines -> candidates -> final ->
#          gradients -> noise -> report) inside one reproducible run folder.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional

from .baselines import evaluate_baselines
from .data import TaskData, prepare_task
from .evolution import QNASSearch, SearchConfig, select_candidates
from .final import FinalConfig, build_jobs, compare_to_reference, run_final, save_final, summarize
from .fitness import EvalSpec
from .genome import Gene, genome_from_json, genome_to_json, structure_summary
from .gradients import barren_plateau_scan, candidate_gradients
from .noise import noise_sweep
from .simplify import prune_by_ablation, simplify
from .space import GeneSpace
from .templates import baseline_suite
from .train import TrainConfig

# Naming the ready-made binary digit tasks the command line can pick from.
TASKS = {
    "even-odd": [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]],
    "low-high": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
    "3v5": [[3], [5]],
    "0v1": [[0], [1]],
}


@dataclass
class RunConfig:
    # ---- task / classical feature extractor
    dataset: str = "digits"
    classes: list = field(default_factory=lambda: [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]])
    extractor: str = "cnn"
    ae_epochs: int = 60
    data_root: str = "data"
    medmnist_size: int = 28
    max_per_class: Optional[int] = None
    # ---- circuit search space
    n_qubits: int = 4
    max_depth: int = 20       # allowing depth 20 so every baseline (up to SEL-L3, depth 19) fits in the space
    max_genes: int = 48
    init_max_genes: int = 14
    connectivity: str = "all"
    allow_data: bool = False
    # ---- evolutionary search
    mode: str = "nsga2"
    objectives: list = field(default_factory=lambda: ["acc", "depth", "n2q"])
    pop_size: int = 20
    generations: int = 30
    p_crossover: float = 0.8
    p_mutation: float = 0.2
    p_insert: float = 0.25
    p_delete: float = 0.2
    warm_start: bool = True   # seeding generation 0 with templates (--cold-start = pure random start)
    # ---- inner-loop training + scalar fitness
    epochs: int = 5
    batch_size: int = 64
    lr: float = 0.1
    init_std: float = 0.3
    alpha: float = 1.0
    beta: float = 0.1
    gamma: float = 0.2
    # ---- final evaluation
    final_epochs: int = 30
    final_seeds: int = 5
    prune: bool = True
    prune_seeds: int = 3
    prune_tolerance: float = 0.005
    # ---- analyses
    grad_qubits: list = field(default_factory=lambda: [2, 4, 6, 8, 10])
    grad_samples: int = 100
    noise: bool = True
    noise_p: list = field(default_factory=lambda: [0.0, 0.001, 0.003, 0.01, 0.03])
    # ---- misc
    seed: int = 0
    workers: int = 1

    # ---------------------------------------------------------------- (de)serialisation
    def to_json(self) -> dict:
        # Saving the config as plain JSON next to the results.
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "RunConfig":
        # Ignoring unknown keys so configs from older versions still load.
        names = {f.name for f in fields(RunConfig)}
        return RunConfig(**{k: v for k, v in d.items() if k in names})

    # ---------------------------------------------------------------- presets
    @staticmethod
    def preset(name: str, **overrides) -> "RunConfig":
        presets = {
            # Smoke-testing the whole pipeline (about 4 minutes on 2 cores).
            "quick": dict(pop_size=10, generations=6, epochs=3, ae_epochs=30, final_epochs=12, final_seeds=2,
                          prune_seeds=1, grad_qubits=[2, 4, 6], grad_samples=50),
            # Running the reported experiment (about 25 minutes on 2 cores).
            "standard": dict(pop_size=30, generations=40),
            # Running a longer search with more seeds (several hours on 2 cores).
            "thorough": dict(pop_size=30, generations=50, epochs=8, final_epochs=40, final_seeds=10, grad_samples=200),
        }
        if name not in presets:
            raise ValueError(f"unknown preset {name!r}; choose from {sorted(presets)}")
        # Starting from the preset and applying any command-line overrides on top.
        cfg = RunConfig(**presets[name])
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg


class Run:
    """A run directory plus lazily built task / search space / evaluation spec."""

    def __init__(self, directory, cfg: Optional[RunConfig] = None, quiet: bool = False):
        # Creating the run folder and either saving a new config or loading the existing one.
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self.dir / "config.json"
        if cfg is not None:
            self.cfg = cfg
            cfg_path.write_text(json.dumps(cfg.to_json(), indent=2))
        elif cfg_path.exists():
            self.cfg = RunConfig.from_json(json.loads(cfg_path.read_text()))
        else:
            raise FileNotFoundError(f"{cfg_path} not found - start the run with 'qnas search' or 'qnas run-all'")
        self.quiet = quiet
        self._task: Optional[TaskData] = None

    # ---------------------------------------------------------------- helpers
    def log(self, msg: str) -> None:
        # Printing progress and appending it to log.txt in the run folder.
        if not self.quiet:
            print(msg, flush=True)
        with open(self.dir / "log.txt", "a") as f:
            f.write(msg + "\n")

    @property
    def task(self) -> TaskData:
        # Preparing the features once per run (and caching them on disk).
        if self._task is None:
            c = self.cfg
            self._task = prepare_task(
                dataset=c.dataset, classes=[tuple(g) for g in c.classes] if c.dataset == "digits" else c.classes,
                n_features=c.n_qubits, extractor=c.extractor, seed=c.seed, root=c.data_root, size=c.medmnist_size,
                max_per_class=c.max_per_class, ae_epochs=c.ae_epochs, cache_dir=str(self.dir / "cache"))
            (self.dir / "task_info.json").write_text(json.dumps(self._task.info, indent=2, default=str))
        return self._task

    @property
    def space(self) -> GeneSpace:
        # Building the search space from the run config.
        c = self.cfg
        return GeneSpace(n_qubits=c.n_qubits, n_features=c.n_qubits, max_depth=c.max_depth, max_genes=c.max_genes,
                         init_max_genes=c.init_max_genes, connectivity=c.connectivity, allow_data=c.allow_data)

    @property
    def spec(self) -> EvalSpec:
        # Building the scoring protocol (inner training + fitness weights) from the run config.
        c = self.cfg
        return EvalSpec(n_qubits=c.n_qubits, n_features=c.n_qubits, max_depth=c.max_depth, max_genes=c.max_genes,
                        alpha=c.alpha, beta=c.beta, gamma=c.gamma, base_seed=c.seed,
                        train=TrainConfig(epochs=c.epochs, batch_size=c.batch_size, lr=c.lr, init_std=c.init_std))

    def _read(self, rel: str):
        # Reading a JSON file written by an earlier stage.
        return json.loads((self.dir / rel).read_text())

    def _write(self, rel: str, obj) -> None:
        # Writing JSON, converting numpy values to plain Python on the way.
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))

    # ---------------------------------------------------------------- stage: search
    def search(self, resume: bool = False) -> QNASSearch:
        c = self.cfg
        task = self.task
        self.log(f"task: {c.dataset} {c.classes} | extractor={c.extractor} | sizes={task.sizes()} | "
                 f"{c.n_qubits} qubits | mode={c.mode} | pop={c.pop_size} gens={c.generations} epochs={c.epochs}")
        scfg = SearchConfig(pop_size=c.pop_size, generations=c.generations, p_crossover=c.p_crossover,
                            p_mutation=c.p_mutation, p_insert=c.p_insert, p_delete=c.p_delete, mode=c.mode,
                            objectives=tuple(c.objectives), seed=c.seed, workers=c.workers)
        # Warm-starting generation 0 with the hand-made templates (up to half the population).
        initial = list(baseline_suite(c.n_qubits, 3).values())[: max(1, c.pop_size // 2)] if c.warm_start else None
        s = QNASSearch(task, self.space, self.spec, scfg, out_dir=str(self.dir / "search"), log=self.log,
                       initial_genomes=initial)
        return s.run(resume=resume)

    # ---------------------------------------------------------------- stage: baselines
    def baselines(self) -> dict:
        self.log("evaluating hand-crafted baselines with the search protocol ...")
        res = evaluate_baselines(self.task, self.spec, max_layers=3)
        # Logging each template on the same axes as the evolved circuits.
        for name, r in res.items():
            m = r["metrics"]
            self.log(f"  {name:<7} acc {m['val_acc']:.3f} | depth {m['depth']:>2} | 2q {m['n_2q']:>2} | params {m['n_params']:>3}")
        self._write("baselines.json", res)
        return res

    # ---------------------------------------------------------------- stage: candidates
    def candidates(self) -> Dict[str, dict]:
        c = self.cfg
        # Reading the archive Pareto front and picking the three named circuits.
        front = [{**r, "genes": genome_from_json(r["genes"])} for r in self._read("search/pareto.json")]
        picked = select_candidates(front)
        out: Dict[str, dict] = {}
        self.log("selected candidates from the Pareto front:")
        for name, genes in picked.items():
            # Simplifying (and, if enabled, pruning) each pick, keeping the original for reference.
            before = structure_summary(genes)
            if c.prune:
                self.log(f"  pruning {name} ({before['n_gates']} gates, {before['n_2q']} two-qubit) ...")
                final_genes, info = prune_by_ablation(genes, self.task, self.spec, n_seeds=c.prune_seeds,
                                                      tolerance=c.prune_tolerance, log=self.log)
            else:
                final_genes, info = simplify(genes), {}
            after = structure_summary(final_genes)
            self.log(f"  {name}: gates {before['n_gates']}->{after['n_gates']} | 2q {before['n_2q']}->{after['n_2q']} | "
                     f"depth {before['depth']}->{after['depth']}")
            out[name] = {"genes": genome_to_json(final_genes), "original_genes": genome_to_json(genes),
                         "before": before, "after": after, "prune": info}
        self._write("candidates.json", out)
        return out

    def load_candidates(self) -> Dict[str, List[Gene]]:
        # Loading the processed candidate circuits.
        return {k: genome_from_json(v["genes"]) for k, v in self._read("candidates.json").items()}

    # ---------------------------------------------------------------- stage: final
    def final(self) -> dict:
        c = self.cfg
        models: Dict[str, List[Gene]] = dict(self.load_candidates())
        # Training the un-pruned (only losslessly simplified) circuits too, to show what pruning cost.
        if c.prune:
            for name, rec in self._read("candidates.json").items():
                full = simplify(genome_from_json(rec["original_genes"]))
                if len(full) > len(models[name]):
                    models[f"{name}-full"] = full
        # Adding the six templates, then running every model on every seed.
        models.update(baseline_suite(c.n_qubits, 3))
        fcfg = FinalConfig(epochs=c.final_epochs, n_seeds=c.final_seeds, workers=c.workers)
        jobs = build_jobs(models, "QNAS-bestF", self.spec, fcfg)
        results = run_final(self.task, self.spec, fcfg, jobs, log=self.log)
        # Comparing each QNAS model against all others and saving the tables.
        summary = summarize(results)
        comparisons = {n: compare_to_reference(summary, n) for n in models if n.startswith("QNAS")}
        save_final(self.dir / "final", results, summary, comparisons, self.load_candidates())
        for name, s in sorted(summary.items(), key=lambda kv: -kv[1]["test_acc_mean"]):
            self.log(f"  {name:<14} test acc {s['test_acc_mean']*100:5.1f}% ± {s['test_acc_std']*100:.1f} | "
                     f"2q {s['n_2q']:>2} | depth {s['depth']:>2} | params {s['n_params']:>3}")
        return summary

    # ---------------------------------------------------------------- stage: gradients
    def gradients(self) -> dict:
        c = self.cfg
        self.log("barren-plateau scan ...")
        # Measuring Var[grad] against qubit count, then gradients of the concrete 4-qubit circuits.
        scan = barren_plateau_scan(qubits=tuple(c.grad_qubits), n_samples=c.grad_samples, seed=c.seed, log=self.log)
        circuits = dict(self.load_candidates())
        for name in ("SEL-L1", "SEL-L2", "SEL-L3", "BEL-L2"):
            circuits[name] = baseline_suite(c.n_qubits, 3)[name]
        cand = candidate_gradients(circuits, c.n_qubits, c.n_qubits, self.task.x_train,
                                   n_samples=max(c.grad_samples, 100), seed=c.seed)
        out = {"scan": scan, "candidates": cand}
        self._write("gradients/gradients.json", out)
        return out

    # ---------------------------------------------------------------- stage: noise
    def noise(self) -> List[dict]:
        c = self.cfg
        # Re-using the seed-0 trained weights from the final stage (no retraining here).
        runs = self._read("final/final_runs.json")
        cands = self.load_candidates()
        suite = baseline_suite(c.n_qubits, 3)
        models = {}
        for name in list(cands) + ["SEL-L2", "BEL-L2"]:
            genes = cands.get(name) or suite[name]
            r0 = next(r for r in runs if r["name"] == name and r["seed"] == 0)
            models[name] = (genes, r0["state"])
        self.log("noisy-hardware sweep (default.mixed) ...")
        rows = noise_sweep(models, self.task.x_test, self.task.y_test, c.n_qubits, c.n_qubits,
                           p_values=c.noise_p, seed=c.seed)
        for r in rows:
            if r["p_1q"] in (0.0, c.noise_p[-1]):
                self.log(f"  {r['circuit']:<14} p={r['p_1q']:<6g} acc {r['accuracy']:.3f}")
        self._write("noise/noise.json", rows)
        return rows

    # ---------------------------------------------------------------- stage: report
    def report(self) -> str:
        # Importing lazily so matplotlib is only needed for the report stage.
        from .report import build_report

        return build_report(self)


def run_all(directory, cfg: RunConfig, resume: bool = False, quiet: bool = False) -> Run:
    """Run every stage in order and write the report."""
    # Running the stages in order; each one reads what the previous one wrote.
    run = Run(directory, cfg, quiet=quiet)
    t0 = time.time()
    run.search(resume=resume)
    run.baselines()
    run.candidates()
    run.final()
    run.gradients()
    if cfg.noise:
        run.noise()
    run.report()
    run.log(f"done in {time.time() - t0:.0f}s -> {run.dir}/results.md")
    return run
