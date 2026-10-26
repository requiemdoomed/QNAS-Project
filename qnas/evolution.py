"""The evolutionary search loop (NSGA-II or a weighted-fitness genetic algorithm) built on DEAP.

Loop::

    genotype --compile--> QNode --train--> metrics --> select / crossover / mutate --> genotype ...

Two modes
---------
``nsga2``     true multi-objective search over (accuracy, depth, #2-qubit gates[, #params]);
              maintains a Pareto front (RQ2).
``weighted``  single-objective GA that maximises the scalar fitness
              ``F = alpha*acc - beta*depth/D_max - gamma*N_2q/N_max`` with tournament selection.

Design notes
------------
* Scores are **deterministic per genome** (seeded from a hash of the genome), cached, and
  independent of the number of worker processes -> reproducible runs and free re-evaluation.
* Duplicates are removed before survivor selection and replaced with random immigrants, which
  keeps the population diverse.
* A checkpoint is written after each generation, so a run can be stopped and resumed.
"""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/evolution.py
# Purpose: running the evolutionary search (NSGA-II or a weighted GA) on top of DEAP,
#          with caching, parallel workers, checkpoints and a Pareto archive.
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from deap import base, tools

from .data import TaskData
from .fitness import (
    EvalSpec,
    evaluate_genome,
    objective_vector,
    objective_weights,
)
from .genome import Gene, describe, genome_hash, genome_key, genome_to_json
from .pareto import hypervolume, non_dominated
from .space import GeneSpace


# ------------------------------------------------------------------------------
# Configuring the search
# ------------------------------------------------------------------------------
@dataclass
class SearchConfig:
    # Collecting every knob of the genetic algorithm in one place.
    pop_size: int = 20
    generations: int = 15
    p_crossover: float = 0.8      # setting P_c, the chance a pair of parents is recombined
    p_mutation: float = 0.2       # setting P_m, the per-gene mutation probability
    p_insert: float = 0.25        # inserting one random gate into a child with this probability
    p_delete: float = 0.20        # deleting one gate from a child with this probability
    mode: str = "nsga2"           # choosing "nsga2" or "weighted"
    objectives: tuple = ("acc", "depth", "n2q")
    tournament_size: int = 3      # sizing the tournament of the weighted GA
    seed: int = 0
    workers: int = 1

    def __post_init__(self):
        # Checking the settings and normalising the objectives to a tuple.
        if self.mode not in ("nsga2", "weighted"):
            raise ValueError("mode must be 'nsga2' or 'weighted'")
        if self.pop_size < 4:
            raise ValueError("pop_size must be >= 4")
        self.objectives = tuple(self.objectives)


# ------------------------------------------------------------------------------
# Wiring up worker processes (module level so multiprocessing can pickle it)
# ------------------------------------------------------------------------------
# Keeping the task and spec in a module-level dict so each worker process
# receives them once (at start-up) rather than with every job.
_WORKER: dict = {}


def _init_worker(task: TaskData, spec: EvalSpec) -> None:
    import torch

    # Pinning each worker to one thread so parallel workers do not fight for cores.
    torch.set_num_threads(1)
    _WORKER["task"], _WORKER["spec"] = task, spec


def _eval_worker(genes: List[Gene]) -> Dict[str, float]:
    # Scoring one genome inside a worker process.
    return evaluate_genome(genes, _WORKER["task"], _WORKER["spec"])


# ------------------------------------------------------------------------------
# Defining the DEAP individual
# ------------------------------------------------------------------------------
def make_individual_class(weights: Sequence[float]):
    # Creating a DEAP fitness class with the right objective directions, and a
    # list-based individual that also carries its metrics and Pareto rank.
    fitness_cls = type("QNASFitness", (base.Fitness,), {"weights": tuple(weights)})

    class Individual(list):
        def __init__(self, genes: Sequence[Gene] = ()):
            super().__init__(genes)
            self.fitness = fitness_cls()
            self.metrics: Dict[str, float] = {}
            self.rank = 0

    return Individual


def normalized_vector(metrics: Dict[str, float], names: Sequence[str], spec: EvalSpec) -> tuple:
    """Objective vector in minimisation form, each scaled to ~[0, 1] (used for hypervolume)."""
    # Turning every objective into "smaller is better" on roughly the same scale.
    out = []
    for n in names:
        if n == "acc":
            out.append(1.0 - metrics["val_acc"])
        elif n == "depth":
            out.append(metrics["depth"] / (spec.max_depth + 1))
        elif n == "n2q":
            out.append(metrics["n_2q"] / (spec.n_max + 1))
        elif n == "params":
            out.append(metrics["n_params"] / (3 * spec.max_genes + 3))
        else:
            raise ValueError(n)
    return tuple(out)


# ------------------------------------------------------------------------------
# Running the search
# ------------------------------------------------------------------------------
class QNASSearch:
    def __init__(
        self,
        task: TaskData,
        space: GeneSpace,
        spec: EvalSpec,
        cfg: SearchConfig,
        out_dir: Optional[str] = None,
        log: Callable[[str], None] = print,
        initial_genomes: Optional[Sequence[Sequence[Gene]]] = None,
    ):
        # Storing the problem, the operators and the scoring protocol.
        self.task, self.space, self.spec, self.cfg = task, space, spec, cfg
        self.out_dir = Path(out_dir) if out_dir else None
        self.log = log
        self.initial_genomes = [list(g) for g in (initial_genomes or [])]
        # Choosing the DEAP fitness: several objectives for NSGA-II, one scalar for the weighted GA.
        self.names = cfg.objectives if cfg.mode == "nsga2" else ("fitness",)
        weights = objective_weights(cfg.objectives) if cfg.mode == "nsga2" else (1.0,)
        self.Individual = make_individual_class(weights)
        # Setting up the seeded RNG, the evaluation cache and the run bookkeeping.
        self.rng = random.Random(cfg.seed)
        self.cache: Dict[tuple, Dict[str, float]] = {}   # mapping genome -> metrics for every unique evaluation
        self.first_gen: Dict[tuple, int] = {}
        self.history: List[dict] = []
        self.pop: list = []
        self.gen = -1
        self._pool = None
        self._new_evals = 0
        self._t0 = time.time()
        self._elapsed_before = 0.0
        self.ref = tuple(1.0 for _ in cfg.objectives)

    # ------------------------------------------------------------------ evaluation
    def _evaluate_genomes(self, genomes: List[List[Gene]]) -> List[Dict[str, float]]:
        # Collecting only genomes never scored before (and each one only once).
        todo, seen = [], set()
        for g in genomes:
            k = genome_key(g)
            if k not in self.cache and k not in seen:
                seen.add(k)
                todo.append(g)
        # Scoring the new genomes in parallel when a pool exists, otherwise in-process.
        if todo:
            if self._pool is not None and len(todo) > 1:
                results = self._pool.map(_eval_worker, todo, chunksize=1)
            else:
                results = [evaluate_genome(g, self.task, self.spec) for g in todo]
            for g, m in zip(todo, results):
                self.cache[genome_key(g)] = m
                self.first_gen[genome_key(g)] = max(self.gen, 0)
        self._new_evals += len(todo)
        return [self.cache[genome_key(g)] for g in genomes]

    def _evaluate(self, inds: list) -> None:
        # Copying the metrics onto each individual and setting its DEAP fitness.
        for ind, m in zip(inds, self._evaluate_genomes([list(i) for i in inds])):
            ind.metrics = m
            if self.cfg.mode == "nsga2":
                ind.fitness.values = objective_vector(m, self.cfg.objectives)
            else:
                ind.fitness.values = (m["fitness"],)

    # ------------------------------------------------------------------ selection helpers
    def _assign_rank_crowding(self, pop: list) -> None:
        # Sorting into non-dominated fronts and storing rank + crowding distance on each individual.
        for rank, front in enumerate(tools.sortNondominated(pop, len(pop))):
            tools.emo.assignCrowdingDist(front)
            for ind in front:
                ind.rank = rank

    def _tournament(self, pop: list, k: int) -> list:
        chosen = []
        for _ in range(k):
            if self.cfg.mode == "nsga2":  # running a binary tournament on (rank, crowding distance)
                a, b = self.rng.sample(pop, 2)
                key = lambda i: (i.rank, -i.fitness.crowding_dist)  # noqa: E731
                chosen.append(a if key(a) <= key(b) else b)
            else:  # running a k-way tournament on the scalar fitness F
                cands = self.rng.sample(pop, min(self.cfg.tournament_size, len(pop)))
                chosen.append(max(cands, key=lambda i: i.fitness.values[0]))
        return chosen

    def _vary(self, parents: list) -> list:
        # Pairing parents for crossover, then mutating every child.
        cfg = self.cfg
        genomes = [list(p) for p in parents]
        for i in range(0, len(genomes) - 1, 2):
            if self.rng.random() < cfg.p_crossover:
                genomes[i], genomes[i + 1] = self.space.crossover(genomes[i], genomes[i + 1], self.rng)
        genomes = [self.space.mutate(g, self.rng, cfg.p_mutation, cfg.p_insert, cfg.p_delete) for g in genomes]
        return [self.Individual(g) for g in genomes]

    def _immigrants(self, existing: set, n: int) -> list:
        # Drawing fresh random circuits that are not already in the population.
        out, tries = [], 0
        while len(out) < n and tries < 200 * max(n, 1):
            tries += 1
            g = self.space.random_genome(self.rng)
            k = genome_key(g)
            if k not in existing:
                existing.add(k)
                out.append(self.Individual(g))
        return out

    def _survivors(self, pop: list, offspring: list) -> list:
        # Removing duplicate circuits from parents + children before selection.
        unique, seen = [], set()
        for ind in pop + offspring:  # listing parents first, so ties favour the incumbents
            k = genome_key(ind)
            if k not in seen:
                seen.add(k)
                unique.append(ind)
        # Topping up with random immigrants when duplicates left the pool too small.
        if len(unique) < self.cfg.pop_size:
            fresh = self._immigrants(seen, self.cfg.pop_size - len(unique))
            self._evaluate(fresh)
            unique += fresh
        # Keeping the best pop_size individuals (NSGA-II ranking, or plain fitness).
        if self.cfg.mode == "nsga2":
            new = tools.selNSGA2(unique, self.cfg.pop_size)
            self._assign_rank_crowding(new)
            return new
        return sorted(unique, key=lambda i: i.fitness.values[0], reverse=True)[: self.cfg.pop_size]

    # ------------------------------------------------------------------ archive / front
    def archive_records(self) -> List[dict]:
        # Listing every circuit ever evaluated, with when it first appeared.
        recs = []
        for k, m in self.cache.items():
            recs.append({"genes": list(k), "metrics": m, "first_gen": self.first_gen.get(k, 0), "hash": genome_hash(k)})
        return recs

    def pareto_front(self) -> List[dict]:
        """Non-dominated set of *every* circuit evaluated during the search (the archive front)."""
        recs = self.archive_records()
        vecs = [normalized_vector(r["metrics"], self.cfg.objectives, self.spec) for r in recs]
        front = [recs[i] for i in non_dominated(vecs)]
        # Keeping one representative per objective vector (equal vectors count as duplicates).
        seen, uniq = set(), []
        for r in sorted(front, key=lambda r: (-r["metrics"]["val_acc"], r["metrics"]["n_2q"], r["metrics"]["depth"])):
            v = normalized_vector(r["metrics"], self.cfg.objectives, self.spec)
            if v not in seen:
                seen.add(v)
                uniq.append(r)
        return uniq

    # ------------------------------------------------------------------ logging
    def _gen_stats(self) -> dict:
        # Summarising the generation: best / mean scores, sizes and both hypervolumes.
        ms = [i.metrics for i in self.pop]
        pop_vecs = [normalized_vector(m, self.cfg.objectives, self.spec) for m in ms]
        front_vecs = [pop_vecs[i] for i in non_dominated(pop_vecs)]
        arch = self.pareto_front()
        arch_vecs = [normalized_vector(r["metrics"], self.cfg.objectives, self.spec) for r in arch]
        elapsed = self._elapsed_before + time.time() - self._t0
        return {
            "gen": self.gen,
            "evals_total": len(self.cache),
            "new_evals": self._new_evals,
            "best_val_acc": max(m["val_acc"] for m in ms),
            "mean_val_acc": float(np.mean([m["val_acc"] for m in ms])),
            "best_fitness": max(m["fitness"] for m in ms),
            "mean_depth": float(np.mean([m["depth"] for m in ms])),
            "mean_n2q": float(np.mean([m["n_2q"] for m in ms])),
            "mean_params": float(np.mean([m["n_params"] for m in ms])),
            "mean_grad_norm": float(np.mean([m["grad_norm_first_epoch"] for m in ms])),
            "hv_population": hypervolume(front_vecs, self.ref),
            "hv_archive": hypervolume(arch_vecs, self.ref),
            "front_size": len(arch),
            "unique_in_pop": len({genome_key(i) for i in self.pop}),
            "elapsed_s": elapsed,
        }

    def _log_gen(self, s: dict) -> None:
        # Printing one compact progress line per generation.
        self.log(
            f"gen {s['gen']:>3}/{self.cfg.generations} | evals {s['evals_total']:>4} (+{s['new_evals']:<3}) | "
            f"best acc {s['best_val_acc']:.3f} | best F {s['best_fitness']:.3f} | "
            f"mean 2q {s['mean_n2q']:.1f} depth {s['mean_depth']:.1f} | front {s['front_size']:>2} "
            f"| HV {s['hv_archive']:.4f} | {s['elapsed_s']:.0f}s"
        )

    # ------------------------------------------------------------------ persistence
    def _config_blob(self) -> dict:
        # Bundling all settings so a resume can check they did not change.
        return {"search": asdict(self.cfg), "spec": asdict(self.spec), "space": asdict(self.space)}

    def _save_checkpoint(self) -> None:
        if not self.out_dir:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        blob = {
            "gen": self.gen,
            "pop": [list(i) for i in self.pop],
            "cache": self.cache,
            "first_gen": self.first_gen,
            "history": self.history,
            "rng": self.rng.getstate(),
            "config": self._config_blob(),
            "elapsed": self._elapsed_before + time.time() - self._t0,
        }
        # Writing to a temporary file first and renaming it, so a crash never leaves
        # a half-written checkpoint behind.
        tmp = self.out_dir / "checkpoint.pkl.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(blob, f)
        os.replace(tmp, self.out_dir / "checkpoint.pkl")
        self._write_history_csv()

    def _write_history_csv(self) -> None:
        if not self.history:
            return
        # Saving one row per generation for the convergence plots.
        with open(self.out_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            w.writeheader()
            w.writerows(self.history)

    def _try_resume(self) -> bool:
        path = self.out_dir / "checkpoint.pkl" if self.out_dir else None
        if not path or not path.exists():
            return False
        with open(path, "rb") as f:
            blob = pickle.load(f)
        # Refusing to resume when a setting that changes scores or selection differs.
        old, new = blob["config"], self._config_blob()
        for part in ("spec", "space"):
            if old[part] != new[part]:
                raise RuntimeError(f"cannot resume: the '{part}' settings differ from the checkpoint")
        for key in ("pop_size", "mode", "objectives", "seed", "p_crossover", "p_mutation"):
            if old["search"][key] != new["search"][key]:
                raise RuntimeError(f"cannot resume: search setting '{key}' differs from the checkpoint")
        # Restoring the cache, history, RNG state and population.
        self.cache, self.first_gen, self.history = blob["cache"], blob["first_gen"], blob["history"]
        self.gen = blob["gen"]
        self.rng.setstate(blob["rng"])
        self._elapsed_before = blob["elapsed"]
        self.pop = [self.Individual(g) for g in blob["pop"]]
        self._evaluate(self.pop)  # restoring fitness values (all cache hits)
        self._new_evals = 0
        if self.cfg.mode == "nsga2":
            self._assign_rank_crowding(self.pop)
        self.log(f"resumed from checkpoint at generation {self.gen}")
        return True

    def save_results(self) -> None:
        """Write archive, Pareto front, final population and config to ``out_dir``."""
        if not self.out_dir:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Writing the config, the front, the full archive and the final population as JSON.
        (self.out_dir / "search_config.json").write_text(json.dumps(self._config_blob(), indent=2, default=list))

        def dump(recs):
            return [{"hash": r["hash"], "first_gen": r["first_gen"], "genes": genome_to_json(r["genes"]),
                     "description": describe(r["genes"]), "metrics": r["metrics"]} for r in recs]

        front = self.pareto_front()
        (self.out_dir / "pareto.json").write_text(json.dumps(dump(front), indent=2))
        archive = sorted(self.archive_records(), key=lambda r: -r["metrics"]["val_acc"])
        (self.out_dir / "archive.json").write_text(json.dumps(dump(archive)))
        pop_recs = [{"genes": list(i), "metrics": i.metrics, "first_gen": self.first_gen.get(genome_key(i), 0),
                     "hash": genome_hash(i)} for i in self.pop]
        (self.out_dir / "population.json").write_text(json.dumps(dump(pop_recs), indent=2))
        self._write_history_csv()

    # ------------------------------------------------------------------ main loop
    def run(self, resume: bool = False) -> "QNASSearch":
        cfg = self.cfg
        self._t0 = time.time()
        # Starting a "spawn" process pool (safe with PyTorch) when more than one worker is asked for.
        if cfg.workers > 1:
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(cfg.workers, initializer=_init_worker, initargs=(self.task, self.spec))
        try:
            resumed = resume and self._try_resume()
            if not resumed:
                # Building generation 0: the warm-start genomes first, then random circuits.
                self.gen = 0
                seen = set()
                init = []
                for g in self.initial_genomes:
                    g = self.space.repair(g, self.rng)
                    if genome_key(g) not in seen:
                        seen.add(genome_key(g))
                        init.append(self.Individual(g))
                init += self._immigrants(seen, cfg.pop_size - len(init))
                self.pop = init[: cfg.pop_size]
                self._new_evals = 0
                self._evaluate(self.pop)
                if cfg.mode == "nsga2":
                    self.pop = tools.selNSGA2(self.pop, len(self.pop))
                    self._assign_rank_crowding(self.pop)
                s = self._gen_stats()
                self.history.append(s)
                self._log_gen(s)
                self._save_checkpoint()
            # Looping: select parents, vary them, score the children, keep the survivors.
            while self.gen < cfg.generations:
                self.gen += 1
                self._new_evals = 0
                parents = self._tournament(self.pop, cfg.pop_size)
                offspring = self._vary(parents)
                self._evaluate(offspring)
                self.pop = self._survivors(self.pop, offspring)
                s = self._gen_stats()
                self.history.append(s)
                self._log_gen(s)
                self._save_checkpoint()
        finally:
            # Closing the worker pool even if the run was interrupted.
            if self._pool is not None:
                self._pool.close()
                self._pool.join()
                self._pool = None
        self.save_results()
        return self


# ------------------------------------------------------------------------------
# Picking named candidates from a Pareto front
# ------------------------------------------------------------------------------
def select_candidates(front: List[dict], acc_tolerance: float = 0.02) -> Dict[str, List[Gene]]:
    """Three representative circuits from a Pareto front (each is a genome):

    ``QNAS-bestF``     highest scalar fitness F (the weighted trade-off);
    ``QNAS-accurate``  highest validation accuracy (ties: fewer two-qubit gates);
    ``QNAS-compact``   fewest two-qubit gates among circuits within ``acc_tolerance`` of the best accuracy.
    """
    if not front:
        raise ValueError("empty Pareto front")
    # Picking the best trade-off, the most accurate, and the cheapest near-best circuit.
    best_f = max(front, key=lambda r: r["metrics"]["fitness"])
    accurate = max(front, key=lambda r: (r["metrics"]["val_acc"], -r["metrics"]["n_2q"], -r["metrics"]["depth"]))
    top = accurate["metrics"]["val_acc"]
    near = [r for r in front if r["metrics"]["val_acc"] >= top - acc_tolerance]
    compact = min(near, key=lambda r: (r["metrics"]["n_2q"], r["metrics"]["depth"], -r["metrics"]["val_acc"]))
    out: Dict[str, List[Gene]] = {}
    for name, rec in (("QNAS-bestF", best_f), ("QNAS-accurate", accurate), ("QNAS-compact", compact)):
        out[name] = list(rec["genes"])
    return out
