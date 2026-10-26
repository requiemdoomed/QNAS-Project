"""Command line interface:  ``python -m qnas <command> [options]``  (or ``qnas <command>`` after ``pip install -e .``)

Commands
--------
demo        quick end-to-end run (a few minutes) into ``runs/demo``
run-all     the whole pipeline: search -> baselines -> candidates -> final -> gradients -> noise -> report
search      evolutionary search only (add --resume to continue an interrupted run)
baselines | candidates | final | gradients | noise | report     individual stages of an existing run
info        versions / hardware of this environment
"""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/cli.py
# Purpose: exposing the pipeline as a command-line tool: python -m qnas <command>.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import sys

from .pipeline import TASKS, Run, RunConfig, run_all

# Naming the datasets that never need a download.
OFFLINE = ("digits", "breast_cancer")


def _add_config_args(p: argparse.ArgumentParser) -> None:
    # Grouping the options by topic so --help stays readable.
    g = p.add_argument_group("task")
    g.add_argument("--dataset", help="digits | breast_cancer | pneumoniamnist | breastmnist | any 2-class MedMNIST flag")
    g.add_argument("--task", choices=sorted(TASKS), help="binary task for --dataset digits (default: even-odd)")
    g.add_argument("--classes", nargs=2, metavar=("A", "B"),
                   help="custom digit groups, e.g. --classes 0,2,4,6,8 1,3,5,7,9  (or: --classes 3 5)")
    g.add_argument("--extractor", choices=("cnn", "pca"), help="classical feature extractor (default: cnn autoencoder)")
    g.add_argument("--data-root", help="folder with MedMNIST .npz files (default: data)")
    g.add_argument("--max-per-class", type=int, help="cap samples per class (MedMNIST) to speed things up")
    s = p.add_argument_group("search space")
    s.add_argument("--qubits", type=int, dest="n_qubits", help="number of qubits = number of classical features (default 4)")
    s.add_argument("--max-depth", type=int)
    s.add_argument("--max-genes", type=int)
    s.add_argument("--connectivity", choices=("all", "linear", "ring"), help="allowed two-qubit gate wiring")
    s.add_argument("--data-reuploading", action="store_true", default=None, help="allow rotations whose angle is an input feature")
    e = p.add_argument_group("evolution")
    e.add_argument("--mode", choices=("nsga2", "weighted"), help="multi-objective NSGA-II (default) or weighted-fitness GA")
    e.add_argument("--objectives", nargs="+", choices=("acc", "depth", "n2q", "params"))
    e.add_argument("--pop", type=int, dest="pop_size")
    e.add_argument("--generations", type=int)
    e.add_argument("--epochs", type=int, help="training epochs per candidate during the search")
    e.add_argument("--cold-start", action="store_true", help="start from purely random circuits (default: seed generation 0 with SEL/BEL templates)")
    e.add_argument("--alpha", type=float); e.add_argument("--beta", type=float); e.add_argument("--gamma", type=float)
    f = p.add_argument_group("final evaluation / analyses")
    f.add_argument("--final-epochs", type=int)
    f.add_argument("--final-seeds", type=int)
    f.add_argument("--no-prune", action="store_true", help="skip ablation pruning of the selected circuits")
    f.add_argument("--no-noise", action="store_true", help="skip the noisy-hardware study")
    m = p.add_argument_group("misc")
    m.add_argument("--seed", type=int)
    m.add_argument("--workers", type=int, help="parallel worker processes (default 1)")
    m.add_argument("--quiet", action="store_true")


def _parse_classes(vals):
    # Turning "0,2,4" into [0, 2, 4] for each of the two class groups.
    def grp(s):
        return [int(x) for x in s.split(",")]
    return [grp(vals[0]), grp(vals[1])]


def _make_cfg(args, preset: str) -> RunConfig:
    # Copying every option the user set onto the chosen preset.
    over = {k: getattr(args, k, None) for k in (
        "dataset", "extractor", "data_root", "max_per_class", "n_qubits", "max_depth", "max_genes", "connectivity",
        "mode", "objectives", "pop_size", "generations", "epochs", "alpha", "beta", "gamma", "final_epochs",
        "final_seeds", "seed", "workers")}
    cfg = RunConfig.preset(preset, **over)
    # Applying the flag-style options that do not map one-to-one onto config fields.
    if getattr(args, "task", None):
        cfg.classes = TASKS[args.task]
    if getattr(args, "classes", None):
        cfg.classes = _parse_classes(args.classes)
    if getattr(args, "data_reuploading", None):
        cfg.allow_data = True
    if getattr(args, "cold_start", False):
        cfg.warm_start = False
    if getattr(args, "no_prune", False):
        cfg.prune = False
    if getattr(args, "no_noise", False):
        cfg.noise = False
    # Falling back to PCA for tabular data and capping MedMNIST sizes to keep runs fast.
    if cfg.dataset == "breast_cancer" and cfg.extractor == "cnn":
        cfg.extractor = "pca"  # using PCA: tabular data has no images for a CNN
    if cfg.dataset not in OFFLINE and cfg.max_per_class is None:
        cfg.max_per_class = 600
    return cfg


def main(argv=None) -> int:
    # Declaring the sub-commands and their options.
    parser = argparse.ArgumentParser(prog="qnas", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("demo", help="quick end-to-end run (a few minutes)")
    p.add_argument("--out", default="runs/demo")
    _add_config_args(p)

    p = sub.add_parser("run-all", help="run the full pipeline")
    p.add_argument("--out", default="runs/standard")
    p.add_argument("--preset", choices=("quick", "standard", "thorough"), default="standard")
    p.add_argument("--resume", action="store_true", help="continue an interrupted search from its checkpoint")
    _add_config_args(p)

    p = sub.add_parser("search", help="evolutionary search only")
    p.add_argument("--out", default="runs/standard")
    p.add_argument("--preset", choices=("quick", "standard", "thorough"), default="standard")
    p.add_argument("--resume", action="store_true")
    _add_config_args(p)

    for name, helptext in (("baselines", "evaluate SEL/BEL templates with the search protocol"),
                           ("candidates", "pick + simplify/prune circuits from the Pareto front"),
                           ("final", "multi-seed retraining and test-set comparison"),
                           ("gradients", "barren-plateau scan and gradient statistics"),
                           ("noise", "accuracy under depolarizing noise"),
                           ("report", "build figures and results.md")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--out", required=True, help="existing run directory")
        p.add_argument("--workers", type=int)
        p.add_argument("--quiet", action="store_true")

    sub.add_parser("info", help="print versions and hardware info")

    args = parser.parse_args(argv)

    import torch

    # Using one thread per process; parallelism comes from --workers instead.
    torch.set_num_threads(1)

    # Printing library versions and hardware.
    if args.cmd == "info":
        import os
        import pennylane as qml
        import deap, sklearn, numpy

        print(f"python {sys.version.split()[0]} | pennylane {qml.__version__} | torch {torch.__version__} | deap {deap.__version__} "
              f"| sklearn {sklearn.__version__} | numpy {numpy.__version__} | cpus {os.cpu_count()}")
        return 0

    # Dispatching to the requested stage, turning a missing dataset into a friendly error.
    try:
        if args.cmd == "demo":
            run_all(args.out, _make_cfg(args, "quick"), quiet=args.quiet)
        elif args.cmd == "run-all":
            run_all(args.out, _make_cfg(args, args.preset), resume=args.resume, quiet=args.quiet)
        elif args.cmd == "search":
            Run(args.out, _make_cfg(args, args.preset), quiet=args.quiet).search(resume=args.resume)
        else:
            run = Run(args.out, quiet=args.quiet)
            if getattr(args, "workers", None):
                run.cfg.workers = args.workers
            getattr(run, args.cmd)()
    except Exception as exc:  # showing a friendly message for the common "dataset missing" case
        from .data import DatasetUnavailable

        if isinstance(exc, DatasetUnavailable):
            print(f"\nERROR: {exc}", file=sys.stderr)
            return 2
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
