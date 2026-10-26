# Changelog

All notable changes to QNAS, written by Swayam Shakya.

## 1.1.0 - 2026-10-26
- Commenting every module, test and example in a consistent style (each comment starts with an -ing verb).
- Adding a header with author and purpose to every source file.
- Caching the connectivity wire pairs (`space._wire_pairs`) instead of rebuilding them on every validity check.
- Closing the `history.csv` file handle in the report builder and removing an unused argument.
- Fixing the test-file name quoted in `templates.py` and the MLP description in `final.py`.
- Making example 2 print the same numbers on every run (fixed torch seed).
- Adding two tests (wire-pair counts, report number formatting): 35 tests in total.
- Re-running the full experiment, the quick demo and a cold-start search with this version; all
  results in `runs/` come from these runs.
- Adding LICENSE (MIT), CITATION.cff, `.gitignore` and a GitHub-ready README.

## 1.0.0 - 2026-10-14
- First complete version: genome and search space, PennyLane circuit compiler, hybrid PyTorch model,
  NSGA-II / weighted GA on DEAP with caching, checkpoints and parallel workers, SEL/BEL baselines,
  lossless simplification and ablation pruning, multi-seed final comparison with Welch t-tests,
  barren-plateau scan, depolarizing-noise study, figures and `results.md` report, 33 tests.
