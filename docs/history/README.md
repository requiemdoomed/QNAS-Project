# Development history

`v1.0_random_start_search_log.txt` is the log of the first full run (version 1.0, October 2026), which
started from purely random circuits with a smaller budget and search space (population 20, 30 generations,
depth limit 14, at most 32 gates). The search collapsed towards tiny circuits (best validation accuracy 69.6%).
After it, the limits were raised (depth 20, 48 gates), the budget was increased (population 30, 40 generations)
and generation 0 was seeded with the six SEL/BEL templates.

With those settings a purely random start no longer collapses: see `runs/cold_start` (version 1.1,
best validation accuracy 79.1%, Pareto front of 6 circuits, hypervolume 0.734 against 0.746 for the
warm-started run in `runs/standard`).
