"""Multi-objective utilities (all vectors are in *minimisation* form).

* ``dominates(a, b)``       - a is no worse everywhere and strictly better somewhere
* ``non_dominated(points)`` - indices of the Pareto-optimal points
* ``hypervolume(points, ref)`` - exact volume dominated by the set and bounded by ``ref``
  (the standard quality indicator of a Pareto front - bigger is better)
"""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/pareto.py
# Purpose: small multi-objective helpers: dominance, the non-dominated set and
#          an exact hypervolume for measuring how good a front is.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import List, Sequence


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    # Requiring "no worse in every objective" and "better in at least one".
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def non_dominated(points: Sequence[Sequence[float]]) -> List[int]:
    """Indices of points not dominated by any other (duplicates are all kept)."""
    # Comparing every point with every other one (O(n^2), fine for small fronts).
    keep = []
    for i, p in enumerate(points):
        if not any(dominates(q, p) for j, q in enumerate(points) if j != i):
            keep.append(i)
    return keep


def _hv(points: List[Sequence[float]], ref: Sequence[float]) -> float:
    # Recursing on dimension: one objective is just a length.
    d = len(ref)
    if d == 1:
        return ref[0] - min(p[0] for p in points)
    pts = sorted(points, key=lambda p: p[-1])  # sweeping along the last objective
    # Slicing along the last objective and adding slab height x lower-dimensional volume.
    volume = 0.0
    for i, p in enumerate(pts):
        upper = pts[i + 1][-1] if i + 1 < len(pts) else ref[-1]
        slab = upper - p[-1]
        if slab > 0:
            volume += slab * _hv([q[:-1] for q in pts[: i + 1]], ref[:-1])
    return volume


def hypervolume(points: Sequence[Sequence[float]], ref: Sequence[float]) -> float:
    """Exact hypervolume (slicing algorithm; fine for the small fronts we have: <= ~100 points)."""
    # Ignoring points that do not lie strictly inside the reference box.
    pts = [tuple(p) for p in points if all(pi < ri for pi, ri in zip(p, ref))]
    if not pts:
        return 0.0
    return _hv(pts, list(ref))
