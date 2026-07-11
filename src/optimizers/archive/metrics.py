"""Quality-diversity & multi-objective metrics + the run report (QD_PARETO_PLAN.md §4.5).

All objectives follow the library's **minimization** convention (smaller is
better), matching ``solution_value`` ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.types import AF


def non_dominated_mask(objectives: AF) -> np.ndarray:
    """Boolean mask of Pareto-non-dominated rows (minimization).

    Row ``i`` is dominated by ``j`` if ``j`` is ``<=`` on every objective and
    ``<`` on at least one. O(n^2 * m); fine for archive-sized inputs.
    """
    f = np.atleast_2d(np.asarray(objectives, dtype=float))
    n = f.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # j dominates i?  (all <= and any <)
        le = np.all(f <= f[i], axis=1)
        lt = np.any(f < f[i], axis=1)
        dominated_by = le & lt
        dominated_by[i] = False
        if np.any(dominated_by):
            keep[i] = False
    return keep


def pareto_front(objectives: AF) -> np.ndarray:
    """Indices of the non-dominated rows (the Pareto front), best-spread first."""
    mask = non_dominated_mask(objectives)
    idx = np.flatnonzero(mask)
    f = np.atleast_2d(np.asarray(objectives, dtype=float))
    return idx[np.argsort(f[idx, 0])]  # order along the first objective for plotting


def hypervolume(objectives: AF, reference: AF) -> float:
    """Dominated hypervolume relative to a (worst-case) ``reference`` point.

    Exact for 1-2 objectives (sweep); Monte-Carlo estimate for >=3. Minimization:
    the volume of the box ``[front, reference]`` dominated by the front.
    """
    f = np.atleast_2d(np.asarray(objectives, dtype=float))
    ref = np.asarray(reference, dtype=float)
    f = f[non_dominated_mask(f)]
    if f.size == 0:
        return 0.0
    m = f.shape[1]
    if m == 1:
        return float(max(0.0, ref[0] - f[:, 0].min()))
    if m == 2:
        fs = f[np.argsort(f[:, 0])]  # ascending first objective
        hv = 0.0
        cur_ref_x = ref[0]
        # Staircase sweep from the largest x: each point contributes the rectangle
        # from itself to the running right-edge, at height (ref_y - y).
        for x, y in fs[::-1]:
            hv += max(0.0, cur_ref_x - x) * max(0.0, ref[1] - y)
            cur_ref_x = x
        return float(hv)
    # m >= 3: Monte-Carlo estimate of the dominated volume in [ideal, ref].
    ideal = f.min(axis=0)
    box = np.maximum(ref - ideal, 0.0)
    vol = float(np.prod(box))
    if vol <= 0.0:
        return 0.0
    rng = np.random.default_rng(0)
    n_samples = 20000
    pts = ideal + rng.random((n_samples, m)) * box
    dominated = np.zeros(n_samples, dtype=bool)
    for row in f:
        dominated |= np.all(row <= pts, axis=1)
    return float(vol * dominated.mean())


def qd_score(values: AF, reference: float | None = None) -> float:
    """QD-score: total quality banked across occupied cells (higher = better).

    For minimization we credit each elite ``reference - value`` (clamped at 0).
    ``reference`` defaults to the worst elite value, so an empty/near-empty
    archive scores ~0 and a full archive of good elites scores high.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 0.0
    ref = float(v.max()) if reference is None else float(reference)
    return float(np.sum(np.maximum(0.0, ref - v)))


@dataclass
class QDReport:
    """Summary of a finished quality-diversity run (see ``IOptimizer.qd_report``)."""

    num_elites: int
    best_fitness: float
    coverage: float | None  # fraction of cells filled (MAP-Elites), else None
    qd_score: float
    # Pareto report over the tracked outputs (None when no outputs were tracked):
    pareto_solutions: AF | None = None  # decision vectors on the front
    pareto_objectives: AF | None = None  # their objective vectors
    all_objectives: AF | None = None  # every elite's objective vector
    hypervolume: float | None = None

    def __repr__(self) -> str:
        cov = f"{self.coverage:.2f}" if self.coverage is not None else "n/a"
        front = 0 if self.pareto_objectives is None else len(self.pareto_objectives)
        return (
            f"QDReport(elites={self.num_elites}, best={self.best_fitness:.4g}, "
            f"coverage={cov}, qd_score={self.qd_score:.4g}, pareto_pts={front})"
        )
