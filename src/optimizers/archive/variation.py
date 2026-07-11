"""Variation operators for the quality-diversity add-on (QD_PARETO_PLAN.md §4.3).

The headline operator is **Iso+LineDD** (Vassiliades & Mouret, GECCO 2018): a child
is a parent plus isotropic Gaussian noise plus a *directional* step along the line
to a second elite. The directional term exploits the empirical correlation between
elites ("elite hypervolume") and is what makes recombination across a diverse
archive explore so much more effectively than isotropic mutation alone — the more
so as the decision space grows, where isotropic mutation degrades.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from ..core.types import AF


def iso_line_dd(
    parents_a: AF,
    parents_b: AF,
    iso_sigma: float,
    line_sigma: float,
    lower: AF,
    upper: AF,
    rng: Generator,
) -> AF:
    """Batched Iso+LineDD variation.

    For each row: ``child = a + iso_sigma * N(0, I) + line_sigma * (b - a) * N(0, 1)``
    where the first Gaussian is per-dimension and the second (line) Gaussian is a
    single scalar per pair. Children are clipped to ``[lower, upper]``.

    Parameters
    ----------
    parents_a, parents_b:
        ``(n, num_vars)`` arrays of parent decision vectors (drawn from the
        archive; ``a`` is the base, ``b`` gives the direction).
    iso_sigma:
        Std-dev of the isotropic component, as a *fraction of each variable's
        domain* (so it is scale-aware across heterogeneous variables).
    line_sigma:
        Std-dev of the scalar directional component (dimensionless).
    lower, upper:
        ``(num_vars,)`` bounds used to scale the isotropic noise and clip.
    """
    a = np.asarray(parents_a, dtype=float)
    b = np.asarray(parents_b, dtype=float)
    n, n_vars = a.shape
    domain = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    # Per-dimension isotropic noise, scaled by each variable's domain.
    iso = rng.standard_normal((n, n_vars)) * (iso_sigma * domain)[None, :]
    # One directional scalar per pair, applied along (b - a).
    line = rng.standard_normal((n, 1)) * line_sigma
    children = a + iso + line * (b - a)
    return np.clip(children, lower, upper)


def iso_line_offspring(
    archive: AF,
    n: int,
    iso_sigma: float,
    line_sigma: float,
    lower: AF,
    upper: AF,
    rng: Generator,
    tournament_k: int = 3,
) -> AF:
    """Generate ``n`` Iso+LineDD children from a **best-first sorted** archive.

    Solver-agnostic so GA/ACO/PSO produce offspring identically in map-elites
    ``qd_variation="iso_line"`` mode (fair for cross-solver comparison). The
    *base* parent is a rank tournament — the min index of ``tournament_k`` uniform
    picks, which favours the front because the archive is sorted best-first, so it
    adds convergence pressure without needing the fitness values. The *direction*
    parent is uniform over the whole archive (diversity). See QD_PARETO_PLAN.md
    §4.3.
    """
    archive = np.asarray(archive, dtype=float)
    N = archive.shape[0]
    k = min(tournament_k, N)
    base_idx = np.min(rng.integers(0, N, size=(n, k)), axis=1)  # rank tournament
    dir_idx = rng.integers(0, N, size=n)  # uniform direction
    return iso_line_dd(
        archive[base_idx], archive[dir_idx], iso_sigma, line_sigma, lower, upper, rng
    )
