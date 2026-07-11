"""CVT-MAP-Elites archive (QD_PARETO_PLAN.md §4.1, §4.2).

A MAP-Elites archive whose cells are the Voronoi regions of ``n_cells`` centroids
(CVT-MAP-Elites, Vassiliades et al. 2018). Centroids are placed by k-means over a
sample of *reachable* descriptors (random decision vectors projected through the
descriptor), so the archive adapts to the descriptor cloud without needing a
priori bounds — the right default when descriptors come from a random projection
of a high-dimensional decision vector.

Each cell keeps the single best-**fitness** elite whose descriptor falls in it.
The archive exposes the same scalar surface as :class:`~optimizers.solution_deck.SolutionDeck`
(``solution_archive`` / ``solution_value`` sorted best-first, ``get_best``, ...),
so the existing solvers read it as an ordinary — but structurally diverse —
parent pool.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from ..core.types import AF, af64, f64, b8
from ..core.random import get_seed
from ..core.variables import InputVariables
from .descriptor import RandomProjectionDescriptor


class CVTArchive:
    """MAP-Elites archive with centroidal (CVT) cells; a drop-in ``Archive``."""

    def __init__(
        self,
        num_vars: int,
        lower: AF,
        upper: AF,
        descriptor_fn: Callable[[AF], AF],
        descriptor_dim: int,
        n_cells: int = 256,
        descriptor_source: str = "projection",
        init_samples: int = 0,
        centroid_samples: int = 0,
        seed: int | None = None,
        dtype: type = f64,
    ):
        self.num_vars = num_vars
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.descriptor_fn = descriptor_fn
        self.descriptor_dim = descriptor_dim
        self.descriptor_source = descriptor_source
        self.n_cells = n_cells
        self._dtype = dtype
        self._seed = get_seed() if seed is None else seed
        # A random batch per generation seeds many cells; scale init to the grid.
        self.init_samples = init_samples if init_samples > 0 else max(2 * n_cells, 200)

        # --- place centroids by k-means over reachable descriptors ---
        n_c_samples = (
            centroid_samples if centroid_samples > 0 else max(10 * n_cells, 5000)
        )
        rng = np.random.default_rng(self._seed)
        sample_x = rng.uniform(self.lower, self.upper, size=(n_c_samples, num_vars))
        sample_d = np.asarray(self.descriptor_fn(sample_x), dtype=float)
        km = KMeans(n_clusters=n_cells, n_init=3, random_state=self._seed)
        km.fit(sample_d)
        self.centroids = km.cluster_centers_
        self._kdtree = cKDTree(self.centroids)

        # --- per-cell elite storage (source of truth) ---
        self._cell_solution: af64 = np.full((n_cells, num_vars), np.nan, dtype=dtype)
        self._cell_value = np.full(n_cells, np.inf, dtype=f64)  # minimization
        self._cell_occupied = np.zeros(n_cells, dtype=bool)

        # --- scalar-surface views (rebuilt after each insertion) ---
        self.solution_archive: af64 = np.empty((0, num_vars), dtype=dtype)
        self.solution_value: af64 = np.empty((0,), dtype=f64)
        self.is_local_optima = np.empty((0,), dtype=b8)
        self.solution_outputs: af64 | None = None
        # ``archive_size`` kept for API parity (consumers read it); for a MAP-Elites
        # archive capacity is the number of cells, not an elitist top-k.
        self.archive_size = n_cells

    # ---- MAP-Elites insertion ----
    def _cells_for(self, solutions: AF, outputs: AF | None) -> AF:
        if self.descriptor_source == "outputs":
            assert outputs is not None, "descriptor_source='outputs' needs outputs"
            desc = np.asarray(self.descriptor_fn(outputs), dtype=float)
        else:
            desc = np.asarray(self.descriptor_fn(solutions), dtype=float)
        _, cells = self._kdtree.query(desc)
        return np.atleast_1d(cells)

    def add_generation(
        self,
        solutions: AF,
        values: AF,
        outputs: AF | None = None,
        local_optima: bool = False,
    ) -> None:
        """Insert a batch: each solution takes its cell iff it beats the incumbent."""
        solutions = np.atleast_2d(np.asarray(solutions, dtype=self._dtype))
        values = np.atleast_1d(np.asarray(values, dtype=f64))
        cells = self._cells_for(solutions, outputs)
        for i in range(solutions.shape[0]):
            c = int(cells[i])
            if (not self._cell_occupied[c]) or values[i] < self._cell_value[c]:
                self._cell_solution[c] = solutions[i]
                self._cell_value[c] = values[i]
                self._cell_occupied[c] = True
        self._rebuild_views()

    def _rebuild_views(self) -> None:
        occ = np.flatnonzero(self._cell_occupied)
        # Sorted best-first so rank-based consumers (ACO CDF, PSO g-best,
        # get_best) behave exactly as with the scalar deck.
        order = occ[np.argsort(self._cell_value[occ])]
        self.solution_archive = self._cell_solution[order]
        self.solution_value = self._cell_value[order]
        self.is_local_optima = np.zeros(order.shape[0], dtype=b8)

    # ---- initialization (MAP-Elites: evaluate a random batch, insert) ----
    def initialize_solution_deck(
        self,
        variables: InputVariables,
        eval_fcn: Callable[[AF], Any],
        preserve_percent: float = 0.0,
        init_type: str = "random",
    ) -> None:
        n = self.init_samples
        batch: af64 = np.empty((n, len(variables)), dtype=self._dtype)
        for d, v in enumerate(variables):
            batch[:, d] = v.initial_random_values(n)
        values = np.array([eval_fcn(batch[i]) for i in range(n)], dtype=f64)
        self.add_generation(batch, values)

    # ---- diverse parent sampling (uniform over occupied cells) ----
    def parents(self, n: int, rng: np.random.Generator) -> af64:
        occ = np.flatnonzero(self._cell_occupied)
        if occ.size == 0:
            raise RuntimeError("archive is empty; nothing to select parents from")
        idx = occ[rng.integers(0, occ.size, size=n)]
        return self._cell_solution[idx]

    @property
    def coverage(self) -> float:
        """Fraction of cells occupied — the MAP-Elites exploration measure."""
        return float(self._cell_occupied.sum()) / float(self.n_cells)

    def cell_data(self) -> tuple[af64, af64, AF]:
        """``(centroids, cell_value, occupied_mask)`` for plotting/inspection."""
        return self.centroids, self._cell_value, self._cell_occupied

    # ---- scalar-surface parity methods ----
    def sort(self) -> None:
        idx = np.argsort(self.solution_value)
        self.solution_archive = self.solution_archive[idx]
        self.solution_value = self.solution_value[idx]
        self.is_local_optima = self.is_local_optima[idx]

    def truncate(self, size: int = -1) -> None:
        # No-op: a MAP-Elites archive is bounded by its cell count, not an
        # elitist top-k. Kept for interface parity.
        return

    def __len__(self) -> int:
        return int(self._cell_occupied.sum())

    def get(self, idx: int) -> tuple[af64, f64, b8]:
        return (
            self.solution_archive[idx],
            self.solution_value[idx],
            self.is_local_optima[idx],
        )

    def get_best(self) -> tuple[af64, f64, b8]:
        self.sort()
        return self.get(0)

    # ---- serialization ----
    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "cvt",
            "num_vars": int(self.num_vars),
            "n_cells": int(self.n_cells),
            "descriptor_dim": int(self.descriptor_dim),
            "descriptor_source": self.descriptor_source,
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "centroids": self.centroids.tolist(),
            "cell_solution": self._cell_solution.tolist(),
            "cell_value": self._cell_value.tolist(),
            "cell_occupied": self._cell_occupied.astype(bool).tolist(),
            "projection_R": (
                self.descriptor_fn.R.tolist()
                if isinstance(self.descriptor_fn, RandomProjectionDescriptor)
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CVTArchive":
        num_vars = int(data["num_vars"])
        lower = np.asarray(data["lower"], dtype=float)
        upper = np.asarray(data["upper"], dtype=float)
        desc = RandomProjectionDescriptor(
            num_vars, int(data["descriptor_dim"]), lower, upper
        )
        if data.get("projection_R") is not None:
            desc.R = np.asarray(data["projection_R"], dtype=float)
        obj = cls.__new__(cls)
        obj.num_vars = num_vars
        obj.lower, obj.upper = lower, upper
        obj.descriptor_fn = desc
        obj.descriptor_dim = int(data["descriptor_dim"])
        obj.descriptor_source = data.get("descriptor_source", "projection")
        obj.n_cells = int(data["n_cells"])
        obj._dtype = f64
        obj._seed = get_seed()
        obj.init_samples = max(2 * obj.n_cells, 200)
        obj.centroids = np.asarray(data["centroids"], dtype=float)
        obj._kdtree = cKDTree(obj.centroids)
        obj._cell_solution = np.asarray(data["cell_solution"], dtype=f64)
        obj._cell_value = np.asarray(data["cell_value"], dtype=f64)
        obj._cell_occupied = np.asarray(data["cell_occupied"], dtype=bool)
        obj.solution_outputs = None
        obj.archive_size = obj.n_cells
        obj._rebuild_views()
        return obj
