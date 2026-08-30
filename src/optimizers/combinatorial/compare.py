"""Comparison reports for the TSP local-search heuristics.

Runs Nearest-Neighbour (construction) and the local-search improvers (2-opt,
3-opt, Lin-Kernighan) on one instance from a common NN start, and reports tour
length, gap to the best found, and runtime — the apples-to-apples comparison the
heuristics are meant for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from sklearn.metrics import pairwise_distances

from ..core.types import AF, AI
from ..core.base import OptimizerResult
from .strategy import (
    NearestNeighborTSP,
    NearestNeighborTSPConfig,
    TwoOptTSP,
    ThreeOptTSP,
    TwoOptTSPConfig,
    LinKernighanTSP,
    LinKernighanTSPConfig,
    LocalSearchBackend,
)


@dataclass
class HeuristicResult:
    name: str
    tour_length: float
    runtime_s: float
    gap_pct: float  # relative to the best tour length in the comparison
    optimal_path: AI


def _resolve_distances(network_routes: AF | None, city_locations: AF | None) -> AF:
    if network_routes is not None:
        return np.ascontiguousarray(network_routes, dtype=np.float64)
    if city_locations is not None:
        return np.ascontiguousarray(
            pairwise_distances(city_locations), dtype=np.float64
        )
    raise ValueError("provide either network_routes or city_locations")


def compare_tsp_heuristics(
    network_routes: AF | None = None,
    city_locations: AF | None = None,
    back_to_start: bool = True,
    three_opt_iterations: int = 5,
    three_opt_neighbors: int = 10,
    candidate_k: int = 8,
    backend: LocalSearchBackend = "cython",
) -> list[HeuristicResult]:
    """Run NN / 2-opt / 3-opt / LK on one instance and return ranked results.

    The three improvers start from the *same* nearest-neighbour tour, so the
    comparison isolates the local search. ``gap_pct`` is relative to the
    shortest tour found.
    """
    distances = _resolve_distances(network_routes, city_locations)

    records: list[tuple[str, OptimizerResult, float]] = []

    t0 = time.perf_counter()
    nn = NearestNeighborTSP(
        config=NearestNeighborTSPConfig(name="nn", back_to_start=back_to_start),
        network_routes=distances.copy(),
    ).solve()
    records.append(("Nearest Neighbor", nn, time.perf_counter() - t0))

    seed_route, seed_val = nn.solution_vector, nn.solution_score

    def _timed(cls: Any, config: Any) -> tuple[OptimizerResult, float]:
        t = time.perf_counter()
        result = cls(
            config=config,
            initial_route=seed_route.copy(),
            initial_value=seed_val,
            network_routes=distances.copy(),
        ).solve()
        return result, time.perf_counter() - t

    r, dt = _timed(
        TwoOptTSP,
        TwoOptTSPConfig(
            name="2opt", back_to_start=back_to_start, local_search_backend=backend
        ),
    )
    records.append(("2-opt", r, dt))

    r, dt = _timed(
        ThreeOptTSP,
        TwoOptTSPConfig(
            name="3opt",
            back_to_start=back_to_start,
            num_iterations=three_opt_iterations,
            nearest_neighbors=three_opt_neighbors,
            local_search_backend=backend,
        ),
    )
    records.append(("3-opt", r, dt))

    r, dt = _timed(
        LinKernighanTSP,
        LinKernighanTSPConfig(
            name="lk",
            back_to_start=back_to_start,
            candidate_k=candidate_k,
            local_search_backend=backend,
        ),
    )
    records.append(("Lin-Kernighan", r, dt))

    best = min(float(res.solution_score) for _, res, _ in records)
    return [
        HeuristicResult(
            name=name,
            tour_length=float(res.solution_score),
            runtime_s=dt,
            gap_pct=(
                100.0 * (float(res.solution_score) - best) / best if best > 0 else 0.0
            ),
            # These heuristics all return a single tour (never a list of tours).
            optimal_path=cast(AI, res.solution_vector),
        )
        for name, res, dt in records
    ]


def format_comparison_table(results: list[HeuristicResult]) -> str:
    """Render comparison results as a fixed-width text table."""
    header = f"{'heuristic':<18}{'length':>12}{'gap %':>9}{'time (ms)':>12}"
    lines = [header, "-" * len(header)]
    for r in sorted(results, key=lambda x: x.tour_length):
        lines.append(
            f"{r.name:<18}{r.tour_length:>12.2f}{r.gap_pct:>9.2f}"
            f"{r.runtime_s * 1e3:>12.2f}"
        )
    return "\n".join(lines)
