"""Comparison reports for the TSP local-search heuristics.

Runs Nearest-Neighbour (construction) and the local-search improvers (2-opt,
3-opt, Lin-Kernighan) on one instance from a common NN start, and reports tour
length, gap to the best found, and runtime — the apples-to-apples comparison the
heuristics are meant for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import pairwise_distances

from ..core.types import AF, AI
from .base import CombinatoricsResult
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


def _warmup(distances: AF, backend: LocalSearchBackend) -> None:
    # Compile the (numba/cython) kernels on a tiny instance so the reported
    # runtimes are steady-state compute, not one-time JIT warm-up.
    n = min(6, distances.shape[0])
    d = np.ascontiguousarray(distances[:n, :n])
    seed = NearestNeighborTSP(
        NearestNeighborTSPConfig(name="w"), network_routes=d.copy()
    ).solve()
    kw = dict(
        initial_route=seed.optimal_path.copy(),
        initial_value=seed.optimal_value,
        network_routes=d.copy(),
    )
    TwoOptTSP(TwoOptTSPConfig(name="w", local_search_backend=backend), **kw).solve()
    ThreeOptTSP(
        TwoOptTSPConfig(name="w", num_iterations=1, local_search_backend=backend), **kw
    ).solve()
    LinKernighanTSP(LinKernighanTSPConfig(name="w"), **kw).solve()


def compare_tsp_heuristics(
    network_routes: AF | None = None,
    city_locations: AF | None = None,
    back_to_start: bool = True,
    three_opt_iterations: int = 5,
    three_opt_neighbors: int = 10,
    candidate_k: int = 8,
    backend: LocalSearchBackend = "numba",
    warmup: bool = True,
) -> list[HeuristicResult]:
    """Run NN / 2-opt / 3-opt / LK on one instance and return ranked results.

    The three improvers start from the *same* nearest-neighbour tour, so the
    comparison isolates the local search. Times exclude JIT warm-up (see
    ``warmup``). ``gap_pct`` is relative to the shortest tour found.
    """
    distances = _resolve_distances(network_routes, city_locations)
    if warmup:
        _warmup(distances, backend)

    records: list[tuple[str, CombinatoricsResult, float]] = []

    t0 = time.perf_counter()
    nn = NearestNeighborTSP(
        NearestNeighborTSPConfig(name="nn", back_to_start=back_to_start),
        network_routes=distances.copy(),
    ).solve()
    records.append(("Nearest Neighbor", nn, time.perf_counter() - t0))

    seed_route, seed_val = nn.optimal_path, nn.optimal_value

    def _timed(cls, config) -> tuple[CombinatoricsResult, float]:
        t = time.perf_counter()
        result = cls(
            config,
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
            name="lk", back_to_start=back_to_start, candidate_k=candidate_k
        ),
    )
    records.append(("Lin-Kernighan", r, dt))

    best = min(float(res.optimal_value) for _, res, _ in records)
    return [
        HeuristicResult(
            name=name,
            tour_length=float(res.optimal_value),
            runtime_s=dt,
            gap_pct=(
                100.0 * (float(res.optimal_value) - best) / best if best > 0 else 0.0
            ),
            optimal_path=res.optimal_path,
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
