"""Lin-Kernighan correctness + TSP local-search comparison reports."""

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from optimizers.combinatorial.strategy import (
    NearestNeighborTSP,
    NearestNeighborTSPConfig,
    LinKernighanTSP,
    LinKernighanTSPConfig,
    candidate_lists,
)
from optimizers.combinatorial.compare import (
    compare_tsp_heuristics,
    format_comparison_table,
    HeuristicResult,
)


def _cities(n, seed):
    return np.random.RandomState(seed).uniform(0, 100, size=(n, 2))


def _is_valid_tour(path, n):
    cyc = path[:-1] if path[0] == path[-1] else path
    return len(cyc) == n and sorted(cyc.tolist()) == list(range(n))


# --------------------------- candidate lists ---------------------------


def test_candidate_lists_shape_and_excludes_self():
    D = pairwise_distances(_cities(20, 0))
    cand = candidate_lists(D, 5)
    assert cand.shape == (20, 5)
    for i in range(20):
        assert i not in cand[i]  # self is never its own candidate
        # candidates are ordered nearest-first
        dists = D[i, cand[i]]
        assert np.all(np.diff(dists) >= 0)


# --------------------------- Lin-Kernighan ---------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_lin_kernighan_valid_and_improves(seed):
    D = pairwise_distances(_cities(100, seed))
    nn = NearestNeighborTSP(
        NearestNeighborTSPConfig(name="nn"), network_routes=D.copy()
    ).solve()
    lk = LinKernighanTSP(
        LinKernighanTSPConfig(name="lk"),
        initial_route=nn.optimal_path.copy(),
        initial_value=nn.optimal_value,
        network_routes=D.copy(),
    ).solve()
    assert _is_valid_tour(lk.optimal_path, 100)
    assert lk.optimal_value <= nn.optimal_value  # never worse than the NN start


def test_lin_kernighan_from_scratch_builds_nn_start():
    # No initial_route → setup_local_search runs NN internally.
    D = pairwise_distances(_cities(60, 7))
    lk = LinKernighanTSP(
        LinKernighanTSPConfig(name="lk"), network_routes=D.copy()
    ).solve()
    assert _is_valid_tour(lk.optimal_path, 60)


@pytest.mark.parametrize("seed", range(8))
def test_lin_kernighan_reported_length_matches_tour(seed):
    # Regression: Or-opt can relocate the depot (city 0) off index 0. The
    # reported optimal_value must equal the true closed-tour length recomputed
    # independently — not overshoot by a spurious return-to-depot edge. (Seeds
    # 0 and 6 at N=40 are cases where the depot actually moves.)
    D = pairwise_distances(_cities(40, seed))
    lk = LinKernighanTSP(
        LinKernighanTSPConfig(name="lk"), network_routes=D.copy()
    ).solve()
    path = lk.optimal_path
    # Depot-first, consistent with the 2-opt/3-opt solvers.
    assert path[0] == 0
    cyc = path[:-1] if path[0] == path[-1] else path
    closed = np.append(cyc, cyc[0])
    true_len = float(D[closed[:-1], closed[1:]].sum())
    assert np.isclose(lk.optimal_value, true_len)


# --------------------------- comparison report ---------------------------


def test_compare_report_structure_and_validity():
    res = compare_tsp_heuristics(city_locations=_cities(120, 5))
    assert [r.name for r in res] == [
        "Nearest Neighbor",
        "2-opt",
        "3-opt",
        "Lin-Kernighan",
    ]
    assert all(isinstance(r, HeuristicResult) for r in res)
    for r in res:
        assert _is_valid_tour(r.optimal_path, 120)
        assert r.runtime_s >= 0.0
        assert r.gap_pct >= 0.0
    # exactly one heuristic is the best (gap 0)
    assert sum(1 for r in res if r.gap_pct == 0.0) >= 1
    assert min(r.gap_pct for r in res) == 0.0


def test_compare_accepts_distance_matrix_directly():
    D = pairwise_distances(_cities(80, 9))
    res = compare_tsp_heuristics(network_routes=D)
    assert len(res) == 4
    assert all(_is_valid_tour(r.optimal_path, 80) for r in res)


def test_lk_best_on_average(capsys):
    """LK is candidate-restricted (not a strict superset of full 2-opt), so it
    wins on average rather than on every single instance."""
    lk_lengths, twoopt_lengths = [], []
    for seed in range(5):
        res = {
            r.name: r for r in compare_tsp_heuristics(city_locations=_cities(120, seed))
        }
        lk_lengths.append(res["Lin-Kernighan"].tour_length)
        twoopt_lengths.append(res["2-opt"].tour_length)
    with capsys.disabled():
        print(
            f"\n[LK vs 2-opt, N=120, 5 seeds] "
            f"mean LK={np.mean(lk_lengths):.1f}  mean 2-opt={np.mean(twoopt_lengths):.1f}"
        )
    assert np.mean(lk_lengths) <= np.mean(twoopt_lengths)


def test_comparison_table_report(capsys):
    with capsys.disabled():
        for seed in (0, 2):
            res = compare_tsp_heuristics(city_locations=_cities(150, seed))
            print(f"\n=== TSP heuristic comparison (N=150, seed={seed}) ===")
            print(format_comparison_table(res))
    # table lists every heuristic
    table = format_comparison_table(
        compare_tsp_heuristics(city_locations=_cities(100, 1))
    )
    for name in ("Nearest Neighbor", "2-opt", "3-opt", "Lin-Kernighan"):
        assert name in table
