"""Correctness + performance tests for the compiled TSP local-search kernels.

The Cython kernels must be *bit-identical* to the numba kernels (hard asserts).
The performance tests measure and print timings; their assertions are kept
robust (large margins / correctness) so they don't flake on CI, while
``pytest -s`` surfaces the actual numbers.

Skipped automatically when the extension hasn't been built
(``python setup.py build_ext --inplace``).
"""

import time

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from optimizers.combinatorial.strategy import (
    _two_opt_kernel,
    _three_opt_kernel,
    _lk_kernel,
    candidate_lists,
    TwoOptTSP,
    ThreeOptTSP,
    TwoOptTSPConfig,
    LinKernighanTSP,
    LinKernighanTSPConfig,
    NearestNeighborTSP,
    NearestNeighborTSPConfig,
    HAS_CYTHON,
)
from optimizers.combinatorial.base import check_path_distance as py_check_path_distance

pytestmark = pytest.mark.skipif(
    not HAS_CYTHON, reason="compiled _tsp_cython extension not built"
)

if HAS_CYTHON:
    from optimizers.combinatorial import _tsp_cython as cy


def _distances(n, seed=0):
    rng = np.random.RandomState(seed)
    return np.ascontiguousarray(pairwise_distances(rng.uniform(0, 100, size=(n, 2))))


def _route(n, back_to_start=True, seed=0):
    rng = np.random.RandomState(seed + 1)
    r = rng.permutation(n)
    return np.append(r, r[0]) if back_to_start else r


def _best_time(fn, reps=5):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _py_two_opt(D, route, back_to_start=True):
    """Reference pure-Python 2-opt (single full pass) — for the speed baseline."""
    r = route.copy()
    N = D.shape[0]
    improved = True
    while improved:
        improved = False
        start = -1 if back_to_start else 0
        for i in range(start, N - 2):
            for j in range(i + 2, N - 1):
                if (
                    D[r[i], r[i + 1]] + D[r[j], r[j + 1]]
                    > D[r[i], r[j]] + D[r[i + 1], r[j + 1]]
                ):
                    r[i + 1 : j + 1] = r[i + 1 : j + 1][::-1]
                    improved = True
    return r


# --------------------------- correctness / parity ---------------------------


@pytest.mark.parametrize("n", [30, 80, 150])
@pytest.mark.parametrize("back_to_start", [True, False])
def test_two_opt_matches_numba(n, back_to_start):
    D = _distances(n)
    route = _route(n, back_to_start)
    r_nb = route.copy()
    nm_nb = _two_opt_kernel(D, r_nb, -1, -1, back_to_start)
    r_cy, nm_cy = cy.two_opt(D, route.copy(), -1, -1, back_to_start)
    assert np.array_equal(r_nb, r_cy)
    assert bool(nm_nb) == bool(nm_cy)


@pytest.mark.parametrize("n", [30, 80, 150])
@pytest.mark.parametrize("nn", [-1, 8])
def test_three_opt_matches_numba(n, nn):
    D = _distances(n)
    route = _route(n, back_to_start=True)
    r_nb = route.copy()
    nm_nb = _three_opt_kernel(D, r_nb, 3, nn)
    r_cy, nm_cy = cy.three_opt(D, route.copy(), 3, nn)
    assert np.array_equal(r_nb, r_cy)
    assert bool(nm_nb) == bool(nm_cy)


def test_check_path_distance_matches_python():
    D = _distances(50)
    route = _route(50, back_to_start=True)
    assert np.isclose(
        cy.check_path_distance(D, route, True),
        py_check_path_distance(D, route, True),
    )


def test_batch_matches_per_row_singles():
    D = _distances(60)
    routes = np.stack([_route(60, True, seed=s) for s in range(16)])
    batch = cy.two_opt_batch(D, routes.copy(), -1, -1, True)
    singles = np.stack(
        [cy.two_opt(D, routes[i].copy(), -1, -1, True)[0] for i in range(16)]
    )
    assert np.array_equal(batch, singles)


def test_three_opt_batch_matches_singles():
    D = _distances(50)
    routes = np.stack([_route(50, True, seed=s) for s in range(12)])
    batch = cy.three_opt_batch(D, routes.copy(), 2, 8)
    singles = np.stack([cy.three_opt(D, routes[i].copy(), 2, 8)[0] for i in range(12)])
    assert np.array_equal(batch, singles)


@pytest.mark.parametrize(
    "solver_cls,extra",
    [
        (TwoOptTSP, {}),
        (ThreeOptTSP, {"num_iterations": 3, "nearest_neighbors": 8}),
    ],
)
def test_solver_backend_parity(solver_cls, extra):
    D = _distances(120, seed=3)
    nn = NearestNeighborTSP(
        NearestNeighborTSPConfig(name="nn"), network_routes=D.copy()
    ).solve()
    kw = dict(
        initial_route=nn.optimal_path.copy(),
        initial_value=nn.optimal_value,
        network_routes=D.copy(),
    )
    r_nb = solver_cls(
        TwoOptTSPConfig(name="x", local_search_backend="numba", **extra), **kw
    ).solve()
    r_cy = solver_cls(
        TwoOptTSPConfig(name="x", local_search_backend="cython", **extra), **kw
    ).solve()
    assert np.array_equal(r_nb.optimal_path, r_cy.optimal_path)
    assert np.isclose(r_nb.optimal_value, r_cy.optimal_value)


# ------------------------------ performance ------------------------------


def test_two_opt_far_faster_than_pure_python(capsys):
    """Robust speed floor: compiled 2-opt must crush a pure-Python 2-opt."""
    D = _distances(200)
    route = _route(200, back_to_start=True)
    t_py = _best_time(lambda: _py_two_opt(D, route), reps=2)
    t_cy = _best_time(lambda: cy.two_opt(D, route.copy(), -1, -1, True), reps=5)
    with capsys.disabled():
        print(
            f"\n[2-opt N=200] pure-python={t_py*1e3:.1f}ms  cython={t_cy*1e3:.2f}ms"
            f"  speedup={t_py/t_cy:.0f}x"
        )
    assert t_cy < t_py / 10.0  # expect ~100x+; 10x is a safe floor


def test_two_opt_cython_vs_numba_benchmark(capsys):
    """Report cython vs (warm) numba across N; assert parity, not a hard ratio."""
    # warm the numba kernel so we time steady-state, not JIT compile
    Dw = _distances(20)
    _two_opt_kernel(Dw, np.arange(20), -1, -1, True)
    with capsys.disabled():
        print("\n[2-opt cython vs warm numba]")
        for n in (200, 500, 1000):
            D = _distances(n)
            base = _route(n, True)
            t_nb = _best_time(lambda: _two_opt_kernel(D, base.copy(), -1, -1, True))
            t_cy = _best_time(lambda: cy.two_opt(D, base.copy(), -1, -1, True))
            r_nb = base.copy()
            _two_opt_kernel(D, r_nb, -1, -1, True)
            r_cy = cy.two_opt(D, base.copy(), -1, -1, True)[0]
            assert np.array_equal(r_nb, r_cy)  # parity is the hard guarantee
            print(
                f"  N={n:4d}  numba={t_nb*1e3:7.2f}ms  cython={t_cy*1e3:7.2f}ms"
                f"  speedup={t_nb/t_cy:.2f}x"
            )


def test_batch_parallel_benchmark(capsys):
    """Parallel batch must match per-row results; report the wall-clock speedup."""
    D = _distances(400)
    routes = np.stack([_route(400, True, seed=s) for s in range(64)])
    t_seq = _best_time(
        lambda: [cy.two_opt(D, routes[i].copy(), -1, -1, True) for i in range(64)],
        reps=2,
    )
    t_batch = _best_time(
        lambda: cy.two_opt_batch(D, routes.copy(), -1, -1, True), reps=2
    )
    batch = cy.two_opt_batch(D, routes.copy(), -1, -1, True)
    singles = np.stack(
        [cy.two_opt(D, routes[i].copy(), -1, -1, True)[0] for i in range(64)]
    )
    assert np.array_equal(batch, singles)
    with capsys.disabled():
        print(
            f"\n[batch 2-opt 64x N=400] sequential={t_seq*1e3:.1f}ms  "
            f"parallel={t_batch*1e3:.1f}ms  speedup={t_seq/t_batch:.2f}x"
        )
    # never slower than ~1.5x the sequential time (catches a broken parallel build)
    assert t_batch < t_seq * 1.5


# ------------------------------ Lin-Kernighan ------------------------------


def _tour(n, seed=0):
    return np.random.RandomState(seed + 1).permutation(n).astype(np.intp)


@pytest.mark.parametrize("n", [60, 120, 200])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lin_kernighan_matches_numba(n, seed):
    D = _distances(n, seed)
    cand = candidate_lists(D, 8)
    tour = _tour(n, seed)
    r_nb = tour.copy().astype(np.int64)
    m_nb = _lk_kernel(D, r_nb, cand, 1000)
    r_cy, m_cy = cy.lin_kernighan(D, tour.copy(), cand, 1000)
    assert np.array_equal(r_nb, r_cy)  # bit-identical to the numba kernel
    assert m_nb == m_cy


def test_lk_batch_matches_singles():
    D = _distances(120, 0)
    cand = candidate_lists(D, 8)
    tours = np.stack([_tour(120, s) for s in range(16)])
    batch = cy.lin_kernighan_batch(D, tours.copy(), cand, 1000)
    singles = np.stack(
        [cy.lin_kernighan(D, tours[i].copy(), cand, 1000)[0] for i in range(16)]
    )
    assert np.array_equal(batch, singles)


def test_lk_solver_backend_parity():
    D = _distances(150, seed=3)
    nn = NearestNeighborTSP(
        NearestNeighborTSPConfig(name="nn"), network_routes=D.copy()
    ).solve()
    kw = dict(
        initial_route=nn.optimal_path.copy(),
        initial_value=nn.optimal_value,
        network_routes=D.copy(),
    )
    r_nb = LinKernighanTSP(
        LinKernighanTSPConfig(name="lk", local_search_backend="numba"), **kw
    ).solve()
    r_cy = LinKernighanTSP(
        LinKernighanTSPConfig(name="lk", local_search_backend="cython"), **kw
    ).solve()
    assert np.array_equal(r_nb.optimal_path, r_cy.optimal_path)
    assert np.isclose(r_nb.optimal_value, r_cy.optimal_value)


def test_lk_cython_vs_numba_benchmark(capsys):
    Dw = _distances(20)
    _lk_kernel(Dw, np.arange(20).astype(np.int64), candidate_lists(Dw, 3), 1000)
    with capsys.disabled():
        print("\n[LK cython vs warm numba]")
        for n in (200, 500, 1000):
            D = _distances(n)
            cand = candidate_lists(D, 8)
            base = _tour(n)
            t_nb = _best_time(
                lambda: _lk_kernel(D, base.copy().astype(np.int64), cand, 1000)
            )
            t_cy = _best_time(lambda: cy.lin_kernighan(D, base.copy(), cand, 1000))
            r_nb = base.copy().astype(np.int64)
            _lk_kernel(D, r_nb, cand, 1000)
            r_cy = cy.lin_kernighan(D, base.copy(), cand, 1000)[0]
            assert np.array_equal(r_nb, r_cy)  # parity is the hard guarantee
            print(
                f"  N={n:4d}  numba={t_nb*1e3:8.2f}ms  cython={t_cy*1e3:8.2f}ms"
                f"  speedup={t_nb/t_cy:.2f}x"
            )


def test_lk_batch_parallel_benchmark(capsys):
    D = _distances(300)
    cand = candidate_lists(D, 8)
    tours = np.stack([_tour(300, s) for s in range(64)])
    t_seq = _best_time(
        lambda: [cy.lin_kernighan(D, tours[i].copy(), cand, 1000) for i in range(64)],
        reps=2,
    )
    t_batch = _best_time(
        lambda: cy.lin_kernighan_batch(D, tours.copy(), cand, 1000), reps=2
    )
    batch = cy.lin_kernighan_batch(D, tours.copy(), cand, 1000)
    singles = np.stack(
        [cy.lin_kernighan(D, tours[i].copy(), cand, 1000)[0] for i in range(64)]
    )
    assert np.array_equal(batch, singles)
    with capsys.disabled():
        print(
            f"\n[LK batch 64x N=300] sequential={t_seq*1e3:.1f}ms  "
            f"parallel={t_batch*1e3:.1f}ms  speedup={t_seq/t_batch:.2f}x"
        )
    assert t_batch < t_seq * 1.5  # never meaningfully slower than sequential
