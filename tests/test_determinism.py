"""A seeded run must be reproducible, including in parallel.

``set_seed(s)`` is the promise that a run is a pure function of ``s``. It was not
kept at ``n_jobs > 1``: every parallel task drew from the one shared
``numpy.random.Generator`` returned by ``core.random.rng()``, which is not
thread-safe, so the numbers a task received depended on when the scheduler got
round to it. With a *pure deterministic* goal function and the same seed, five
identical GA runs made 1294, 1272, 1293, 1273 and 1260 evaluations and reached
five different optima.

Each parallel task now gets its own stream, spawned in the parent from the seed
and keyed by task index (``core.random.spawn_streams`` /
``core.parallel.GenerationRunner.run``).

The guarantee these tests pin is reproducibility *at a given* ``n_jobs``, not
across values of it. The population is split ``population_size // n_jobs`` ways,
so the worker count changes how the search is partitioned and therefore where it
goes -- a property of data-parallel population search, not of the seeding.
"""

import io
import contextlib

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from optimizers.combinatorial.aco import AntColonyTSP, AntColonyTSPConfig
from optimizers.combinatorial.aco_mst import AntColonyMST
from optimizers.combinatorial.ga import GeneticAlgorithmTSP, GeneticAlgorithmTSPConfig
from optimizers.continuous.aco import AntColonyOptimizer, AntColonyOptimizerConfig
from optimizers.continuous.ga import (
    GeneticAlgorithmOptimizer,
    GeneticAlgorithmOptimizerConfig,
)
from optimizers.continuous.pso import (
    ParticleSwarmOptimizer,
    ParticleSwarmOptimizerConfig,
)
from optimizers.combinatorial.mtsp import AntColonyMTSPConfig, AntColonyMTSP
from optimizers.continuous.variables import InputContinuousVariable
from optimizers.core.random import (
    rng,
    set_seed,
    spawn_stream_roots,
    spawn_streams,
    use_stream,
    use_stream_root,
)

OPTIMIZERS = {
    "ga": (GeneticAlgorithmOptimizer, GeneticAlgorithmOptimizerConfig),
    "pso": (ParticleSwarmOptimizer, ParticleSwarmOptimizerConfig),
    "aco": (AntColonyOptimizer, AntColonyOptimizerConfig),
}


def sphere(x):
    """A pure function of x. Any variation in the result is the optimizer's."""
    return float(np.sum((np.asarray(x, dtype=float) - 0.3) ** 2))


def solve_once(kind, *, seed=0, n_jobs=1, prefer="threads", local_grad_optim="none"):
    """One seeded run. Returns ``(best score, evaluation count)``."""
    optimizer_cls, config_cls = OPTIMIZERS[kind]
    set_seed(seed)

    evaluations = []

    def goal(x):
        evaluations.append(1)
        return sphere(x)

    variables = [InputContinuousVariable(f"p{i}", -1.0, 1.0) for i in range(4)]
    config = config_cls(
        name=f"determinism-{kind}",
        num_generations=4,
        population_size=16,
        solution_archive_size=32,
        stop_after_iterations=8,
        n_jobs=n_jobs,
        joblib_prefer=prefer,
        local_grad_optim=local_grad_optim,
    )
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        result = optimizer_cls(config=config, fcn=goal, variables=variables).solve()
    return round(float(result.solution_score), 12), len(evaluations)


@pytest.mark.parametrize("kind", list(OPTIMIZERS))
@pytest.mark.parametrize("n_jobs", [1, 3])
def test_same_seed_same_result(kind, n_jobs):
    """The headline promise, at one and at several workers."""
    runs = [solve_once(kind, n_jobs=n_jobs) for _ in range(3)]
    assert len(set(runs)) == 1, f"{kind} at n_jobs={n_jobs} varied across runs: {runs}"


@pytest.mark.parametrize("local_grad_optim", ["none", "perturb", "single-var-grad"])
def test_local_search_is_also_reproducible(local_grad_optim):
    """The local polish runs inside the worker tasks, so it draws from the
    task's stream too."""
    runs = [
        solve_once("ga", n_jobs=3, local_grad_optim=local_grad_optim) for _ in range(3)
    ]
    assert len(set(runs)) == 1, f"{local_grad_optim} varied: {runs}"


def test_processes_backend_is_reproducible():
    runs = [solve_once("ga", n_jobs=2, prefer="processes") for _ in range(3)]
    assert len(set(runs)) == 1, f"processes backend varied: {runs}"


def test_different_seeds_still_differ():
    """Reproducibility must not have been bought by making the seed inert."""
    assert solve_once("ga", seed=0, n_jobs=3) != solve_once("ga", seed=1, n_jobs=3)


# ---------------------------------------------------------------------------
# The stream machinery itself
# ---------------------------------------------------------------------------


def test_spawned_streams_are_deterministic_and_distinct():
    set_seed(7)
    first = [g.random(4).tolist() for g in spawn_streams(3)]
    set_seed(7)
    second = [g.random(4).tolist() for g in spawn_streams(3)]

    assert first == second, "the same seed must produce the same stream family"
    assert first[0] != first[1] != first[2], "workers must not share a stream"


def test_successive_spawns_give_fresh_numbers():
    """One family per generation: reproducible, but not the same numbers every
    generation."""
    set_seed(7)
    generation_one = [g.random(4).tolist() for g in spawn_streams(2)]
    generation_two = [g.random(4).tolist() for g in spawn_streams(2)]
    assert generation_one != generation_two


def test_use_stream_scopes_to_the_calling_thread_and_restores():
    set_seed(7)
    outer = rng()
    (stream,) = spawn_streams(1)

    with use_stream(stream):
        assert rng() is stream
    assert rng() is outer, "the task's stream must not leak past the task"


def test_use_stream_nests():
    set_seed(7)
    first, second = spawn_streams(2)
    with use_stream(first):
        with use_stream(second):
            assert rng() is second
        assert rng() is first


def test_set_seed_clears_a_stale_stream():
    """Re-seeding mid-flight must not leave the thread standing in a stream
    derived from the previous seed."""
    set_seed(7)
    (stream,) = spawn_streams(1)
    with use_stream(stream):
        set_seed(8)
        assert rng() is not stream


def test_worker_streams_do_not_disturb_the_main_stream():
    """The main thread's numbers are the same whether or not workers ran, so
    adding the worker family did not shift existing seeded behaviour."""
    set_seed(11)
    expected = rng().random(5).tolist()

    set_seed(11)
    spawn_streams(4)
    for generator in spawn_streams(4):
        generator.random(10)
    assert rng().random(5).tolist() == expected


# ---------------------------------------------------------------------------
# Nested parallelism: spawn_stream_roots / use_stream_root
#
# AntColonyMTSP dispatches one independent, itself-multi-generation ACO run
# per cluster in parallel. Each of those runs calls spawn_streams() many times
# internally, so giving the clusters plain spawned Generators (like leaf
# tasks get) isn't enough -- two clusters running concurrently would each be
# advancing spawn_streams()'s single shared counter from a different thread.
# spawn_stream_roots()/use_stream_root() give each such task its own
# independent root to spawn *from*, so its internal spawn_streams() calls
# can't collide with any other concurrently-running task's.
# ---------------------------------------------------------------------------


def test_stream_roots_are_deterministic_and_distinct():
    set_seed(7)
    first = [np.random.default_rng(r).random(4).tolist() for r in spawn_stream_roots(3)]
    set_seed(7)
    second = [
        np.random.default_rng(r).random(4).tolist() for r in spawn_stream_roots(3)
    ]

    assert first == second, "the same seed must produce the same root family"
    assert first[0] != first[1] != first[2], "roots must not collide"


def test_use_stream_root_isolates_nested_spawn_streams():
    """Two tasks standing in different roots must not see each other's
    nested spawn_streams() numbers, even though both call it the same way."""
    set_seed(7)
    root_a, root_b = spawn_stream_roots(2)

    with use_stream_root(root_a):
        a_children = [g.random(4).tolist() for g in spawn_streams(2)]
    with use_stream_root(root_b):
        b_children = [g.random(4).tolist() for g in spawn_streams(2)]

    assert a_children != b_children


def test_use_stream_root_scopes_to_the_calling_thread_and_restores():
    """Nested spawn_streams() calls inside a use_stream_root scope must not
    advance the shared global root's own spawn counter -- so code outside the
    scope sees exactly the numbers it would have if the scope never ran."""
    set_seed(7)
    (nested_root,) = spawn_stream_roots(1)
    with use_stream_root(nested_root):
        spawn_streams(5)  # busywork inside the nested root
    outside = [g.random(4).tolist() for g in spawn_streams(2)]

    set_seed(7)
    spawn_stream_roots(1)  # same single spawn from the global root as above
    expected_outside = [g.random(4).tolist() for g in spawn_streams(2)]

    assert outside == expected_outside


def test_use_stream_root_nests():
    set_seed(7)
    first, second = spawn_stream_roots(2)
    with use_stream_root(first):
        with use_stream_root(second):
            inner = spawn_streams(1)[0].random(4).tolist()
        middle = spawn_streams(1)[0].random(4).tolist()
    assert inner != middle, "nested and restored scopes must draw from different roots"


# ---------------------------------------------------------------------------
# AntColonyMTSP: clusters solve concurrently via joblib; each must be immune
# to how the others happen to be scheduled.
# ---------------------------------------------------------------------------


def solve_mtsp_once(seed, *, n_jobs=4):
    """One seeded AntColonyMTSP run over a fixed city layout."""
    city_locations = np.random.default_rng(42).uniform(0.0, 10.0, size=(24, 2))
    set_seed(seed)
    config = AntColonyMTSPConfig(
        name="determinism-mtsp",
        num_generations=3,
        population_size=8,
        n_clusters=3,
        clustering_method="kmeans",
        stop_after_iterations=8,
        n_jobs=n_jobs,
        joblib_prefer="threads",
    )
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        result = AntColonyMTSP(config=config, city_locations=city_locations).solve()
    return round(float(result.solution_score), 9)


def test_mtsp_parallel_clusters_are_reproducible():
    """Clusters run concurrently (n_clusters=3, n_jobs=4); a seeded run must
    still be a pure function of the seed regardless of scheduling."""
    runs = [solve_mtsp_once(3) for _ in range(3)]
    assert len(set(runs)) == 1, f"mtsp varied across runs: {runs}"


def test_mtsp_different_seeds_still_differ():
    assert solve_mtsp_once(3) != solve_mtsp_once(4)


# ---------------------------------------------------------------------------
# Combinatorial solvers (ACO-TSP, ACO-MST, GA-TSP): these dispatch their
# per-generation worker fan-out through core.parallel.GenerationRunner (the
# same copy-fixed-data-once machinery the continuous solvers use), so they
# inherit the same reproducibility guarantee this module pins above -- pin it
# directly here too, since it's the part of the combinatorial solvers most
# likely to regress silently if that dispatch is ever touched again.
# ---------------------------------------------------------------------------


def _small_distance_matrix(n=12, seed=42):
    points = np.random.default_rng(seed).uniform(0.0, 10.0, size=(n, 2))
    return pairwise_distances(points)


def solve_aco_tsp_once(seed, *, n_jobs=3):
    set_seed(seed)
    config = AntColonyTSPConfig(
        name="determinism-aco-tsp",
        num_generations=3,
        population_size=8,
        n_jobs=n_jobs,
        joblib_prefer="threads",
        stop_after_iterations=8,
    )
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        result = AntColonyTSP(
            config=config, network_routes=_small_distance_matrix()
        ).solve()
    return round(float(result.solution_score), 9)


def solve_aco_mst_once(seed, *, n_jobs=3):
    set_seed(seed)
    config = AntColonyTSPConfig(
        name="determinism-aco-mst",
        num_generations=3,
        population_size=8,
        n_jobs=n_jobs,
        joblib_prefer="threads",
        stop_after_iterations=8,
    )
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        result = AntColonyMST(
            config=config, network_routes=_small_distance_matrix()
        ).solve()
    return round(float(result.solution_score), 9)


def solve_ga_tsp_once(seed, *, n_jobs=3):
    set_seed(seed)
    config = GeneticAlgorithmTSPConfig(
        name="determinism-ga-tsp",
        num_generations=3,
        population_size=8,
        solution_archive_size=8,
        n_jobs=n_jobs,
        joblib_prefer="threads",
        stop_after_iterations=8,
    )
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        result = GeneticAlgorithmTSP(
            config=config, network_routes=_small_distance_matrix()
        ).solve()
    return round(float(result.solution_score), 9)


COMBINATORIAL_SOLVERS = {
    "aco-tsp": solve_aco_tsp_once,
    "aco-mst": solve_aco_mst_once,
    "ga-tsp": solve_ga_tsp_once,
}


@pytest.mark.parametrize("kind", list(COMBINATORIAL_SOLVERS))
@pytest.mark.parametrize("n_jobs", [1, 3])
def test_combinatorial_solver_same_seed_same_result(kind, n_jobs):
    solve_once = COMBINATORIAL_SOLVERS[kind]
    runs = [solve_once(5, n_jobs=n_jobs) for _ in range(3)]
    assert len(set(runs)) == 1, f"{kind} at n_jobs={n_jobs} varied across runs: {runs}"


@pytest.mark.parametrize("kind", list(COMBINATORIAL_SOLVERS))
def test_combinatorial_solver_different_seeds_still_differ(kind):
    solve_once = COMBINATORIAL_SOLVERS[kind]
    assert solve_once(5, n_jobs=3) != solve_once(6, n_jobs=3)
