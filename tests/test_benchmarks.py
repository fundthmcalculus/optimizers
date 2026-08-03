"""Correctness tests for the GA/ACO/PSO benchmark harness (PERF_CONTINUOUS_REPORT.md).

Fast/small by design (a handful of generations, tiny populations) -- these
check correctness of the scalar/batch reference implementations and the
optional batched-evaluation fast path, not timing. See
``benchmarks/run_benchmark.py`` for the actual (slower, multi-seed) timing
comparison and plot.
"""

import numpy as np
import pytest

from optimizers.benchmarks.functions import (
    TEST_FUNCTIONS,
    ackley,
    ackley_batch,
    rastrigin,
    rastrigin_batch,
    rosenbrock,
    rosenbrock_batch,
    sphere,
    sphere_batch,
)
from optimizers.benchmarks.harness import BenchmarkSpec, run_benchmark_grid
from optimizers.benchmarks.cython_kernels import (
    HAS_CYTHON,
    ackley_batch_cy,
    rosenbrock_batch_cy,
)
from optimizers.continuous.aco import AntColonyOptimizer, AntColonyOptimizerConfig
from optimizers.continuous.ga import (
    GeneticAlgorithmOptimizer,
    GeneticAlgorithmOptimizerConfig,
)
from optimizers.continuous.pso import (
    ParticleSwarmOptimizer,
    ParticleSwarmOptimizerConfig,
)
from optimizers.continuous.variables import InputContinuousVariable
from optimizers.core.random import set_seed


@pytest.mark.parametrize(
    "scalar_fn,batch_fn",
    [
        (sphere, sphere_batch),
        (rosenbrock, rosenbrock_batch),
        (ackley, ackley_batch),
        (rastrigin, rastrigin_batch),
    ],
)
def test_scalar_and_batch_reference_agree(scalar_fn, batch_fn):
    rng = np.random.default_rng(0)
    x = rng.uniform(-5, 5, size=(25, 7))
    expected = np.array([scalar_fn(row) for row in x])
    actual = batch_fn(x)
    assert np.allclose(expected, actual)


def test_test_functions_registry_bounds_and_optimum():
    for name, spec in TEST_FUNCTIONS.items():
        assert spec.name == name
        assert spec.lower_bound < spec.upper_bound
        zeros = np.zeros(4)
        # Every registered function's own convention: f(0,...,0) is at (or very
        # near, for rosenbrock which is centered at 1) its documented optimum
        # only for sphere/ackley/rastrigin; rosenbrock is checked separately.
        if name != "rosenbrock":
            assert np.isclose(spec.scalar(zeros), spec.optimum_value, atol=1e-6)
    ones = np.ones(4)
    assert np.isclose(TEST_FUNCTIONS["rosenbrock"].scalar(ones), 0.0, atol=1e-9)


@pytest.mark.parametrize(
    "optimizer_cls,config_cls",
    [
        (AntColonyOptimizer, AntColonyOptimizerConfig),
        (GeneticAlgorithmOptimizer, GeneticAlgorithmOptimizerConfig),
        (ParticleSwarmOptimizer, ParticleSwarmOptimizerConfig),
    ],
)
def test_batched_evaluation_matches_scalar_path(optimizer_cls, config_cls):
    # local_grad_optim="none" is required for the batched-evaluation fast path
    # (see PERF_CONTINUOUS_REPORT.md); with it enabled, GA/ACO/PSO should reach
    # a bit-identical result whether or not a batch_fcn is supplied, since both
    # paths must compute the same numbers -- only how many Python-level calls
    # it takes to get there differs.
    def make_variables():
        return [InputContinuousVariable(f"x{i}", -5.0, 5.0) for i in range(5)]

    def run(batch_fcn):
        set_seed(3)
        config = config_cls(
            name="t",
            population_size=16,
            num_generations=8,
            solution_archive_size=40,
            n_jobs=1,
            joblib_prefer="threads",
            local_grad_optim="none",
        )
        optimizer = optimizer_cls(
            config=config, variables=make_variables(), fcn=ackley, batch_fcn=batch_fcn
        )
        return optimizer.solve()

    result_scalar = run(None)
    result_batch = run(ackley_batch)
    assert np.isclose(result_scalar.solution_score, result_batch.solution_score)
    assert np.allclose(result_scalar.solution_vector, result_batch.solution_vector)


def test_run_benchmark_grid_smoke():
    # Tiny grid: exercises the harness end to end without the full (slow)
    # multi-seed sweep run by benchmarks/run_benchmark.py.
    spec = BenchmarkSpec(
        optimizers=["GA"],
        functions=["sphere"],
        modes=("scalar", "batch"),
        seeds=(0, 1),
        n_dim=3,
        population_size=10,
        num_generations=5,
        solution_archive_size=20,
    )
    results = run_benchmark_grid(spec)
    assert len(results) == 2 * 2  # modes x seeds
    assert all(r.wall_time > 0 for r in results)
    assert all(np.isfinite(r.best_value) for r in results)


@pytest.mark.skipif(not HAS_CYTHON, reason="compiled _bench_cython extension not built")
def test_cython_kernels_match_numpy_reference():
    rng = np.random.default_rng(1)
    x = rng.uniform(-5, 5, size=(50, 12))
    assert np.allclose(ackley_batch(x), ackley_batch_cy(x), atol=1e-9)
    assert np.allclose(rosenbrock_batch(x), rosenbrock_batch_cy(x), atol=1e-6)
