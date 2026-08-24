"""Timing harness for the GA/ACO/PSO continuous solvers.

Runs each solver against the common test functions in
:mod:`optimizers.benchmarks.functions`, across several seeds so the caller can
report a mean and a spread (error bars), rather than a single noisy sample.

This module is intentionally free of matplotlib / CLI concerns -- see
``benchmarks/run_benchmark.py`` at the repo root for the script that calls
this harness and renders the comparison plot, and
``tests/test_benchmarks.py`` for a fast correctness-only smoke test.

Scope note: every run here uses ``local_grad_optim="none"``. The combinatorial
solvers and the gradient-descent/local-search optimizers are out of scope for
this benchmark (see docs/history/PERF_CONTINUOUS_REPORT.md).
"""

import time
from dataclasses import dataclass, field
from typing import Literal

from ..continuous.aco import AntColonyOptimizer, AntColonyOptimizerConfig
from ..continuous.ga import GeneticAlgorithmOptimizer, GeneticAlgorithmOptimizerConfig
from ..continuous.pso import ParticleSwarmOptimizer, ParticleSwarmOptimizerConfig
from ..continuous.variables import InputContinuousVariable
from ..core.base import IOptimizerConfig
from ..core.random import set_seed
from ..core.variables import InputVariable
from .functions import TEST_FUNCTIONS, TestFunction

EvalMode = Literal["scalar", "batch"]

OPTIMIZERS = {
    "ACO": (AntColonyOptimizer, AntColonyOptimizerConfig),
    "GA": (GeneticAlgorithmOptimizer, GeneticAlgorithmOptimizerConfig),
    "PSO": (ParticleSwarmOptimizer, ParticleSwarmOptimizerConfig),
}


@dataclass
class BenchmarkResult:
    optimizer: str
    function: str
    mode: EvalMode
    seed: int
    wall_time: float
    best_value: float


@dataclass
class BenchmarkSpec:
    optimizers: list[str] = field(default_factory=lambda: list(OPTIMIZERS))
    functions: list[str] = field(default_factory=lambda: list(TEST_FUNCTIONS))
    modes: tuple[EvalMode, ...] = ("scalar", "batch")
    seeds: tuple[int, ...] = tuple(range(8))
    n_dim: int = 10
    population_size: int = 60
    num_generations: int = 60
    solution_archive_size: int = 200


def _make_variables(test_fn: TestFunction, n_dim: int) -> list[InputVariable]:
    return [
        InputContinuousVariable(f"x{i}", test_fn.lower_bound, test_fn.upper_bound)
        for i in range(n_dim)
    ]


def time_one_run(
    optimizer_name: str,
    test_fn: TestFunction,
    mode: EvalMode,
    seed: int,
    n_dim: int,
    population_size: int,
    num_generations: int,
    solution_archive_size: int,
) -> BenchmarkResult:
    """Run one solver, on one function, one seed, and time it (wall clock).

    ``local_grad_optim`` is always ``"none"`` -- this harness is scoped to the
    GA/ACO/PSO population loop itself, not the local-search add-ons.
    """
    optimizer_cls, config_cls = OPTIMIZERS[optimizer_name]
    set_seed(seed)
    variables = _make_variables(test_fn, n_dim)
    config: IOptimizerConfig = config_cls(
        name=f"{optimizer_name}-{test_fn.name}-{mode}-{seed}",
        population_size=population_size,
        num_generations=num_generations,
        solution_archive_size=solution_archive_size,
        n_jobs=1,
        joblib_prefer="threads",
        local_grad_optim="none",
    )
    batch_fcn = test_fn.batch if mode == "batch" else None
    optimizer = optimizer_cls(
        config=config, variables=variables, fcn=test_fn.scalar, batch_fcn=batch_fcn
    )
    start = time.perf_counter()
    result = optimizer.solve()
    elapsed = time.perf_counter() - start
    return BenchmarkResult(
        optimizer=optimizer_name,
        function=test_fn.name,
        mode=mode,
        seed=seed,
        wall_time=elapsed,
        best_value=float(result.solution_score),
    )


def run_benchmark_grid(spec: BenchmarkSpec) -> list[BenchmarkResult]:
    """Run the full (optimizer x function x mode x seed) grid and return results."""
    results: list[BenchmarkResult] = []
    for optimizer_name in spec.optimizers:
        for function_name in spec.functions:
            test_fn = TEST_FUNCTIONS[function_name]
            for mode in spec.modes:
                for seed in spec.seeds:
                    results.append(
                        time_one_run(
                            optimizer_name,
                            test_fn,
                            mode,
                            seed,
                            spec.n_dim,
                            spec.population_size,
                            spec.num_generations,
                            spec.solution_archive_size,
                        )
                    )
    return results
