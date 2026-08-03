"""Common continuous test functions for benchmarking GA/ACO/PSO.

Every function ships two equivalent, pure-Python/NumPy implementations:

* the ``scalar`` form ``f(x) -> float``, matching the ``GoalFcn`` contract the
  optimizers have always accepted (one candidate vector in, one score out);
* the ``batch`` form ``f_batch(X) -> AF``, taking the whole generation's
  candidate matrix ``(n, d)`` at once and returning ``(n,)`` scores.

The batch form is not a different algorithm -- it is the same arithmetic,
written so NumPy evaluates every row in one vectorized pass instead of the
optimizer calling the scalar form once per individual. It is the reference
implementation for the optional ``batch_fcn`` hot-path described in
PERF_CONTINUOUS_REPORT.md. Both forms are plain Python; there is no compiled
kernel here (see ``optimizers.benchmarks.cython_kernels`` for the optional
compiled follow-up).
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..core.types import AF


def sphere(x: AF) -> float:
    """The N-dimensional sphere function. Global minimum: f(0,...,0) = 0."""
    return float(np.sum(x**2))


def sphere_batch(x: AF) -> AF:
    result: AF = np.sum(x**2, axis=-1)
    return result


def rosenbrock(x: AF) -> float:
    """The N-dimensional Rosenbrock ("banana") function.

    https://en.wikipedia.org/wiki/Rosenbrock_function
    Global minimum: f(1,...,1) = 0.
    """
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rosenbrock_batch(x: AF) -> AF:
    result: AF = np.sum(
        100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1.0 - x[..., :-1]) ** 2,
        axis=-1,
    )
    return result


_ACKLEY_A, _ACKLEY_B, _ACKLEY_C = 20.0, 0.2, 2 * np.pi
_RASTRIGIN_A = 10.0


def ackley(x: AF) -> float:
    """The N-dimensional Ackley function. Global minimum: f(0,...,0) = 0.

    Note: takes a single ``x`` argument (no optional tunables) on purpose --
    the optimizers' ``GoalFcn``/``BatchGoalFcn`` contract distinguishes
    ``f(x)`` from ``f(x, args)`` by counting a callable's parameters (see
    ``IOptimizer._accepts_args``), so a second positional-or-keyword parameter
    here would be misread as the runtime-metadata ``args`` dict.
    """
    d = len(x)
    a, b, c = _ACKLEY_A, _ACKLEY_B, _ACKLEY_C
    sum_sq = float(np.sum(x**2))
    sum_cos = float(np.sum(np.cos(c * x)))
    return float(-a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + np.e)


def ackley_batch(x: AF) -> AF:
    d = x.shape[-1]
    a, b, c = _ACKLEY_A, _ACKLEY_B, _ACKLEY_C
    result: AF = (
        -a * np.exp(-b * np.sqrt(np.sum(x**2, axis=-1) / d))
        - np.exp(np.sum(np.cos(c * x), axis=-1) / d)
        + a
        + np.e
    )
    return result


def rastrigin(x: AF) -> float:
    """The N-dimensional Rastrigin function. Global minimum: f(0,...,0) = 0."""
    d = len(x)
    a = _RASTRIGIN_A
    return float(a * d + np.sum(x**2 - a * np.cos(2 * np.pi * x)))


def rastrigin_batch(x: AF) -> AF:
    d = x.shape[-1]
    a = _RASTRIGIN_A
    result: AF = a * d + np.sum(x**2 - a * np.cos(2 * np.pi * x), axis=-1)
    return result


@dataclass(frozen=True)
class TestFunction:
    """A benchmark objective bundled with its bounds and known optimum."""

    name: str
    scalar: Callable[[AF], float]
    batch: Callable[[AF], AF]
    lower_bound: float
    upper_bound: float
    optimum_value: float = 0.0


TEST_FUNCTIONS: dict[str, TestFunction] = {
    "sphere": TestFunction("sphere", sphere, sphere_batch, -5.0, 5.0),
    "rosenbrock": TestFunction("rosenbrock", rosenbrock, rosenbrock_batch, -5.0, 10.0),
    "ackley": TestFunction("ackley", ackley, ackley_batch, -32.768, 32.768),
    "rastrigin": TestFunction("rastrigin", rastrigin, rastrigin_batch, -5.12, 5.12),
}
