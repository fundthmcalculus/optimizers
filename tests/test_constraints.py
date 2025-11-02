import numpy as np
import pytest

from optimizers.continuous.ga import (
    GeneticAlgorithmOptimizerConfig,
    GeneticAlgorithmOptimizer,
)
from optimizers.continuous.variables import InputContinuousVariable
from optimizers.core.types import AF


@pytest.fixture()
def ga_small_cfg():
    # Keep things small and fast for CI while still exploring enough
    return GeneticAlgorithmOptimizerConfig(
        name="GA-constraints",
        num_generations=6,
        population_size=12,
        solution_archive_size=24,
        n_jobs=1,
        stop_after_iterations=3,
        local_grad_optim="none",
    )


def test_inequality_constraints_affect_ordering(ga_small_cfg):
    # Objective prefers x1 ~= 1, x2 ~= 0
    def quad_obj(x: AF):
        x = np.asarray(x)
        return float((x[0] - 1.0) ** 2 + (x[1] - 0.0) ** 2)

    # Inequality: x1 - 0.5 <= 0  (i.e., x1 <= 0.5)
    def ineq_g(x: AF):
        x = np.asarray(x)
        return float(x[0] - 0.5)

    variables = [
        InputContinuousVariable("x1", -2.0, 2.0),
        InputContinuousVariable("x2", -2.0, 2.0),
    ]

    opt = GeneticAlgorithmOptimizer(
        ga_small_cfg,
        quad_obj,
        variables,
        inequality_constraints=[ineq_g],
    )
    res = opt.solve()

    # Best respecting constraints should be feasible (violation ~ 0)
    assert res.total_constraint_violation is not None
    assert res.total_constraint_violation <= 1e-2
    assert res.ineq_relative_violations is not None
    assert np.all(res.ineq_relative_violations >= 0)

    # Since unconstrained optimum is at x1=1 (>0.5), the unconstrained-best
    # candidate in the deck should be closer to 1 than the reported (feasible) best.
    assert res.unconstrained_best_vector is not None
    assert res.solution_vector is not None

    feasible_x1 = float(res.solution_vector[0])
    unconstrained_x1 = float(res.unconstrained_best_vector[0])

    # Unconstrained best should be on the violating side, i.e., > 0.5 and closer to 1
    assert unconstrained_x1 > 0.5
    assert abs(unconstrained_x1 - 1.0) < abs(feasible_x1 - 1.0)

    # And typically the unconstrained score is better than the feasible one
    assert res.unconstrained_best_score is not None
    assert res.unconstrained_best_score <= res.solution_score + 1e-3


def test_equality_constraint_reporting(ga_small_cfg):
    # Objective prefers x2 ~= 0, equality requires x2 == 0.1
    def obj(x: AF):
        x = np.asarray(x)
        return float(x[0] ** 2 + (x[1] - 0.0) ** 2)

    def eq_h(x: AF):
        x = np.asarray(x)
        return float(x[1] - 0.1)  # = 0 when x2 == 0.1

    variables = [
        InputContinuousVariable("x1", -1.0, 1.0),
        InputContinuousVariable("x2", -1.0, 1.0),
    ]

    opt = GeneticAlgorithmOptimizer(
        ga_small_cfg,
        obj,
        variables,
        equality_constraints=[eq_h],
    )
    res = opt.solve()

    # Reported best should try to satisfy equality (near x2=0.1)
    x2_best = float(res.solution_vector[1])
    assert abs(x2_best - 0.1) < 0.2  # loose tolerance due to small GA budget

    # Check reported arrays and relation eq_relative == abs(eq_values)
    assert res.eq_values is not None
    assert res.eq_relative_violations is not None
    assert np.allclose(
        np.abs(np.asarray(res.eq_values, dtype=float)),
        np.asarray(res.eq_relative_violations, dtype=float),
        rtol=1e-6,
        atol=1e-6,
    )

    # The unconstrained-best should lie closer to x2=0 (the unconstrained optimum)
    assert res.unconstrained_best_vector is not None
    x2_unconstrained = float(res.unconstrained_best_vector[1])
    assert abs(x2_unconstrained - 0.0) < abs(x2_best - 0.0) + 1e-6
