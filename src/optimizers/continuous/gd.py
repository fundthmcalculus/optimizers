from dataclasses import dataclass

import joblib
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from tqdm.std import tqdm

from ..core.base import (
    IOptimizerConfig,
    OptimizerResult,
    GoalFcn,
    InputArguments,
)
from ..solution_deck import (
    WrappedGoalFcn,
    InputVariables,
)
from ..core.tqdm_joblib import tqdm_joblib
from .base import IOptimizer
from .variables import InputContinuousVariable, InputDiscreteVariable


@dataclass
class GradientDescentOptimizerConfig(IOptimizerConfig):
    parallel_discrete_search: bool = False
    discrete_search_size: int = -1  # Defaults to number of discrete variables


def solve_gd(variables: InputVariables, fcn: WrappedGoalFcn) -> tuple[OptimizeResult, list[float]]:
    x0 = [x.initial_value for x in variables]
    # Effectively pin the discrete values.
    bounds = [
        (
            (x.lower_bound, x.upper_bound)
            if isinstance(x, InputContinuousVariable)
            else (x.initial_value, x.initial_value)
        )
        for x in variables
    ]
    res: OptimizeResult = minimize(fcn, np.array(x0), bounds=bounds)
    x0_val = fcn(x0)
    x1_val = fcn(res.x)
    return res, [x0_val, x1_val]


def solve_gd_from_x0(
    x0: np.ndarray, variables: InputVariables, fcn: WrappedGoalFcn
) -> tuple[OptimizeResult, list[float]]:
    # Effectively pin the discrete values.
    bounds = [
        (
            (x.lower_bound, x.upper_bound)
            if isinstance(x, InputContinuousVariable)
            else (x0[ij], x0[ij])
        )
        for ij, x in enumerate(variables)
    ]
    res: OptimizeResult = minimize(fcn, np.array(x0), bounds=bounds)
    x0_val = fcn(x0)
    x1_val = fcn(res.x)
    return OptimizerResult(solution_vector=res.x, solution_score=res.fun), [x0_val, x1_val]


def solve_gd_with_mutate(
    variables: InputVariables, mutate_idx: int, fcn: WrappedGoalFcn
) -> OptimizerResult:
    x0 = [x.initial_value for x in variables]
    # Effectively pin the discrete values.
    bounds = [
        (
            (x.lower_bound, x.upper_bound)
            if isinstance(x, InputContinuousVariable)
            else (x.initial_value, x.initial_value)
        )
        for x in variables
    ]
    # Mutate the variable at the given index.
    x0[mutate_idx] = variables[mutate_idx].initial_random_value()
    bounds[mutate_idx] = (x0[mutate_idx], x0[mutate_idx])
    res: OptimizeResult = minimize(fcn, np.array(x0), bounds=bounds)
    return OptimizerResult(solution_vector=res.x, solution_score=res.fun)


def solve_gd_for_1var(
    x0: np.ndarray, variables: InputVariables, var_idx: int, fcn: WrappedGoalFcn
) -> OptimizerResult:
    bounds = [(x0[ij], x0[ij]) for ij in range(len(x0))]
    bounds[var_idx] = (variables[var_idx].lower_bound, variables[var_idx].upper_bound)
    res: OptimizeResult = minimize(fcn, np.array(x0), bounds=bounds)
    return OptimizerResult(solution_vector=res.x, solution_score=res.fun)


def _count_discrete_vars(variables: InputVariables) -> tuple[int, list[int]]:
    disc_vars = [
        (x, ij)
        for ij, x in enumerate(variables)
        if isinstance(x, InputDiscreteVariable)
    ]
    return len(disc_vars), [p[1] for p in disc_vars]


class GradientDescentOptimizer(IOptimizer):
    def __init__(
        self,
        config: IOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: InputArguments | None = None,
        inequality_constraints: list[GoalFcn] | None = None,
        equality_constraints: list[GoalFcn] | None = None,
    ):
        super().__init__(
            config,
            fcn,
            variables,
            args,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )
        self.config: GradientDescentOptimizerConfig = GradientDescentOptimizerConfig(
            **{**config.__dict__}
        )

    def solve(self, preserve_percent: float = 0.0) -> OptimizerResult:
        """Solve the optimization problem using gradient descent.

        Args:
            preserve_percent: This variable left unused

        Returns:
            The result of the optimization.
        """

        def compute_constraints_for_x(x: np.ndarray):
            # Evaluate wrapped constraints (already handle args) for a single x
            ineq_vals = None
            eq_vals = None
            ineq_rel = None
            eq_rel = None
            total = None
            if getattr(self, "wrapped_ineq_constraints", None):
                ineq_vals = np.array(
                    [g(x) for g in self.wrapped_ineq_constraints], dtype=float
                )
                ineq_rel = np.maximum(ineq_vals, 0.0)
            if getattr(self, "wrapped_eq_constraints", None):
                eq_vals = np.array(
                    [h(x) for h in self.wrapped_eq_constraints], dtype=float
                )
                eq_rel = np.abs(eq_vals)
            if ineq_rel is not None or eq_rel is not None:
                total = 0.0
                n_cons = 0
                if ineq_rel is not None:
                    total += float(np.sum(ineq_rel))
                    n_cons += int(ineq_rel.shape[0])
                if eq_rel is not None:
                    total += float(np.sum(eq_rel))
                    n_cons += int(eq_rel.shape[0])
                n_cons = max(1, n_cons)
                total = total / n_cons
            return ineq_vals, eq_vals, ineq_rel, eq_rel, total

        if self.config.parallel_discrete_search:
            # Look at the number of discrete variables
            n_disc_vars, disc_var_idxs = _count_discrete_vars(self.variables)
            if self.config.discrete_search_size < 1:
                self.config.discrete_search_size = n_disc_vars
            # Randomly pick the which discrete variable to tweak on each run
            rand_vars = np.random.randint(
                low=0, high=n_disc_vars, size=self.config.discrete_search_size
            )
            with tqdm_joblib(
                tqdm(desc="Gradient Descent Optimization", total=len(rand_vars))
            ):
                parallel = joblib.Parallel(
                    n_jobs=self.config.n_jobs,
                    prefer=self.config.joblib_prefer,
                )
                job_output: list[OptimizerResult] = parallel(
                    joblib.delayed(solve_gd_with_mutate)(
                        self.variables, disc_var_idxs[r_v], self.wrapped_fcn
                    )
                    for r_v in rand_vars
                )
                # Pick the best solution of the output options
                job_output = sorted(job_output, key=lambda x: x.solution_score)
                best = job_output[0]
                ineq_vals, eq_vals, ineq_rel, eq_rel, total = compute_constraints_for_x(
                    best.solution_vector
                )
                return OptimizerResult(
                    solution_vector=best.solution_vector,
                    solution_score=best.solution_score,
                    solution_history=best.solution_history,
                    stop_reason=best.stop_reason,
                    generations_completed=best.generations_completed,
                    total_constraint_violation=total,
                    ineq_relative_violations=ineq_rel,
                    eq_relative_violations=eq_rel,
                    ineq_values=ineq_vals,
                    eq_values=eq_vals,
                    unconstrained_best_score=best.solution_score,
                    unconstrained_best_vector=best.solution_vector,
                )
        else:
            res, history = solve_gd(self.variables, self.wrapped_fcn)
            ineq_vals, eq_vals, ineq_rel, eq_rel, total = compute_constraints_for_x(
                res.x
            )
            return OptimizerResult(
                solution_vector=res.x,
                solution_score=res.fun,
                solution_history=np.array(history),
                stop_reason="max_iterations",
                generations_completed=1,
                total_constraint_violation=total,
                ineq_relative_violations=ineq_rel,
                eq_relative_violations=eq_rel,
                ineq_values=ineq_vals,
                eq_values=eq_vals,
                unconstrained_best_score=res.fun,
                unconstrained_best_vector=res.x,
            )
