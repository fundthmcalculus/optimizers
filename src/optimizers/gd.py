from dataclasses import dataclass

import joblib
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from tqdm.std import tqdm

from .optimizer_base import (
    IOptimizerConfig,
    IOptimizer,
    GoalFcn,
    OptimizerResult,
    InputArguments,
)
from .tqdm_joblib import tqdm_joblib
from .variables import InputContinuousVariable, InputDiscreteVariable, InputVariables


@dataclass
class GradientDescentOptimizerConfig(IOptimizerConfig):
    parallel_discrete_search: bool = False
    discrete_search_size: int = -1  # Defaults to number of discrete variables


def solve_gd(variables: InputVariables, fcn: GoalFcn) -> OptimizerResult:
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
    return OptimizerResult(solution_vector=res.x, solution_score=res.fun)


def solve_gd_from_x0(
    x0: np.ndarray, variables: InputVariables, fcn: GoalFcn
) -> OptimizerResult:
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
    return OptimizerResult(solution_vector=res.x, solution_score=res.fun)


def solve_gd_with_mutate(
    variables: InputVariables, mutate_idx: int, fcn: GoalFcn
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
    x0: np.ndarray, variables: InputVariables, var_idx: int, fcn: GoalFcn
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
        name: str,
        config: GradientDescentOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: InputArguments | None = None,
    ):
        super().__init__(name, config, fcn, variables, args)
        self.config: GradientDescentOptimizerConfig = config

    def solve(self) -> OptimizerResult:

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
            ) as progress_bar:
                parallel = joblib.Parallel(
                    n_jobs=self.config.joblib_num_procs,
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
                return job_output[0]
        else:
            return solve_gd(self.variables, self.wrapped_fcn)
