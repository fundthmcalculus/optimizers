import abc
import tqdm
import joblib
import numpy as np

from ..solution_deck import (
    SolutionDeck,
    GoalFcn,
    InputArguments,
    WrappedGoalFcn,
    InputVariables,
)
from ..core.types import AF, F
from ..core.base import IOptimizerConfig, OptimizerResult


class OptimizerBase(abc.ABC):
    """Base class for all optimizer implementations"""

    def __init__(
        self,
        name: str,
        config: IOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: InputArguments | None = None,
        existing_soln_deck: SolutionDeck | None = None,
    ):
        self.name: str = name
        self.config: IOptimizerConfig = config
        self.variables: InputVariables = variables
        self.args: InputArguments | None = args
        # Wrap the goal function if needed
        if args:

            def __wrapped_fcn(x: AF) -> AF:
                return fcn(x, args)

            wrapped_fcn = __wrapped_fcn
        else:
            wrapped_fcn = fcn
        self.wrapped_fcn: WrappedGoalFcn = wrapped_fcn
        self.soln_deck = existing_soln_deck or SolutionDeck(
            archive_size=config.solution_archive_size, num_vars=len(variables)
        )

    @abc.abstractmethod
    def solve(self, preserve_percent: float = 0.0) -> OptimizerResult:
        """
        Solve the given problem.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        """
        # Set the default values for the config
        if self.config.solution_archive_size < 0:
            self.config.solution_archive_size = len(self.variables) * 2
        if self.config.population_size < 0:
            self.config.population_size = self.config.solution_archive_size // 3
        if self.config.joblib_num_procs < 0:
            self.config.joblib_num_procs = joblib.cpu_count() - 1

    def __str__(self):
        return f"Solver(name={self.name})"


def setup_for_generations(config: IOptimizerConfig):
    generation_pbar = tqdm.trange(config.num_generations, desc="Optimizer generation")
    n_jobs = config.joblib_num_procs
    if n_jobs < 1:
        n_jobs = joblib.cpu_count() - 1
    individuals_per_job = max(1, config.population_size // n_jobs)
    parallel = joblib.Parallel(n_jobs=n_jobs, prefer=config.joblib_prefer)
    return generation_pbar, individuals_per_job, n_jobs, parallel


def check_stop_early(
    config: IOptimizerConfig, best_soln_history: AF, solution_values: AF
) -> bool:
    stop_early = False
    if solution_values[0] <= config.target_score:
        print("Target score reached, terminating early.")
        stop_early = True
    # Check if the solution hasn't improved
    recent_history = best_soln_history[-config.stop_after_iterations :]
    if np.allclose(recent_history, recent_history[0], rtol=1e-2, atol=1e-2) and np.all(
        recent_history > 0
    ):
        print(
            f"No improvement in last {config.stop_after_iterations} iterations. Stopping early."
        )
        stop_early = True
    return stop_early


def cdf(q: F, N: int) -> AF:
    """
    Parameters
    ----------
    q: float The weighting parameter for better ranked solutions.
    N: int The number of solutions in the solution archive.

    Returns
    -------
    af64 The cumulative density function.
    """
    j = np.r_[1 : N + 1]
    c1 = 1 - np.exp(-q * j / N)
    # Unity scaling, and since the CDF is positive-definite, we can use the last entry.
    return c1 / c1[-1]
