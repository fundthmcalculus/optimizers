import abc
import tqdm
import joblib
from dataclasses import dataclass
from typing import Literal

from solution_deck import SolutionDeck, GoalFcn, InputArguments
from variables import InputVariables
from opt_types import *


@dataclass
class IOptimizerConfig:
    """Base class for optimizer configurations."""

    name: str
    """The name of the optimizer. This is used for logging purposes."""
    num_generations: int = 50
    """The number of generations to run the optimizer"""
    population_size: int = 30
    """The population size for each generation of the optimizer."""
    solution_archive_size: int = 100
    """Size of solution archive used as memory of good solutions"""
    stop_after_iterations: int = 50
    """Stop after a certain number of iterations. This is used for early stopping if nothing improves"""
    target_score: float = 0.0
    """The target score for the optimizer to achieve. This is used for early stopping."""
    joblib_num_procs: int = 4
    """The number of processes to use for parallel execution. -1 means use all available cores."""
    joblib_prefer: Literal["threads", "processes"] = "processes"
    """The preferred execution mode for joblib. See https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html#joblib.Parallel for more details."""


@dataclass
class OptimizerResult:
    """Base class for optimizer results."""

    solution_score: f64
    """The score of the best solution found by the optimizer."""
    solution_vector: af64
    """The best solution found by the optimizer."""
    solution_history: af64 = None
    """The history of the best solutions found by the optimizer."""
    stopped_early: bool = False
    """Whether the optimizer stopped early due to convergence criteria."""
    generations_completed: int = 0
    """Number of generations completed before stopping."""

    def __repr__(self):
        return f"{self.__class__.__name__}(val={self.solution_score}, x={self.solution_vector})"

    def __str__(self):
        return self.__repr__()
    
    def __plus__(self, other):
        if not isinstance(other, OptimizerResult):
            return NotImplemented
        combined_history = None
        if self.solution_history is not None and other.solution_history is not None:
            combined_history = np.concatenate((self.solution_history, other.solution_history))
        elif self.solution_history is not None:
            combined_history = self.solution_history
        elif other.solution_history is not None:
            combined_history = other.solution_history
        
        return OptimizerResult(
            solution_score=min(self.solution_score, other.solution_score),
            solution_vector=self.solution_vector if self.solution_score <= other.solution_score else other.solution_vector,
            solution_history=combined_history,
            stopped_early=self.stopped_early or other.stopped_early,
            generations_completed=self.generations_completed + other.generations_completed
        )


class IOptimizer(abc.ABC):
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
        self.args: InputArguments = args
        # Wrap the goal function if needed
        if args:
            wrapped_fcn = lambda x: fcn(x, args)
        else:
            wrapped_fcn = fcn
        self.wrapped_fcn = wrapped_fcn
        self.soln_deck = existing_soln_deck or SolutionDeck(archive_size=config.solution_archive_size, num_vars=len(variables))

    @abc.abstractmethod
    def solve(self, preserve_percent: float = 0.0) -> OptimizerResult:
        """
        Solve the given problem.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def __str__(self):
        return f"Solver(name={self.name})"


def setup_for_generations(config: IOptimizerConfig):
    generation_pbar = tqdm.trange(config.num_generations, desc=f"Optimizer generation")
    n_jobs = config.joblib_num_procs
    if n_jobs < 1:
        n_jobs = joblib.cpu_count() - 1
    individuals_per_job = max(1, config.population_size // n_jobs)
    parallel = joblib.Parallel(n_jobs=n_jobs, prefer=config.joblib_prefer)
    return generation_pbar, individuals_per_job, n_jobs, parallel


def check_stop_early(
    config: IOptimizerConfig, best_soln_history: af64, solution_values: af64
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


def cdf(q: float, N: int) -> af64:
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
    # Unity scaling, and since the CDF is positive definite, we can use the last entry.
    return c1 / c1[-1]