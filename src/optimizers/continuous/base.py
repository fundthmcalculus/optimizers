import abc
import tqdm
import joblib
import numpy as np
from typing import Optional

from ..core import InputVariables
from ..core.base import (
    IOptimizerConfig,
    OptimizerResult,
    OptimizerRun,
    StopReason,
    ensure_literal_choice,
    JoblibPrefer,
)
from ..core.types import AF, F
from ..solution_deck import (
    GoalFcn,
    InputArguments,
    SolutionDeck,
    WrappedGoalFcn,
    WrappedConstraintFcn,
)


class IOptimizer(abc.ABC):
    """Base class for all optimizer implementations"""

    def __init__(
        self,
        config: IOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: Optional[InputArguments] = None,
        existing_soln_deck: Optional[SolutionDeck] = None,
        inequality_constraints: Optional[list[GoalFcn]] = None,
        equality_constraints: Optional[list[GoalFcn]] = None,
    ):
        self.config: IOptimizerConfig = config
        self.variables: InputVariables = variables
        self.args: Optional[InputArguments] = args
        # Wrap the goal function if needed
        if args:

            def __wrapped_fcn(x: AF) -> AF:
                return fcn(x, args)

            wrapped_fcn = __wrapped_fcn
        else:
            wrapped_fcn = fcn
        self.wrapped_fcn: WrappedGoalFcn = wrapped_fcn
        # Wrap constraint functions similarly
        wrapped_ineq: list[WrappedConstraintFcn] | None = None
        wrapped_eq: list[WrappedConstraintFcn] | None = None
        if inequality_constraints:
            wrapped_ineq = []
            for g in inequality_constraints:
                if args:
                    wrapped_ineq.append(lambda x, g=g: g(x, args))
                else:
                    wrapped_ineq.append(g)  # type: ignore[arg-type]
        if equality_constraints:
            wrapped_eq = []
            for h in equality_constraints:
                if args:
                    wrapped_eq.append(lambda x, h=h: h(x, args))
                else:
                    wrapped_eq.append(h)  # type: ignore[arg-type]
        # Save wrapped constraints for use by optimizers that don't use SolutionDeck internally
        self.wrapped_ineq_constraints = wrapped_ineq or []
        self.wrapped_eq_constraints = wrapped_eq or []
        self.soln_deck = existing_soln_deck or SolutionDeck(
            archive_size=config.solution_archive_size,
            num_vars=len(variables),
            inequality_constraints=self.wrapped_ineq_constraints,
            equality_constraints=self.wrapped_eq_constraints,
        )

    @abc.abstractmethod
    def solve(self, preserve_percent: float = 0.0) -> OptimizerResult:
        """
        Solve the given problem.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def initialize(
        self, preserve_percent: float
    ) -> tuple[AF, tqdm.tqdm, int, int, int, joblib.Parallel, bool]:
        self.validate_config()
        self.soln_deck.initialize_solution_deck(
            self.variables, self.wrapped_fcn, preserve_percent
        )
        self.soln_deck.sort()
        best_soln_history = np.zeros(self.config.num_generations)

        # Add the progress bar
        generation_pbar, individuals_per_job, n_jobs, parallel = setup_for_generations(
            self.config
        )
        stopped_early = False
        generations_completed = 0
        return (
            best_soln_history,
            generation_pbar,
            generations_completed,
            individuals_per_job,
            n_jobs,
            parallel,
            stopped_early,
        )

    def update_solution_deck(
        self, generation_pbar: tqdm, job_output: list[OptimizerRun]
    ):
        for output in job_output:
            self.soln_deck.append(
                solutions=output.population_solutions,
                values=output.population_values,
                local_optima=self.config.local_grad_optim != "none",
            )
            self.soln_deck.deduplicate()
        generation_pbar.set_postfix(best_value=self.soln_deck.solution_value[0])

    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        """
        # Validate joblib prefer value against allowed Literal options
        ensure_literal_choice("joblib_prefer", self.config.joblib_prefer, JoblibPrefer)
        # Set the default values for the config
        if self.config.solution_archive_size < 0:
            self.config.solution_archive_size = len(self.variables) * 2
        if self.config.population_size < 0:
            self.config.population_size = self.config.solution_archive_size // 3
        if self.config.n_jobs < 0:
            self.config.n_jobs = joblib.cpu_count() - 1

    def __str__(self):
        return f"Solver(name={self.config.name})"


def setup_for_generations(config: IOptimizerConfig):
    generation_pbar = tqdm.trange(config.num_generations, desc="Optimizer generation")
    n_jobs = config.n_jobs
    if n_jobs < 1:
        n_jobs = joblib.cpu_count() - 1
    individuals_per_job = max(1, config.population_size // n_jobs)
    parallel = joblib.Parallel(n_jobs=n_jobs, prefer=config.joblib_prefer)
    return generation_pbar, individuals_per_job, n_jobs, parallel


def check_stop_early(
    config: IOptimizerConfig, best_soln_history: AF, solution_values: AF
) -> StopReason:
    if solution_values[0] <= config.target_score:
        print("Target score reached, terminating early.")
        return "target_score"
    # Check if the solution hasn't improved
    recent_history = best_soln_history[-config.stop_after_iterations :]
    if np.allclose(recent_history, recent_history[0], rtol=1e-2, atol=1e-2) and np.all(
        recent_history > 0
    ):
        print(
            f"No improvement in last {config.stop_after_iterations} iterations. Stopping early."
        )
        return "no_improvement"
    return "none"


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
