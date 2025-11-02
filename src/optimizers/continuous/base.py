import abc
import tqdm
import joblib
import numpy as np
import time
import uuid
import inspect
from typing import Optional, Callable, Any, Literal

from ..core import InputVariables
from ..core.base import (
    IOptimizerConfig,
    OptimizerResult,
    OptimizerRun,
    StopReason,
    ensure_literal_choice,
    JoblibPrefer,
    setup_for_generations,
    Phase,
)
from ..core.types import AF, F
from ..solution_deck import (
    GoalFcn,
    InputArguments,
    SolutionDeck,
    WrappedGoalFcn,
    WrappedConstraintFcn,
)


class _ArgProvider:
    """Holds and provides merged input arguments (user args + runtime metadata).

    This object is intentionally lightweight and picklable so wrapped callables
    can be serialized for parallel workers while still carrying the current metadata.
    """

    def __init__(self, base_args: Optional[InputArguments] = None):
        self.base_args: InputArguments = dict(base_args or {})
        self.meta: InputArguments = {}
        self._eval_count: int = 0

    def current(self) -> InputArguments:
        # merge without mutating user-provided dicts
        merged = {**self.base_args, **self.meta}
        return merged

    def bump_eval(self) -> None:
        # local to the process where the function executes
        self._eval_count += 1
        now = time.time()
        self.meta["eval_count"] = self._eval_count
        self.meta["now"] = now
        # Elapsed seconds since run start
        start = self.meta.get("start_time", now)
        try:
            self.meta["elapsed_time"] = float(now - start)
        except Exception:
            self.meta["elapsed_time"] = 0.0


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

        # Runtime metadata provider shared by wrapped callables
        self._arg_provider = _ArgProvider(args)
        self._run_id = str(uuid.uuid4())
        self._start_time = time.time()
        # Initialize baseline metadata
        self._arg_provider.meta.update(
            {
                "run_id": self._run_id,
                "start_time": self._start_time,
                "generation": 0,
                "max_generation": config.num_generations,
                "phase": "init",
                "n_jobs": config.n_jobs,
                "population_size": config.population_size,
                "optimizer_name": getattr(config, "name", type(config).__name__),
            }
        )

        # Detect whether functions accept (x, args) or just (x)
        def _accepts_args(func: Any) -> bool:
            try:
                sig = inspect.signature(func)
                return len(sig.parameters) >= 2
            except (ValueError, TypeError):
                return True  # assume safe to pass args

        def _wrap_goal(func: GoalFcn) -> WrappedGoalFcn:
            takes_args = _accepts_args(func)

            def __wrapped(x: AF, _f=func, _ap=self._arg_provider):
                _ap.bump_eval()
                if takes_args:
                    return _f(x, _ap.current())
                return _f(x)

            return __wrapped  # type: ignore[return-value]

        def _wrap_constraint(func: GoalFcn) -> WrappedConstraintFcn:
            takes_args = _accepts_args(func)

            def __wrapped(x: AF, _f=func, _ap=self._arg_provider):
                _ap.bump_eval()
                if takes_args:
                    return _f(x, _ap.current())
                return _f(x)

            return __wrapped  # type: ignore[return-value]

        # Wrap the goal function and constraint functions
        self.wrapped_fcn: WrappedGoalFcn = _wrap_goal(fcn)

        wrapped_ineq: list[WrappedConstraintFcn] | None = None
        wrapped_eq: list[WrappedConstraintFcn] | None = None
        if inequality_constraints:
            wrapped_ineq = [_wrap_constraint(g) for g in inequality_constraints]
        if equality_constraints:
            wrapped_eq = [_wrap_constraint(h) for h in equality_constraints]

        # Save wrapped constraints for use by optimizers that don't use SolutionDeck internally
        self.wrapped_ineq_constraints = wrapped_ineq or []
        self.wrapped_eq_constraints = wrapped_eq or []
        self.soln_deck = existing_soln_deck or SolutionDeck(
            archive_size=config.solution_archive_size,
            num_vars=len(variables),
            inequality_constraints=self.wrapped_ineq_constraints,
            equality_constraints=self.wrapped_eq_constraints,
        )

    def _set_phase(self, phase: Phase) -> None:
        # Validate at runtime for extra safety in non-type-checked contexts
        try:
            ensure_literal_choice("phase", phase, Phase)
        except Exception:
            # Keep fail-soft: still set, but this should not happen given our callers
            pass
        self._arg_provider.meta["phase"] = phase

    def _set_generation(self, generation: int) -> None:
        self._arg_provider.meta["generation"] = generation

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
