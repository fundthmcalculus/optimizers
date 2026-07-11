import os
from typing import Literal, Optional, TypeVar, get_args, Callable, Union, Any
from dataclasses import dataclass, fields
import numpy as np
from joblib import cpu_count, Parallel
from tqdm import trange, tqdm

from .types import AF, F

_TRUTHY = {"1", "true", "yes", "on"}

# Fast-tests caps. Applied to every optimizer config when ``OPTIMIZERS_FAST`` is
# set, so the CI/agentic pipeline finishes in a reasonable timeline instead of
# running the full-fidelity workloads. The caps are deliberately at (not below)
# the values the correctness-threshold tests rely on (e.g. the map-elites
# coverage checks run 15 generations over a population of 40), so only the heavy
# long-running configs are shrunk while every assertion keeps holding.
_FAST_MAX_GENERATIONS = 15
_FAST_MAX_POPULATION = 40


def fast_tests_enabled() -> bool:
    """Whether the ``OPTIMIZERS_FAST`` environment variable caps optimizer workloads.

    This is the runtime analogue of ``OPTIMIZERS_NO_SHOW`` (which forces headless
    plotting): a single global switch that keeps the test/CI pipeline fast. It is
    off by default, so ordinary runs use the full configured workloads.
    """
    return os.environ.get("OPTIMIZERS_FAST", "").strip().lower() in _TRUTHY


JoblibPrefer = Literal["threads", "processes"]
StopReason = Literal["none", "target_score", "no_improvement", "max_iterations"]
LocalOptimType = Literal["none", "grad", "single-var-grad", "perturb"]
Phase = Literal["init", "evolve", "finalize"]
InputArguments = dict[str, Any]
GoalFcn = Union[
    Callable[[AF], F],
    Callable[[AF, InputArguments], F],
    Callable[[AF], float],
    Callable[[AF, InputArguments], float],
]
WrappedGoalFcn = Callable[[AF], F]
ConstraintFcn = GoalFcn
WrappedConstraintFcn = WrappedGoalFcn


def literal_options(literal_type: Any) -> list[Any]:
    """Return the list of allowed values for a typing.Literal type."""
    try:
        return list(get_args(literal_type))
    except Exception:
        return []


def ensure_literal_choice(value: Any, literal_type: Any) -> None:
    """Validate a value against a typing.Literal and raise a helpful error.

    Args:
        value: The provided value
        literal_type: The Literal type alias to validate against
    Raises:
        ValueError: if value not in allowed options
    """
    allowed = literal_options(literal_type)
    if allowed and value not in allowed:
        allowed_str = ", ".join(repr(x) for x in allowed)
        raise ValueError(
            f"Invalid {type(literal_type)}={value!r}. Allowed options: {allowed_str}"
        )


T = TypeVar("T")


def create_from_dict(data: dict[str, Any], cls: type[T]) -> T:
    """Create a dataclass instance from a dictionary.

    Args:
        data: Dictionary containing field values
        cls: Dataclass type to instantiate

    Returns:
        Instance of the dataclass with fields populated from the dictionary
    """
    field_names = {f.name for f in fields(cls)}
    filtered_data = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered_data)


def setup_for_generations(
    config: "IOptimizerConfig",
) -> tuple[tqdm, int, int, Parallel]:
    generation_pbar = trange(config.num_generations, desc="Optimizer generation")
    n_jobs = config.n_jobs
    if n_jobs < 1:
        n_jobs = cpu_count() - 1
    individuals_per_job = max(1, config.population_size // n_jobs)
    parallel = Parallel(n_jobs=n_jobs, prefer=config.joblib_prefer)
    return generation_pbar, individuals_per_job, n_jobs, parallel


ObjectiveMode = Literal["scalar", "map-elites", "pareto"]


@dataclass
class OptimizerRun:
    """Structured return values from a given optimizer run: PSO, ACO, etc"""

    population_values: AF  # (N_generations x N_vars)
    population_solutions: AF  # (1 x N_vars)
    population_outputs: AF | None = None  # (N x n_outputs) tracked outputs, QD add-on


@dataclass
class IOptimizerConfig:
    """Base class for optimizer configurations."""

    name: str = ""
    """The name of the optimizer. This is used for logging purposes."""
    num_generations: int = 50
    """The number of generations to run the optimizer"""
    population_size: int = 30
    """The population size for each generation of the optimizer."""
    solution_archive_size: int = 100
    """Size of solution archive used as memory of good solutions"""
    stop_after_iterations: int = 15
    """Stop after a certain number of iterations. This is used for early stopping if nothing improves"""
    target_score: F = 0.0
    """The target score for the optimizer to achieve. This is used for early stopping."""
    n_jobs: int = 4
    """The number of jobs to use for parallel execution. -1 means use all available cores."""
    joblib_prefer: Literal["threads", "processes"] = "threads"
    """The preferred execution mode for joblib."""
    local_grad_optim: LocalOptimType = "none"
    """Preferred local gradient optimization, ignored by the gradient descent method for obvious reasons"""
    objective_mode: "ObjectiveMode" = "scalar"
    """Objective tracking mode (quality-diversity add-on). ``"scalar"`` (default)
    is the classic single-objective behaviour. ``"map-elites"`` / ``"pareto"``
    tell the optimizer the goal function returns ``(fitness, outputs)`` and that
    the extra outputs should be tracked on each archived solution. See
    QD_PARETO_PLAN.md. Phase 1 wires the tracking only; it does not yet change
    how search selects parents."""
    n_outputs: int = 0
    """Number of tracked objectives/outputs for the Pareto report. When ``> 0``
    (and ``objective_mode`` is not ``"scalar"``) the goal function returns
    ``(fitness, outputs)``: ``fitness`` still drives the search, and the
    ``outputs`` vector is collected for the multi-objective report (see
    ``IOptimizer.qd_report``). ``0`` (default) = plain scalar objective."""
    descriptor_source: Literal["projection", "outputs"] = "projection"
    """MAP-Elites descriptor source. ``"projection"`` (default) derives the
    descriptor from a fixed random projection of the decision vector (works for
    any dimension, scalar objective). ``"outputs"`` uses columns of the goal
    function's returned outputs (requires ``(fitness, outputs)`` returns)."""
    descriptor_dim: int = 2
    """Dimensionality of the MAP-Elites descriptor / projection target."""
    archive_cells: int = 256
    """Number of MAP-Elites (CVT) cells — the archive capacity in map-elites mode."""
    qd_variation: Literal["native", "iso_line"] = "native"
    """MAP-Elites variation operator, applied uniformly to GA/ACO/PSO.
    ``"native"`` (default) keeps each solver's own operator (GA crossover, ACO
    ant sampling, PSO velocity) but sources parents from the diverse CVT archive —
    preserving convergence while the archive curbs premature convergence.
    ``"iso_line"`` replaces it with the shared Iso+LineDD operator for every
    solver (more explorative; can lag on smooth objectives; makes the three
    solvers directly comparable under one variation operator)."""
    iso_sigma: float = 0.01
    """Iso+LineDD isotropic std-dev, as a fraction of each variable's domain."""
    line_sigma: float = 0.2
    """Iso+LineDD directional (line) std-dev (dimensionless)."""

    def __post_init__(self) -> None:
        # When fast mode is on, cap the dominant runtime drivers so the pipeline
        # stays quick. ``min`` never *raises* a value, so small configs (e.g. the
        # map-elites tests) are left untouched. Forcing the joblib backend to
        # threads avoids re-spawning worker *processes* every generation (which
        # re-imports numpy/numba/matplotlib and dominates the wall-clock — ~8x
        # slower here); the per-individual computation, and therefore every
        # assertion, is identical either way. Subclasses that add their own
        # ``__post_init__`` should call ``super().__post_init__()``.
        if fast_tests_enabled():
            self.num_generations = min(self.num_generations, _FAST_MAX_GENERATIONS)
            self.population_size = min(self.population_size, _FAST_MAX_POPULATION)
            self.joblib_prefer = "threads"


@dataclass
class OptimizerResult:
    """Base class for optimizer results.

    Extended to include optional constraint violation information and an unconstrained-best result.
    """

    solution_score: F
    """The score of the best solution found by the optimizer (respecting deck ordering)."""
    solution_vector: AF
    """The best solution found by the optimizer (respecting deck ordering)."""
    solution_history: Optional[AF] = None
    """The history of the best solutions found by the optimizer."""
    stop_reason: StopReason = "none"
    """Whether the optimizer stopped early due to convergence criteria."""
    generations_completed: int = 0
    """Number of generations completed before stopping."""

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(val={self.solution_score}, x={self.solution_vector}, "
            f"stop_reason={self.stop_reason})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __add__(self, other: "OptimizerResult") -> "OptimizerResult":
        if not isinstance(other, OptimizerResult):
            raise ValueError("Cannot add non-OptimizerResult object")
        combined_history = None
        if self.solution_history is not None and other.solution_history is not None:
            combined_history = np.concatenate(
                (self.solution_history, other.solution_history)
            )
        elif self.solution_history is not None:
            combined_history = self.solution_history
        elif other.solution_history is not None:
            combined_history = other.solution_history

        stop_reason: StopReason = "none"
        if self.stop_reason == "target_score" or other.stop_reason == "target_score":
            stop_reason = "target_score"
        elif (
            self.stop_reason == "no_improvement"
            or other.stop_reason == "no_improvement"
        ):
            stop_reason = "no_improvement"
        elif (
            self.stop_reason == "max_iterations"
            or other.stop_reason == "max_iterations"
        ):
            stop_reason = "max_iterations"

        return OptimizerResult(
            solution_score=min(self.solution_score, other.solution_score),
            solution_vector=(
                self.solution_vector
                if self.solution_score <= other.solution_score
                else other.solution_vector
            ),
            solution_history=combined_history,
            stop_reason=stop_reason,
            generations_completed=self.generations_completed
            + other.generations_completed,
        )
