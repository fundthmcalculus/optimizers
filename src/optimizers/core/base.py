from typing import Literal, Optional, TypeVar, Type, get_args
from dataclasses import dataclass, fields
import numpy as np
from joblib import cpu_count, Parallel
from tqdm import trange, tqdm

from .types import AF, F

JoblibPrefer = Literal["threads", "processes"]
StopReason = Literal["none", "target_score", "no_improvement", "max_iterations"]
LocalOptimType = Literal["none", "grad", "single-var-grad", "perturb"]
Phase = Literal["init", "evolve", "finalize"]


def literal_options(literal_type) -> list:
    """Return the list of allowed values for a typing.Literal type."""
    try:
        return list(get_args(literal_type))
    except Exception:
        return []


def ensure_literal_choice(value, literal_type) -> None:
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


def create_from_dict(data: dict, cls: Type[T]) -> T:
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


@dataclass
class OptimizerRun:
    """Structured return values from a given optimizer run: PSO, ACO, etc"""

    population_values: AF  # (N_generations x N_vars)
    population_solutions: AF  # (1 x N_vars)


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
    # Constraint-related outputs (relative violations)
    total_constraint_violation: Optional[F] = None
    ineq_relative_violations: Optional[AF] = None
    eq_relative_violations: Optional[AF] = None
    # Raw constraint results for the reported best solution
    ineq_values: Optional[AF] = None
    eq_values: Optional[AF] = None
    # Best overall result ignoring constraints for user awareness
    unconstrained_best_score: Optional[F] = None
    unconstrained_best_vector: Optional[AF] = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(val={self.solution_score}, x={self.solution_vector}, "
            f"stop_reason={self.stop_reason})"
        )

    def __str__(self):
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
