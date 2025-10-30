from typing import Literal, Optional, TypeVar, Type
from dataclasses import dataclass, fields
import numpy as np

from .types import AF, F

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


StopReason = Literal["none", "target_score", "no_improvement", "max_iterations"]


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
    target_score: F = 0.0
    """The target score for the optimizer to achieve. This is used for early stopping."""
    n_jobs: int = 4
    """The number of jobs to use for parallel execution. -1 means use all available cores."""
    joblib_prefer: Literal["threads", "processes"] = "threads"
    """The preferred execution mode for joblib."""


@dataclass
class OptimizerResult:
    """Base class for optimizer results."""

    solution_score: F
    """The score of the best solution found by the optimizer."""
    solution_vector: AF
    """The best solution found by the optimizer."""
    solution_history: Optional[AF] = None
    """The history of the best solutions found by the optimizer."""
    stop_reason: StopReason = "none"
    """Whether the optimizer stopped early due to convergence criteria."""
    generations_completed: int = 0
    """Number of generations completed before stopping."""

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
