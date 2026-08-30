"""The ``Archive`` interface — the shared "parent memory" every solver reads from.

Every optimizer (GA / PSO / ACO / GD) delegates its memory of good solutions to a
single object. Historically that object is the :class:`~optimizers.solution_deck.SolutionDeck`
(a scalar, elitist, best-first deck). The quality-diversity / multi-objective
add-on (see ``docs/history/QD_PARETO_PLAN.md``) works by swapping that object for archives with
richer selection semantics — a MAP-Elites grid, a Pareto non-dominated set —
*without* changing the solvers, because they all program against this one
interface.

Phase 1 (this module) only *defines* the interface and documents the surface the
solvers rely on; ``SolutionDeck`` already satisfies it and is aliased as
:data:`ScalarArchive`. Later phases add ``GridArchive`` (MAP-Elites) and
``ParetoArchive`` as drop-in implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..core.types import AF, b8

if TYPE_CHECKING:
    from ..core.base import WrappedGoalFcn
    from ..core.samplers import SamplerType
    from ..core.variables import InputVariables
    from ..solution_deck import InitializationType


@runtime_checkable
class Archive(Protocol):
    """Structural interface for a solver's solution memory.

    The **scalar surface** (``solution_archive`` / ``solution_value`` / ``sort`` /
    ``get_best`` / ``truncate`` / ``__len__``) is what today's solvers consume, so
    any archive must keep it working — for multi-output archives the
    ``solution_value`` is a *scalar surrogate* (e.g. per-cell fitness for
    MAP-Elites, or non-domination rank for Pareto) so rank-based consumers
    (ACO's CDF, GA's tournament, PSO's global best) behave sensibly.

    The **multi-output surface** (``solution_outputs``) is the add-on: the vector
    of tracked outputs recorded alongside each solution, used for the Pareto
    report and (later) MAP-Elites descriptors.
    """

    # --- scalar surface (relied on by every existing solver) ---
    solution_archive: AF  # (N, num_vars) decision vectors
    solution_value: AF  # (N,) scalar fitness / surrogate, ascending = better

    def sort(self) -> None: ...

    def truncate(self, size: int = -1) -> None: ...

    def get_best(self) -> tuple[AF, float, b8]: ...

    def __len__(self) -> int: ...

    # --- population lifecycle (both implementations, and every solver, use
    # these; they were missing from the protocol even though `soln_deck` is
    # exactly one of the two) ---

    def initialize_solution_deck(
        self,
        variables: "InputVariables",
        eval_fcn: "WrappedGoalFcn",
        preserve_percent: float = 0.0,
        init_type: "InitializationType" = "random",
        sampler_type: "SamplerType" = "uniform",
    ) -> None: ...

    def add_generation(
        self,
        solutions: AF,
        values: AF,
        outputs: AF | None = None,
        local_optima: bool = False,
    ) -> None: ...

    # --- multi-output surface (quality-diversity add-on) ---
    solution_outputs: AF | None  # (N, n_outputs) tracked outputs, or None (scalar)
