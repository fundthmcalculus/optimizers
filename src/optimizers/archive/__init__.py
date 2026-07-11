"""Solution archives — the swappable "parent memory" behind every solver.

``Archive`` is the shared interface (see :mod:`optimizers.archive.base`).
``ScalarArchive`` is the classic single-objective, best-first deck (today's
:class:`~optimizers.solution_deck.SolutionDeck`) and remains the default. The
quality-diversity / multi-objective implementations (``GridArchive`` for
MAP-Elites, ``ParetoArchive``) land in later phases of ``QD_PARETO_PLAN.md``.
"""

from .base import Archive
from .cvt import CVTArchive
from .descriptor import RandomProjectionDescriptor, OutputsDescriptor
from .variation import iso_line_dd, iso_line_offspring
from .metrics import (
    non_dominated_mask,
    pareto_front,
    hypervolume,
    qd_score,
    QDReport,
)
from ..solution_deck import SolutionDeck

# The current scalar deck IS the scalar archive; alias it under the forward-
# looking name so callers can migrate without a rename churn.
ScalarArchive = SolutionDeck

__all__ = [
    "Archive",
    "ScalarArchive",
    "SolutionDeck",
    "CVTArchive",
    "RandomProjectionDescriptor",
    "OutputsDescriptor",
    "iso_line_dd",
    "iso_line_offspring",
    "non_dominated_mask",
    "pareto_front",
    "hypervolume",
    "qd_score",
    "QDReport",
]
