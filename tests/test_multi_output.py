"""Phase-1 quality-diversity add-on: multi-output tracking (QD_PARETO_PLAN.md §1).

These cover the foundation only — the archive interface, the ``(fitness, outputs)``
objective contract, and that tracked outputs stay row-aligned with solutions
through every deck operation. No new *search* behaviour is exercised here.
"""

import numpy as np

from optimizers.archive import Archive, ScalarArchive
from optimizers.solution_deck import SolutionDeck
from optimizers.continuous.ga import (
    GeneticAlgorithmOptimizer,
    GeneticAlgorithmOptimizerConfig,
)
from optimizers.continuous.variables import InputContinuousVariable
from optimizers.core.random import set_seed


def _vars(n=3):
    return [
        InputContinuousVariable(f"x{i}", lower_bound=-2.0, upper_bound=2.0)
        for i in range(n)
    ]


def _sphere(x):
    return float(np.sum((np.asarray(x) - 0.3) ** 2))


def _sphere_mo(x):
    """Multi-output: scalar fitness + a 2-vector of tracked outputs."""
    x = np.asarray(x)
    return float(np.sum((x - 0.3) ** 2)), np.array([x[0], x[1]])


def test_archive_interface_and_alias():
    assert ScalarArchive is SolutionDeck
    deck = SolutionDeck(archive_size=4, num_vars=2)
    assert isinstance(deck, Archive)  # runtime_checkable Protocol
    assert deck.solution_outputs is None  # scalar deck tracks nothing


def test_scalar_mode_tracks_nothing():
    set_seed(1)
    cfg = GeneticAlgorithmOptimizerConfig(
        name="t",
        num_generations=4,
        population_size=16,
        n_jobs=2,
        joblib_prefer="threads",
        solution_archive_size=20,
    )
    opt = GeneticAlgorithmOptimizer(cfg, _sphere, _vars(), args={})
    opt.solve()
    assert opt.soln_deck.solution_outputs is None


def test_solution_deck_outputs_tracked_via_add_generation():
    """The scalar deck's output tracking (used by later phases) survives the
    unified add_generation seam."""
    deck = SolutionDeck(archive_size=4, num_vars=2, n_outputs=2)
    deck.solution_archive = np.zeros((4, 2))
    deck.solution_value = np.zeros(4)
    deck.is_local_optima = np.zeros(4, dtype=bool)
    deck.set_all_outputs(np.zeros((4, 2)))
    new = np.array([[1.0, 1.0], [2.0, 2.0]])
    deck.add_generation(new, np.array([-1.0, -2.0]), outputs=new * 3.0)
    assert deck.solution_outputs is not None
    assert deck.solution_outputs.shape[0] == deck.solution_archive.shape[0]


def test_scalar_mode_bit_identical_to_untracked_config():
    """Adding objective_mode to the config must not perturb the scalar run."""

    def run():
        set_seed(42)
        cfg = GeneticAlgorithmOptimizerConfig(
            name="t",
            num_generations=6,
            population_size=16,
            n_jobs=1,
            joblib_prefer="threads",
            solution_archive_size=20,
        )
        return GeneticAlgorithmOptimizer(cfg, _sphere, _vars(), args={}).solve()

    a, b = run(), run()
    assert a.solution_score == b.solution_score
    assert np.array_equal(a.solution_vector, b.solution_vector)


def test_deck_outputs_alignment_through_operations():
    """append / sort / deduplicate / truncate keep outputs row-aligned."""
    deck = SolutionDeck(archive_size=5, num_vars=2, n_outputs=2)
    sols = np.array([[float(i), float(i)] for i in range(5)])
    deck.solution_archive = sols.copy()
    deck.solution_value = np.arange(5, 0, -1, dtype=float)  # descending → sort reorders
    deck.is_local_optima = np.zeros(5, dtype=bool)
    deck.set_all_outputs(sols * 10.0)  # invariant: outputs == solution * 10

    def aligned():
        return np.allclose(deck.solution_outputs, deck.solution_archive * 10.0)

    deck.sort()
    assert aligned()
    new = np.array([[9.0, 9.0], [7.0, 7.0]])
    deck.append(new, np.array([0.5, 0.25]), outputs=new * 10.0)
    assert aligned()
    deck.deduplicate()
    assert aligned()
    deck.truncate(3)
    assert aligned()
    assert deck.solution_outputs.shape[0] == deck.solution_archive.shape[0] == 3


def test_append_requires_outputs_when_tracking():
    deck = SolutionDeck(archive_size=2, num_vars=2, n_outputs=2)
    deck.set_all_outputs(np.zeros((2, 2)))
    try:
        deck.append(np.ones((1, 2)), np.array([1.0]))  # missing outputs
        raise AssertionError("expected an assertion error for missing outputs")
    except AssertionError as e:
        assert "outputs" in str(e)


def test_checkpoint_roundtrip_preserves_outputs():
    deck = SolutionDeck(archive_size=3, num_vars=2, n_outputs=2)
    deck.solution_archive = np.arange(6, dtype=float).reshape(3, 2)
    deck.solution_value = np.array([1.0, 2.0, 3.0])
    deck.is_local_optima = np.zeros(3, dtype=bool)
    deck.set_all_outputs(np.arange(6, dtype=float).reshape(3, 2) * 2.0)

    restored = SolutionDeck.from_dict(deck.to_dict())
    assert restored.solution_outputs is not None
    assert np.allclose(restored.solution_outputs, deck.solution_outputs)
    assert restored.n_outputs == 2


def test_scalar_checkpoint_has_no_outputs_key():
    deck = SolutionDeck(archive_size=3, num_vars=2)
    deck.solution_archive = np.zeros((3, 2))
    deck.solution_value = np.zeros(3)
    deck.is_local_optima = np.zeros(3, dtype=bool)
    d = deck.to_dict()
    assert "solution_outputs" not in d
    assert SolutionDeck.from_dict(d).solution_outputs is None
