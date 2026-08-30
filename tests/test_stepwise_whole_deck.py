"""``StepWiseOptimizer(optimize_whole_solution_deck=True)``.

The flag was never set anywhere in the tree, and the branch behind it had six
defects -- it crashed on its first iteration, so none of them could be observed.
The module's mypy suppressions covered the rest. These tests exist mainly so the
flag is exercised at all.

See #151.
"""

import numpy as np
import pytest

from optimizers.archive.cvt import CVTArchive
from optimizers.continuous.step import StepWiseOptimizer, StepWiseOptimizerConfig
from optimizers.continuous.variables import InputContinuousVariable
from optimizers.core.random import set_seed


def _sphere(x):
    """Minimum 0 at (1.5, 1.5)."""
    return float(np.sum((np.asarray(x) - 1.5) ** 2))


def _vars():
    return [
        InputContinuousVariable("x", -5.0, 5.0),
        InputContinuousVariable("y", -5.0, 5.0),
    ]


def _cfg(**kw):
    kw.setdefault("name", "stepwise")
    kw.setdefault("num_generations", 10)
    kw.setdefault("population_size", 6)
    kw.setdefault("solution_archive_size", 6)
    kw.setdefault("n_jobs", 1)
    kw.setdefault("optimize_whole_solution_deck", True)
    return StepWiseOptimizerConfig(**kw)


def _solve(seed=11, **kw):
    set_seed(seed)
    optim = StepWiseOptimizer(config=_cfg(**kw), fcn=_sphere, variables=_vars())
    return optim, optim.solve()


# ------------------------------ it runs at all ------------------------------


def test_whole_deck_path_completes():
    """It used to raise ``ValueError: min() iterable argument is empty``."""
    _, result = _solve()
    assert result.solution_vector is not None
    assert np.isfinite(result.solution_score)


@pytest.mark.parametrize("seed", [11, 12, 13, 14])
def test_score_describes_the_returned_vector(seed):
    """The reported best used to come from a different point than the vector.

    The running best was seeded from the deck's stored score -- which, since the
    deck was never initialised, was whatever ``np.empty`` had left behind. A 0.0
    in that memory became the reported result while the vector sat elsewhere.
    """
    _, result = _solve(seed=seed)
    assert result.solution_score == pytest.approx(_sphere(result.solution_vector))


def test_it_actually_optimizes():
    """The generation loop never called ``local_perturb_optim`` at all.

    It appended the same unchanged value once per generation, so the branch was
    a no-op with a progress bar.
    """
    _, result = _solve()
    assert result.solution_score < 1.0, "should get close to the optimum at (1.5, 1.5)"


def test_history_is_monotone_non_increasing():
    _, result = _solve()
    history = list(result.solution_history)
    assert history == sorted(history, reverse=True)


def test_every_deck_entry_is_written_back():
    """Each entry is optimised in place, so the deck must improve too."""
    optim, _ = _solve()
    best = float(optim.soln_deck.get_best()[1])
    assert best < 1.0


def test_the_deck_is_initialized_not_read_from_np_empty():
    """Stored scores must match their rows once the run is done."""
    optim, _ = _solve()
    for idx in range(len(optim.soln_deck)):
        row, value, _ = optim.soln_deck.get(idx)
        assert float(value) == pytest.approx(_sphere(row))


# ----------------------------- guarded failures -----------------------------


def test_a_cell_based_archive_is_refused():
    """``get``/``set`` are index-addressable operations CVTArchive lacks.

    It used to reach an AttributeError three frames down.
    """
    set_seed(1)
    optim = StepWiseOptimizer(config=_cfg(), fcn=_sphere, variables=_vars())
    optim.soln_deck = CVTArchive(
        num_vars=2,
        lower=np.full(2, -5.0),
        upper=np.full(2, 5.0),
        descriptor_fn=lambda x: np.atleast_2d(x)[:, :1],
        descriptor_dim=1,
        n_cells=4,
    )
    with pytest.raises(NotImplementedError, match="SolutionDeck"):
        optim.solve()


# --------------------- the ordinary path is untouched -----------------------


def test_the_default_path_still_works():
    _, result = _solve(optimize_whole_solution_deck=False)
    assert np.isfinite(result.solution_score)
    assert result.solution_vector is not None
