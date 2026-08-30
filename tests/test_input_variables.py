"""The ``InputVariable`` contract, and that both subclasses stay inside it.

``InputDiscreteVariable`` used to widen the contract instead of honouring it:
the base class promises ``float`` for scalars and ``af64`` for populations, and
the discrete subclass returned numpy dtype unions (``float64|float32|float16|
int64|int32|int16``) whose width nothing downstream preserves -- every consumer
writes these into a float64 buffer one call later. mypy called it what it was, a
Liskov violation, and the whole module sat behind a suppression list.

These tests pin the contract itself rather than the annotations, so the
suppression cannot quietly come back.
"""

import numpy as np
import pytest

from optimizers.continuous.variables import (
    InputContinuousVariable,
    InputDiscreteVariable,
    InputVariable,
)
from optimizers.core.random import set_seed
from optimizers.core.types import f64


@pytest.fixture(params=["discrete", "continuous"])
def variable(request) -> InputVariable:
    if request.param == "discrete":
        return InputDiscreteVariable("d", values=np.array([10, 20, 30, 40]))
    return InputContinuousVariable("c", lower_bound=10.0, upper_bound=40.0)


# ------------------------------ the contract --------------------------------


def test_scalar_methods_return_builtin_float(variable):
    """Not ``np.float64``, and above all not a dtype union.

    ``float`` is what the base class declares; a numpy scalar happens to satisfy
    most uses of one, which is exactly why the drift went unnoticed.
    """
    for value in (
        variable.random_value(),
        variable.initial_random_value(),
        variable.perturb_value(20.0),
        variable.range_value(0.5),
        variable.lower_bound,
        variable.upper_bound,
        variable.domain,
        variable.initial_value,
    ):
        assert type(value) is float, f"got {type(value).__name__}"


def test_vectorized_methods_return_float64_arrays(variable):
    """The population methods must produce ``af64``, whatever the input dtype."""
    n = 6
    current = np.full(n, 20.0)
    for array in (
        variable.random_values(current),
        variable.perturb_values(current),
        variable.initial_random_values(n),
        variable.initial_random_velocities(n),
    ):
        assert array.dtype == f64, f"got {array.dtype}"
        assert array.shape == (n,)


def test_archive_weighted_draws_are_also_float64(variable):
    """The ``other_values`` branch is a separate code path in both subclasses."""
    archive = np.array([10.0, 20.0, 20.0, 30.0])
    assert type(variable.random_value(20.0, archive)) is float
    drawn = variable.random_values(np.full(4, 20.0), archive)
    assert drawn.dtype == f64


# --------------------------- discrete-specific ------------------------------


def test_integer_choices_are_normalised_but_still_the_same_choices():
    """Storing float64 must not change *which* values can come out."""
    var = InputDiscreteVariable("i", values=np.array([10, 20, 30, 40]))
    assert var.values.dtype == f64
    drawn = var.initial_random_values(200)
    assert np.all(np.isin(drawn, [10, 20, 30, 40]))
    assert var.lower_bound == 10.0
    assert var.upper_bound == 40.0


def test_values_accepts_a_plain_sequence():
    assert np.array_equal(
        InputDiscreteVariable("l", values=[1, 2, 3]).values, [1.0, 2.0, 3.0]
    )


@pytest.mark.parametrize("pinned", [0, 0.0, 1, 2])
def test_initial_value_is_honoured_including_falsy_ones(pinned):
    """``initial_value or self.random_value()`` silently ignored a pinned 0.

    Zero is the single most likely value to pin, so the truthiness test threw
    away the argument in exactly the case a caller was most likely to use it.
    """
    var = InputDiscreteVariable("d", values=np.array([0, 1, 2]), initial_value=pinned)
    assert var.initial_value == float(pinned)


def test_initial_value_defaults_to_a_draw_from_the_choices():
    var = InputDiscreteVariable("d", values=np.array([5.0, 6.0]))
    assert var.initial_value in (5.0, 6.0)


def test_empty_values_is_rejected_at_construction():
    """Otherwise the failure surfaces later as an opaque numpy error."""
    with pytest.raises(ValueError, match="at least one value"):
        InputDiscreteVariable("e", values=[])


def test_range_value_spans_the_whole_choice_set():
    var = InputDiscreteVariable("d", values=np.array([10, 20, 30, 40]))
    mapped = {var.range_value(p) for p in np.linspace(0.0, 1.0, 41)}
    assert mapped == {10.0, 20.0, 30.0, 40.0}


def test_get_nearest_value_snaps_to_the_closest_choice():
    var = InputDiscreteVariable("d", values=np.array([10, 20, 30, 40]))
    assert var.get_nearest_value(23.0) == 20.0
    assert var.get_nearest_value(-5.0) == 10.0
    assert var.get_nearest_value(1e9) == 40.0


def test_multidimensional_values_is_rejected_at_construction():
    """``ArrayLike`` admits a list of lists; a choice set is one-dimensional.

    Without this the array is accepted, ``rng.choice`` starts picking rows, and
    the failure surfaces somewhere else entirely.
    """
    with pytest.raises(ValueError, match="1-D"):
        InputDiscreteVariable("m", values=[[1, 2], [3, 4]])


def test_values_does_not_alias_the_caller_s_array():
    """Float64 input used to be stored by reference and int input by copy."""
    supplied = np.array([1.0, 2.0, 3.0])
    var = InputDiscreteVariable("a", values=supplied)
    supplied[0] = 99.0
    assert var.values[0] == 1.0


# ------------------- archive weighting follows the choice set ----------------
#
# The weights used to be built from ``np.unique(concatenate(values, archive))``,
# which is sorted and deduplicated, then handed to ``rng.choice`` alongside
# ``self.values`` in construction order. They lined up only by luck.


def _archive_fractions(var, archive, n=6000, seed=1):
    set_seed(seed)
    draws = [var.random_value(archive[0], archive) for _ in range(n)]
    return {x: draws.count(x) / n for x in set(draws)}


def test_weights_follow_the_choice_set_not_its_sort_order():
    """The archive favours 10; an unsorted set used to hand that mass to 30."""
    archive = np.array([10.0, 10.0, 10.0, 20.0])
    frac = _archive_fractions(
        InputDiscreteVariable("u", values=np.array([30.0, 10.0, 20.0])), archive
    )
    # weights are (1 + occurrences): 10 -> 4/7, 20 -> 2/7, 30 -> 1/7
    assert frac[10.0] == pytest.approx(4 / 7, abs=0.03)
    assert frac[20.0] == pytest.approx(2 / 7, abs=0.03)
    assert frac[30.0] == pytest.approx(1 / 7, abs=0.03)


def test_a_sorted_set_is_weighted_the_same_way():
    """Same multiset of choices, same distribution -- order must not matter."""
    archive = np.array([10.0, 10.0, 10.0, 20.0])
    sorted_frac = _archive_fractions(
        InputDiscreteVariable("s", values=np.array([10.0, 20.0, 30.0])), archive
    )
    unsorted_frac = _archive_fractions(
        InputDiscreteVariable("u", values=np.array([30.0, 10.0, 20.0])), archive
    )
    for choice in (10.0, 20.0, 30.0):
        assert sorted_frac[choice] == pytest.approx(unsorted_frac[choice], abs=0.03)


def test_duplicated_choices_no_longer_raise():
    """``rng.choice`` used to reject the mismatched p vector outright."""
    var = InputDiscreteVariable("d", values=np.array([10.0, 10.0, 20.0]))
    assert var.random_value(10.0, np.array([10.0, 20.0])) in (10.0, 20.0)


def test_archive_values_outside_the_choice_set_are_ignored():
    """They used to make the p vector longer than the choice set."""
    var = InputDiscreteVariable("o", values=np.array([10.0, 20.0]))
    assert var.random_value(10.0, np.array([10.0, 99.0])) in (10.0, 20.0)


def test_every_choice_keeps_a_non_zero_probability():
    """The +1 floor: an option absent from the archive must still be reachable."""
    var = InputDiscreteVariable("f", values=np.array([1.0, 2.0, 3.0]))
    archive = np.full(500, 1.0)
    weights = var._archive_weights(archive)
    assert np.all(weights > 0)
    assert weights.sum() == pytest.approx(1.0)


def test_weights_match_the_old_implementation_when_it_was_correct():
    """Sorted + unique + archive drawn from the set: the only working case.

    Pinning this is what makes the fix safe to land -- seeded runs reproduce
    across it rather than merely producing a valid distribution.
    """
    values = np.array([10.0, 20.0, 30.0, 40.0])
    archive = np.array([10.0, 10.0, 10.0, 20.0, 30.0])
    var = InputDiscreteVariable("s", values=values)

    legacy_counts = np.unique(np.concatenate((values, archive)), return_counts=True)[1]
    legacy_p = legacy_counts / legacy_counts.sum()

    assert np.allclose(var._archive_weights(archive), legacy_p)
