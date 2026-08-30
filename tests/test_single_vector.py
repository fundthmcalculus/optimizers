"""``single_vector`` -- the one place the list-of-routes assumption is checked.

``Solution`` is ``AF | AI | list[AI]``. The list member is the multi-TSP solver
alone; every other solver returns one vector and every consumer expects one.
Consumers used to take that on trust, which is what let the mypy baseline hide
six errors behind an ``arg-type`` suppression.
"""

import numpy as np
import pytest

from optimizers.core.base import OptimizerResult, single_vector


def _result(vector):
    return OptimizerResult(solution_score=1.0, solution_vector=vector)


def test_returns_a_float_vector_unchanged():
    v = np.array([1.0, 2.0, 3.0])
    assert single_vector(_result(v)) is v


def test_returns_an_integer_route_unchanged():
    v = np.arange(5, dtype=np.int64)
    assert single_vector(_result(v)) is v


def test_a_multi_tsp_result_is_rejected():
    """np.asarray would have built a 2-D array here and carried on."""
    routes = [np.arange(4, dtype=np.int64), np.arange(4, dtype=np.int64)]
    with pytest.raises(TypeError, match="list of 2"):
        single_vector(_result(routes))


def test_the_error_names_the_length():
    routes = [np.arange(3, dtype=np.int64)] * 7
    with pytest.raises(TypeError, match="list of 7"):
        single_vector(_result(routes))
