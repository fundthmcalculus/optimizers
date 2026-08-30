"""The ACO config contract, and the two None paths that used to escape it.

Both bugs here were reachable from the public API and invisible to CI: the
`operator` and `arg-type` codes were disabled for these modules, so mypy saw
them and said nothing.
"""

import numpy as np
import pytest

from optimizers.combinatorial.aco import AntColonyTSP, AntColonyTSPConfig, p_xy
from optimizers.combinatorial.aco_mst import AntColonyMST
from optimizers.combinatorial.aco_mst import p_xy as p_xy_mst


def _distances(n, seed=3):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 100, (n, 2))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, 0.0)
    return np.ascontiguousarray(d, dtype=np.float64)


def _cfg(**kw):
    kw.setdefault("name", "aco")
    kw.setdefault("num_generations", 2)
    kw.setdefault("population_size", 4)
    kw.setdefault("n_jobs", 1)
    return AntColonyTSPConfig(**kw)


# ------------------------- hot_start needs its length -----------------------


def test_hot_start_without_length_is_rejected_at_construction():
    """``solve`` divides by hot_start_length as soon as a route is present.

    Supplying only the route used to reach that division with None and raise
    ``TypeError: unsupported operand type(s) for /: 'int' and 'NoneType'`` from
    inside the generation loop, a long way from the config that caused it.
    """
    with pytest.raises(ValueError, match="must be given together"):
        _cfg(hot_start=np.arange(6, dtype=np.int64))


def test_hot_start_length_without_route_is_rejected():
    """The mirror case: a length nothing ever reads."""
    with pytest.raises(ValueError, match="must be given together"):
        _cfg(hot_start_length=42.0)


def test_neither_is_the_ordinary_case():
    assert _cfg().hot_start is None


def test_hot_start_pair_solves():
    """The combination that used to crash now runs to completion."""
    n = 7
    d = _distances(n)
    route = np.arange(n, dtype=np.int64)
    length = float(sum(d[route[i], route[i + 1]] for i in range(n - 1)))
    cfg = _cfg(hot_start=route, hot_start_length=length, local_optimize=False)
    result = AntColonyTSP(config=cfg, network_routes=d).solve()
    assert result.solution_vector is not None
    assert np.isfinite(result.solution_score)


# ---------------------- a dead end is an error, not None --------------------


@pytest.mark.parametrize("solver", [AntColonyTSP, AntColonyMST])
def test_all_dead_ends_raises_instead_of_returning_none(solver):
    """An all-zero distance matrix makes every desirability zero.

    Every ant then dead-ends on its first step, nothing is ever recorded as the
    best tour, and ``solution_vector`` stayed None -- which would surface
    somewhere else entirely, as an AttributeError on the caller's side.
    """
    d = np.zeros((6, 6), dtype=np.float64)
    cfg = _cfg(local_optimize=False)
    with pytest.raises(ValueError, match="no valid tour"):
        solver(config=cfg, network_routes=d).solve()


# --------------------------- p_xy has one return type -----------------------


@pytest.mark.parametrize("fn,extra", [(p_xy, (0,)), (p_xy_mst, ())])
def test_p_xy_zero_branch_returns_an_array(fn, extra):
    """It used to return the bare int 0 as a sentinel.

    The callers only ever test ``np.sum(p) == 0``, which an all-zero array
    satisfies identically -- and the array is what the signature promised.
    """
    zeros = np.zeros((4, 4))
    allowed = np.ones(4, dtype=bool)
    p = fn(zeros, zeros, allowed, *extra)

    assert isinstance(p, np.ndarray)
    assert np.sum(p) == 0
    assert not np.any(np.isnan(p))
