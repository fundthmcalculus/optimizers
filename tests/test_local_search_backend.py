"""Backend selection for the TSP local search.

There is one accelerator now: the compiled Cython extension. numba used to
provide a second, and was removed once the Cython kernels measured faster on
every kernel and size (2-opt 1.8-2.8x, 3-opt 1.1-1.4x, LK 1.1-1.6x) while
costing 140 MB of install and ~300 ms of import.

That leaves a sharper cliff than before: with no compiled extension the kernels
run as interpreted Python, which is ~900x slower at N=200 rather than the ~2x
numba used to cost. These tests pin the two things that protect against it --
that the fallback is never silent, and that ``"python"`` means what it says.

Availability is monkeypatched rather than uninstalled, so this runs in the
ordinary test environment. Bit-identical results between the two backends are
asserted in test_tsp_cython.py.
"""

import warnings

import numpy as np
import pytest

from optimizers.combinatorial import strategy as st


class _Cfg:
    def __init__(self, backend):
        self.local_search_backend = backend


# --------------------------- backend selection -----------------------------


@pytest.mark.parametrize(
    "backend,has_cython,expect_cython",
    [
        ("cython", True, True),
        ("cython", False, False),  # falls back, and warns (below)
        ("python", True, False),  # explicit opt-out is honoured even if built
        ("python", False, False),
    ],
)
def test_backend_selection(monkeypatch, backend, has_cython, expect_cython):
    monkeypatch.setattr(st, "HAS_CYTHON", has_cython)
    monkeypatch.setattr(st, "_warned_no_accelerator", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert st._use_cython_backend(_Cfg(backend)) is expect_cython


def test_missing_backend_attribute_defaults_to_cython(monkeypatch):
    """Configs predating ``local_search_backend`` must get the fast path."""
    monkeypatch.setattr(st, "HAS_CYTHON", True)
    assert st._use_cython_backend(object()) is True


# ------------------------- the legacy numba alias --------------------------


def test_legacy_numba_value_maps_to_cython(monkeypatch):
    """Old configs and pickles still say "numba"; they meant "go fast".

    Mapping it to the interpreted kernels instead would turn a stale config
    value into a silent ~900x slowdown, which is the opposite of what whoever
    wrote it wanted.
    """
    monkeypatch.setattr(st, "HAS_CYTHON", True)
    monkeypatch.setattr(st, "_warned_legacy_numba", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert st._use_cython_backend(_Cfg("numba")) is True


def test_legacy_numba_value_warns_once(monkeypatch):
    monkeypatch.setattr(st, "HAS_CYTHON", True)
    monkeypatch.setattr(st, "_warned_legacy_numba", False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            st._use_cython_backend(_Cfg("numba"))

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, "must not warn once per solve"
    assert "numba" in str(deprecations[0].message)


# ------------------------------- the warning -------------------------------


@pytest.mark.parametrize("backend", ["cython", "numba"])
def test_warns_when_falling_back_to_interpreted(monkeypatch, backend):
    """Falling back is never silent -- for either spelling that asks to go fast.

    Parametrised deliberately: an earlier version returned early for one of the
    spellings and skipped the warning on exactly the path where the caller had
    asked for a compiled backend and not got one.
    """
    monkeypatch.setattr(st, "HAS_CYTHON", False)
    monkeypatch.setattr(st, "_warned_no_accelerator", False)
    monkeypatch.setattr(st, "_warned_legacy_numba", True)  # isolate the one warning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            st._use_cython_backend(_Cfg(backend))

    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime) == 1, "must not warn once per solve"
    assert "900x" in str(runtime[0].message)


def test_explicit_python_backend_does_not_warn(monkeypatch):
    """``"python"`` was chosen, not fallen back to -- warning would be noise."""
    monkeypatch.setattr(st, "HAS_CYTHON", False)
    monkeypatch.setattr(st, "_warned_no_accelerator", False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        st._use_cython_backend(_Cfg("python"))

    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


def test_no_warning_when_the_extension_is_available(monkeypatch):
    monkeypatch.setattr(st, "HAS_CYTHON", True)
    monkeypatch.setattr(st, "_warned_no_accelerator", False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        st._use_cython_backend(_Cfg("cython"))

    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


# ---------------------------- numba is really gone --------------------------


def test_strategy_does_not_import_numba():
    """A regression here would re-add 140 MB and the macOS x86_64 wheel problem."""
    assert not hasattr(st, "njit")
    assert not hasattr(st, "HAS_NUMBA")


def test_kernels_are_plain_python_functions():
    """No JIT wrapper left on them -- ``py_func`` is a numba artefact."""
    for kernel in (st._two_opt_kernel, st._three_opt_kernel, st._lk_kernel):
        assert not hasattr(kernel, "py_func")
        assert callable(kernel)


# ------------------------- the interpreted kernels work ---------------------


def _distances(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 100, (n, 2))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, 0.0)
    return np.ascontiguousarray(d, dtype=np.float64)


def test_interpreted_two_opt_still_improves_a_tour():
    """The fallback has to be correct, not merely present.

    Kept small (N=60): this is the interpreted path, so it is the one place in
    the suite where problem size costs real seconds.
    """
    n = 60
    d = _distances(n, seed=11)
    route = np.ascontiguousarray(np.arange(n, dtype=np.int64))

    def tour_len(r):
        return float(sum(d[r[i], r[i + 1]] for i in range(len(r) - 1)))

    before = tour_len(route)
    improved = route.copy()
    st._two_opt_kernel(d, improved, -1, -1, False)

    assert sorted(improved.tolist()) == list(range(n)), "must stay a permutation"
    assert tour_len(improved) < before
