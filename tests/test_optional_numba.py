"""numba is an optional extra, not a hard dependency.

It was made optional because its own dependency ``llvmlite`` publishes no macOS
x86_64 wheel, which made this package uninstallable on Intel Macs regardless of
anything in it. These tests pin the resulting behaviour so a future change
cannot quietly make numba required again, or quietly turn a missing accelerator
into a silent 100x slowdown.

The backend truth table is exercised by monkeypatching the availability flags
rather than by uninstalling numba, so it runs in the ordinary test environment.
End-to-end parity across real installs (numba-only, cython-only, neither) is
covered by the CI matrix.
"""

import warnings

import numpy as np
import pytest

from optimizers.combinatorial import strategy as st

# ------------------------------ the njit shim ------------------------------


def test_noop_njit_bare_form_returns_the_function():
    """``@_noop_njit`` with no parentheses must hand back the function itself."""

    def f(x):
        return x * 2

    assert st._noop_njit(f) is f
    assert st._noop_njit(f)(21) == 42


def test_noop_njit_called_form_returns_a_decorator():
    """``@_noop_njit(cache=True)`` -- the spelling the kernels actually use."""

    def f(x):
        return x + 1

    decorated = st._noop_njit(cache=True)(f)
    assert decorated is f
    assert decorated(1) == 2


def test_noop_njit_accepts_numba_signature_options():
    """Extra positional/keyword options must not break the shim."""
    decorated = st._noop_njit("void(f8[:])", nogil=True, fastmath=True)(len)
    assert decorated is len


# --------------------------- backend selection -----------------------------


class _Cfg:
    def __init__(self, backend):
        self.local_search_backend = backend


@pytest.mark.parametrize(
    "backend,has_numba,has_cython,expect_cython",
    [
        # numba present: behaviour is exactly what it was before numba became
        # optional. These four rows are the regression guard on that promise.
        ("numba", True, True, False),
        ("numba", True, False, False),
        ("cython", True, True, True),
        ("cython", True, False, False),
        # numba absent: the compiled extension is preferred over interpreting
        # the kernels, which is orders of magnitude slower, not marginally.
        ("numba", False, True, True),
        ("cython", False, True, True),
        # nothing available: pure Python, and the caller is told.
        ("numba", False, False, False),
        ("cython", False, False, False),
    ],
)
def test_backend_selection(monkeypatch, backend, has_numba, has_cython, expect_cython):
    monkeypatch.setattr(st, "HAS_NUMBA", has_numba)
    monkeypatch.setattr(st, "HAS_CYTHON", has_cython)
    monkeypatch.setattr(st, "_warned_no_accelerator", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert st._use_cython_backend(_Cfg(backend)) is expect_cython


def test_missing_backend_attribute_defaults_to_numba(monkeypatch):
    """Configs predating ``local_search_backend`` must still work."""
    monkeypatch.setattr(st, "HAS_NUMBA", True)
    monkeypatch.setattr(st, "HAS_CYTHON", True)
    assert st._use_cython_backend(object()) is False


# ------------------------------- the warning -------------------------------


def test_warns_once_when_no_accelerator_is_available(monkeypatch):
    """A silent 100x slowdown is worse than a noisy one -- but warn only once."""
    monkeypatch.setattr(st, "HAS_NUMBA", False)
    monkeypatch.setattr(st, "HAS_CYTHON", False)
    monkeypatch.setattr(st, "_warned_no_accelerator", False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            st._use_cython_backend(_Cfg("numba"))

    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime) == 1, "the warning must not repeat once per solve"
    assert "numba" in str(runtime[0].message)


def test_does_not_warn_when_cython_is_available(monkeypatch):
    """Auto-promotion to the compiled kernel is not a degraded state."""
    monkeypatch.setattr(st, "HAS_NUMBA", False)
    monkeypatch.setattr(st, "HAS_CYTHON", True)
    monkeypatch.setattr(st, "_warned_no_accelerator", False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        st._use_cython_backend(_Cfg("numba"))

    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


# --------------------------- results are unchanged --------------------------


def _distances(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 100, (n, 2))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, 0.0)
    return np.ascontiguousarray(d, dtype=np.float64)


def test_kernels_are_plain_functions_under_the_shim():
    """The shim must leave a callable that behaves identically.

    ``njit`` compiles the same Python source, so an un-JITed kernel has to
    produce the same tour -- this asserts that the *undecorated* kernel body is
    reachable and correct, which is what the no-numba install actually runs.
    """
    n = 40
    d = _distances(n, seed=11)
    route = np.ascontiguousarray(np.arange(n, dtype=np.int64))

    # py_func is numba's handle on the original Python function; without numba
    # the kernel already *is* that function.
    kernel = getattr(st._two_opt_kernel, "py_func", st._two_opt_kernel)

    jit_route = route.copy()
    py_route = route.copy()
    st._two_opt_kernel(d, jit_route, -1, -1, True)
    kernel(d, py_route, -1, -1, True)

    assert np.array_equal(jit_route, py_route)
