"""Optional compiled backend for the batched benchmark functions.

Mirrors the fallback pattern already used for the TSP local-search kernels
(``optimizers.combinatorial.strategy``): if the ``.pyx`` extension was built
(``python setup.py build_ext --inplace`` / a wheel install with a working C
compiler), ``HAS_CYTHON`` is ``True`` and the compiled kernels are exposed;
otherwise this module degrades to the pure-NumPy batch functions from
``functions.py``, so nothing here is a hard dependency. See
docs/history/PERF_CONTINUOUS_REPORT.md for the measured Cython-vs-NumPy comparison -- the
honest finding is that it is a modest, not dramatic, win on top of the
algorithmic (batching) speedup.
"""

import numpy as np

from ..core.types import AF
from .functions import ackley_batch, rosenbrock_batch

try:
    from . import _bench_cython  # type: ignore[attr-defined]

    HAS_CYTHON = True
except ImportError:  # pragma: no cover - exercised only in unbuilt checkouts
    _bench_cython = None
    HAS_CYTHON = False


def ackley_batch_cy(x: AF) -> AF:
    if HAS_CYTHON:
        return np.asarray(
            _bench_cython.ackley_batch_cy(np.ascontiguousarray(x, dtype=np.float64))
        )
    return ackley_batch(x)


def rosenbrock_batch_cy(x: AF) -> AF:
    if HAS_CYTHON:
        return np.asarray(
            _bench_cython.rosenbrock_batch_cy(np.ascontiguousarray(x, dtype=np.float64))
        )
    return rosenbrock_batch(x)
