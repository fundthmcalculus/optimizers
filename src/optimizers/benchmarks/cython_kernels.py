"""Optional compiled backend for the batched benchmark functions.

Mirrors the fallback pattern already used for the TSP local-search kernels
(``optimizers.combinatorial.strategy``): if the ``.pyx`` extension was built,
``HAS_CYTHON`` is ``True`` and the compiled kernels are exposed; otherwise this
module degrades to the pure-NumPy batch functions from ``functions.py``, so
nothing here is a hard dependency.

Installed copies do get the kernels. The ``hatch_build.py`` hook compiles them
into the wheel, and the released wheels are built per-platform with
``OPTIMIZERS_REQUIRE_CYTHON=1``, so a release that lost them fails to build
rather than shipping quietly (issue #132 -- before that hook, hatchling ran no
build step at all and *every* installed copy had ``HAS_CYTHON is False``).

``HAS_CYTHON`` can still legitimately be ``False``: a source install on a
machine with no C compiler degrades on purpose, and so does an sdist install
that cannot compile.
See
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
