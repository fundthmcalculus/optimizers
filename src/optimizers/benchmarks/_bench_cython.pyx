# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compiled batched test-function kernels -- the language-level follow-up.

The algorithmic win (vectorizing the goal-function call across a whole
generation instead of once per candidate, see ``optimizers.benchmarks.harness``
and PERF_CONTINUOUS_REPORT.md) is captured entirely in NumPy in
``functions.py``. This module is the optional next step: the *same* formula,
as a single fused ``nogil``/``prange`` loop with no intermediate NumPy
temporaries, for the case where that matters (many small rows, cheap
objective, memory-traffic-bound rather than FLOP-bound). Results are
numerically identical to ``ackley_batch``/``rosenbrock_batch`` (same operation
order, IEEE-754 double precision) -- this is purely a compiled kernel,
not a different algorithm.

Optional extension: if this fails to compile (no C compiler / no Cython at
install time), ``cython_kernels.py`` falls back to the pure-NumPy batch
functions, so nothing in the library depends on this module being built.
"""

import numpy as np
from libc.math cimport exp, sqrt, cos, M_PI
from cython.parallel cimport prange


cdef void _ackley_row(
    double[:, ::1] x, Py_ssize_t i, Py_ssize_t d, double[::1] out_v
) noexcept nogil:
    # A plain (non-prange) accumulation, called from inside the outer prange
    # loop -- keeping the running sums local to this function's own frame
    # sidesteps Cython's automatic OpenMP reduction-variable inference (which
    # otherwise forbids reading an accumulator mid-iteration of the enclosing
    # parallel loop).
    cdef Py_ssize_t j
    cdef double sum_sq = 0.0
    cdef double sum_cos = 0.0
    cdef double a = 20.0
    cdef double b = 0.2
    cdef double c = 2.0 * M_PI
    cdef double e = 2.718281828459045235360287471352662498
    for j in range(d):
        sum_sq += x[i, j] * x[i, j]
        sum_cos += cos(c * x[i, j])
    out_v[i] = (
        -a * exp(-b * sqrt(sum_sq / d))
        - exp(sum_cos / d)
        + a
        + e
    )


def ackley_batch_cy(double[:, ::1] x):
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t d = x.shape[1]
    out = np.empty(n, dtype=np.float64)
    cdef double[::1] out_v = out
    cdef Py_ssize_t i
    for i in prange(n, nogil=True, schedule="static"):
        _ackley_row(x, i, d, out_v)
    return out


cdef void _rosenbrock_row(
    double[:, ::1] x, Py_ssize_t i, Py_ssize_t d, double[::1] out_v
) noexcept nogil:
    cdef Py_ssize_t j
    cdef double total = 0.0
    cdef double term1, term2
    for j in range(d - 1):
        term1 = x[i, j + 1] - x[i, j] * x[i, j]
        term2 = 1.0 - x[i, j]
        total += 100.0 * term1 * term1 + term2 * term2
    out_v[i] = total


def rosenbrock_batch_cy(double[:, ::1] x):
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t d = x.shape[1]
    out = np.empty(n, dtype=np.float64)
    cdef double[::1] out_v = out
    cdef Py_ssize_t i
    for i in prange(n, nogil=True, schedule="static"):
        _rosenbrock_row(x, i, d, out_v)
    return out
