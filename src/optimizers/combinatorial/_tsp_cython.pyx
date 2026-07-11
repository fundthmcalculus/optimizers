# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compiled TSP local-search kernels (2-opt / 3-opt) — see CYTHON_ANALYSIS.md §3.

These mirror the numba kernels in ``strategy.py`` *exactly* (same moves, same
tie-breaking), so results are identical — the point is a compiled, ahead-of-time
artifact with no JIT warm-up and, crucially, ``nogil`` inner loops. The ``*_batch``
functions run many independent tours in parallel over OpenMP threads
(``prange``) with zero data copying, which is the thread-level parallelism the
pure-Python/GIL-bound versions cannot deliver.

Route arrays are ``np.intp`` (C ``Py_ssize_t``) and the distance matrix is
contiguous ``float64``; the Python wrappers coerce inputs. City indices are the
matrix side ``N``; a route may be length ``N`` or ``N+1`` (a ``back_to_start``
route appends the depot), so ``N`` (loop bounds) and ``route_len`` (in-bounds cap)
are tracked separately.
"""

import numpy as np
from cython.parallel cimport prange


cdef bint _two_opt_row(
    double[:, ::1] D,
    Py_ssize_t[:, ::1] routes,
    Py_ssize_t r,
    Py_ssize_t N,
    Py_ssize_t route_len,
    long num_iterations,
    long nearest_neighbors,
    bint back_to_start,
) noexcept nogil:
    cdef Py_ssize_t ij, jk, k_nn, a, b, ri, start
    cdef Py_ssize_t tmp
    cdef double d1, d2
    cdef bint no_moves = True
    cdef long cur_iter = 0
    while cur_iter < num_iterations or num_iterations == -1:
        cur_iter += 1
        no_moves = True
        start = -1 if back_to_start else 0
        for ij in range(start, N - 2):
            # wraparound is off, so map the ij == -1 (return-to-start) case by hand
            ri = ij if ij >= 0 else route_len - 1
            k_nn = N - 1
            if nearest_neighbors > 0 and ij + nearest_neighbors < k_nn:
                k_nn = ij + nearest_neighbors
            for jk in range(ij + 2, k_nn):
                d1 = (
                    D[routes[r, ri], routes[r, ij + 1]]
                    + D[routes[r, jk], routes[r, jk + 1]]
                )
                d2 = (
                    D[routes[r, ri], routes[r, jk]]
                    + D[routes[r, ij + 1], routes[r, jk + 1]]
                )
                if d1 > d2:
                    a = ij + 1
                    b = jk
                    while a < b:
                        tmp = routes[r, a]
                        routes[r, a] = routes[r, b]
                        routes[r, b] = tmp
                        a += 1
                        b -= 1
                    no_moves = False
        if no_moves:
            break
    return no_moves


cdef bint _three_opt_row(
    double[:, ::1] D,
    Py_ssize_t[:, ::1] routes,
    Py_ssize_t r,
    Py_ssize_t N,
    Py_ssize_t route_len,
    long num_iterations,
    long nearest_neighbors,
) noexcept nogil:
    cdef Py_ssize_t ij, jk, kl, k_nn, l_nn
    cdef Py_ssize_t A, B, C, Dd, E, Fi
    cdef Py_ssize_t a, b, c, d, e, f
    cdef Py_ssize_t route_max = route_len - 1
    cdef double d0, d1, d2, d3, d4, d5, d6, d7, best_len
    cdef int best
    cdef bint no_moves = True
    cdef long cur_iter
    for cur_iter in range(num_iterations):
        no_moves = True
        for ij in range(0, N - 4):
            k_nn = N - 2
            if nearest_neighbors > 0 and ij + nearest_neighbors < k_nn:
                k_nn = ij + nearest_neighbors
            for jk in range(ij + 2, k_nn):
                l_nn = N
                if nearest_neighbors > 0 and jk + nearest_neighbors < l_nn:
                    l_nn = jk + nearest_neighbors
                if l_nn > route_max:
                    l_nn = route_max
                for kl in range(jk + 2, l_nn):
                    A = ij
                    B = ij + 1
                    C = jk
                    Dd = jk + 1
                    E = kl
                    Fi = kl + 1
                    a = routes[r, A]
                    b = routes[r, B]
                    c = routes[r, C]
                    d = routes[r, Dd]
                    e = routes[r, E]
                    f = routes[r, Fi]
                    d0 = D[a, b] + D[c, d] + D[e, f]
                    d1 = D[a, e] + D[d, c] + D[b, f]
                    d2 = D[a, b] + D[c, e] + D[d, f]
                    d3 = D[a, c] + D[b, d] + D[e, f]
                    d4 = D[a, c] + D[b, e] + D[d, f]
                    d5 = D[a, e] + D[d, b] + D[c, f]
                    d6 = D[a, d] + D[e, c] + D[b, f]
                    d7 = D[a, d] + D[e, b] + D[c, f]
                    best = 0
                    best_len = d0
                    if d1 < best_len:
                        best = 1
                        best_len = d1
                    if d2 < best_len:
                        best = 2
                        best_len = d2
                    if d3 < best_len:
                        best = 3
                        best_len = d3
                    if d4 < best_len:
                        best = 4
                        best_len = d4
                    if d5 < best_len:
                        best = 5
                        best_len = d5
                    if d6 < best_len:
                        best = 6
                        best_len = d6
                    if d7 < best_len:
                        best = 7
                        best_len = d7
                    if best == 0:
                        continue
                    elif best == 1:
                        routes[r, E] = b
                        routes[r, Dd] = c
                        routes[r, C] = d
                        routes[r, B] = e
                    elif best == 2:
                        routes[r, E] = d
                        routes[r, Dd] = e
                    elif best == 3:
                        routes[r, C] = b
                        routes[r, B] = c
                    elif best == 4:
                        routes[r, C] = b
                        routes[r, B] = c
                        routes[r, E] = d
                        routes[r, Dd] = e
                    elif best == 5:
                        routes[r, E] = b
                        routes[r, Dd] = c
                        routes[r, B] = d
                        routes[r, C] = e
                    elif best == 6:
                        routes[r, Dd] = b
                        routes[r, E] = c
                        routes[r, C] = d
                        routes[r, B] = e
                    elif best == 7:
                        routes[r, Dd] = b
                        routes[r, E] = c
                        routes[r, B] = d
                        routes[r, C] = e
                    no_moves = False
        if no_moves:
            break
    return no_moves


# --------------------------- Python entry points ---------------------------

cdef inline object _as_routes2d(route):
    """Coerce a 1-D route to a C-contiguous (1, L) intp array (a view when possible)."""
    r = np.ascontiguousarray(route, dtype=np.intp)
    return r.reshape(1, r.shape[0])


def two_opt(distances, route, long num_iterations=-1, long nearest_neighbors=-1,
            bint back_to_start=True):
    """In-place 2-opt on a single route. Returns ``(route, no_moves)``.

    ``route`` is returned because coercion to ``intp``/contiguous may copy; use
    the returned array. Matches ``strategy._two_opt_kernel`` move-for-move.
    """
    cdef double[:, ::1] D = np.ascontiguousarray(distances, dtype=np.float64)
    R2 = _as_routes2d(route)
    cdef Py_ssize_t[:, ::1] Rv = R2
    cdef Py_ssize_t N = D.shape[0]
    cdef Py_ssize_t route_len = Rv.shape[1]
    cdef bint no_moves
    with nogil:
        no_moves = _two_opt_row(D, Rv, 0, N, route_len, num_iterations,
                                nearest_neighbors, back_to_start)
    return R2[0], no_moves


def three_opt(distances, route, long num_iterations=-1, long nearest_neighbors=-1):
    """In-place 3-opt on a single route. Returns ``(route, no_moves)``."""
    cdef double[:, ::1] D = np.ascontiguousarray(distances, dtype=np.float64)
    R2 = _as_routes2d(route)
    cdef Py_ssize_t[:, ::1] Rv = R2
    cdef Py_ssize_t N = D.shape[0]
    cdef Py_ssize_t route_len = Rv.shape[1]
    cdef bint no_moves
    with nogil:
        no_moves = _three_opt_row(D, Rv, 0, N, route_len, num_iterations,
                                  nearest_neighbors)
    return R2[0], no_moves


def two_opt_batch(distances, routes, long num_iterations=-1,
                  long nearest_neighbors=-1, bint back_to_start=True):
    """2-opt on many equal-length routes in parallel (OpenMP ``prange``, ``nogil``).

    ``routes`` is ``(m, L)``; each row is optimized independently and in place.
    Returns the coerced ``(m, L)`` intp array. This is the thread-level, zero-copy
    parallelism Cython enables (CYTHON_ANALYSIS.md §3a).
    """
    cdef double[:, ::1] D = np.ascontiguousarray(distances, dtype=np.float64)
    Rs = np.ascontiguousarray(routes, dtype=np.intp)
    cdef Py_ssize_t[:, ::1] Rv = Rs
    cdef Py_ssize_t m = Rv.shape[0]
    cdef Py_ssize_t N = D.shape[0]
    cdef Py_ssize_t route_len = Rv.shape[1]
    cdef Py_ssize_t i
    with nogil:
        for i in prange(m, schedule="dynamic"):
            _two_opt_row(D, Rv, i, N, route_len, num_iterations,
                         nearest_neighbors, back_to_start)
    return Rs


def three_opt_batch(distances, routes, long num_iterations=-1,
                    long nearest_neighbors=-1):
    """3-opt on many equal-length routes in parallel (OpenMP ``prange``, ``nogil``)."""
    cdef double[:, ::1] D = np.ascontiguousarray(distances, dtype=np.float64)
    Rs = np.ascontiguousarray(routes, dtype=np.intp)
    cdef Py_ssize_t[:, ::1] Rv = Rs
    cdef Py_ssize_t m = Rv.shape[0]
    cdef Py_ssize_t N = D.shape[0]
    cdef Py_ssize_t route_len = Rv.shape[1]
    cdef Py_ssize_t i
    with nogil:
        for i in prange(m, schedule="dynamic"):
            _three_opt_row(D, Rv, i, N, route_len, num_iterations,
                           nearest_neighbors)
    return Rs


def check_path_distance(distances, route, bint back_to_start=True):
    """Total tour length (``nogil`` reduction). Matches ``base.check_path_distance``."""
    cdef double[:, ::1] D = np.ascontiguousarray(distances, dtype=np.float64)
    R = np.ascontiguousarray(route, dtype=np.intp)
    cdef Py_ssize_t[::1] Rv = R
    cdef Py_ssize_t n = Rv.shape[0]
    cdef Py_ssize_t i
    cdef double total = 0.0
    with nogil:
        for i in range(n - 1):
            total += D[Rv[i], Rv[i + 1]]
        if back_to_start and n >= 1:
            total += D[Rv[n - 1], 0]
    return total
