from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numba import njit

from ..core import IOptimizerConfig
from .base import TSPBase, CombinatoricsResult, check_path_distance
from ..core.types import AI, F, AF

# Optional compiled backend (2-opt / 3-opt). Built via `setup.py build_ext`
# (see CYTHON_ANALYSIS.md); if it isn't compiled, the numba kernels are used, so
# a plain source checkout still runs without a build step.
try:
    # Compiled extension: no source/stub for mypy to read (built ahead-of-time),
    # so the submodule attribute is invisible to static analysis whether or not
    # the .so is present — silence just this optional-backend import.
    from . import _tsp_cython  # type: ignore[attr-defined]

    HAS_CYTHON = True
except ImportError:  # pragma: no cover - exercised only in unbuilt checkouts
    _tsp_cython = None
    HAS_CYTHON = False

LocalSearchBackend = Literal["numba", "cython"]


@dataclass
class TwoOptTSPConfig(IOptimizerConfig):
    back_to_start: bool = True
    """Whether to return to the start node"""
    num_iterations: int = -1
    """Number of iterations to run"""
    nearest_neighbors: int = -1
    """Only check the next nodes, which makes this O(nk), but lower chance of finding crossovers"""
    local_search_backend: LocalSearchBackend = "numba"
    """Which compiled 2-opt/3-opt kernel to use. ``"numba"`` (default) is the JIT
    kernel; ``"cython"`` uses the ahead-of-time compiled ``nogil`` extension when
    it has been built (falls back to numba otherwise). Results are identical."""


def _swap_segment(ij: int, jk: int, new_route: AI) -> AI:
    ij += 1
    while ij < jk:
        temp = new_route[ij]
        new_route[ij] = new_route[jk]
        new_route[jk] = temp
        ij += 1
        jk -= 1
    return new_route


# --- numba kernels for the local-search hot loops (report item #13) -----------
# These are the classic numba sweet spot: tight scalar loops over a distance
# matrix. cache=True persists the compiled code to disk so the ~1s compile is
# paid once, not per run. The logic mirrors the pure-Python versions exactly so
# results are unchanged; the speedup grows with N (2-opt is O(n^2) per pass,
# 3-opt O(n^3)).


@njit(cache=True)
def _two_opt_kernel(
    distances: AF,
    route: AI,
    num_iterations: int,
    nearest_neighbors: int,
    back_to_start: bool,
) -> bool:
    # N is the city count (distance-matrix side), NOT the route length: a route
    # with back_to_start appends a return-to-depot node the loops must not touch.
    N = distances.shape[0]
    no_moves = True
    cur_iter = 0
    while cur_iter < num_iterations or num_iterations == -1:
        cur_iter += 1
        no_moves = True
        start = -1 if back_to_start else 0
        for ij in range(start, N - 2):
            k_nn = N - 1
            if nearest_neighbors > 0 and ij + nearest_neighbors < k_nn:
                k_nn = ij + nearest_neighbors
            for jk in range(ij + 2, k_nn):
                d1 = (
                    distances[route[ij], route[ij + 1]]
                    + distances[route[jk], route[jk + 1]]
                )
                d2 = (
                    distances[route[ij], route[jk]]
                    + distances[route[ij + 1], route[jk + 1]]
                )
                if d1 > d2:
                    # Reverse route[ij+1 .. jk] in place (== _swap_segment).
                    a = ij + 1
                    b = jk
                    while a < b:
                        tmp = route[a]
                        route[a] = route[b]
                        route[b] = tmp
                        a += 1
                        b -= 1
                    no_moves = False
        if no_moves:
            break
    return no_moves


@njit(cache=True)
def _three_opt_kernel(
    distances: AF, route: AI, num_iterations: int, nearest_neighbors: int
) -> bool:
    # N is the city count (see _two_opt_kernel), not the route length.
    N = distances.shape[0]
    # kl+1 must stay in-bounds. The old Python loop used ``l_nn = N`` and only
    # avoided an IndexError because back_to_start appends a depot node (route
    # length N+1); with no appended node it crashed. Capping by route length is
    # a no-op for the well-defined back_to_start case and prevents njit (which
    # skips bounds checks) from reading out of bounds otherwise.
    route_max = route.shape[0] - 1
    no_moves = True
    for _cur_iter in range(num_iterations):
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
                    D = jk + 1
                    E = kl
                    Fi = kl + 1
                    a = route[A]
                    b = route[B]
                    c = route[C]
                    d = route[D]
                    e = route[E]
                    f = route[Fi]
                    # The 8 reconnection lengths (no per-iteration allocation).
                    d0 = distances[a, b] + distances[c, d] + distances[e, f]
                    d1 = distances[a, e] + distances[d, c] + distances[b, f]
                    d2 = distances[a, b] + distances[c, e] + distances[d, f]
                    d3 = distances[a, c] + distances[b, d] + distances[e, f]
                    d4 = distances[a, c] + distances[b, e] + distances[d, f]
                    d5 = distances[a, e] + distances[d, b] + distances[c, f]
                    d6 = distances[a, d] + distances[e, c] + distances[b, f]
                    d7 = distances[a, d] + distances[e, b] + distances[c, f]
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
                        route[E] = b
                        route[D] = c
                        route[C] = d
                        route[B] = e
                    elif best == 2:
                        route[E] = d
                        route[D] = e
                    elif best == 3:
                        route[C] = b
                        route[B] = c
                    elif best == 4:
                        route[C] = b
                        route[B] = c
                        route[E] = d
                        route[D] = e
                    elif best == 5:
                        route[E] = b
                        route[D] = c
                        route[B] = d
                        route[C] = e
                    elif best == 6:
                        route[D] = b
                        route[E] = c
                        route[C] = d
                        route[B] = e
                    elif best == 7:
                        route[D] = b
                        route[E] = c
                        route[B] = d
                        route[C] = e
                    no_moves = False
        if no_moves:
            break
    return no_moves


def candidate_lists(distances: AF, k: int) -> AI:
    """``k`` nearest neighbours per city (self excluded), as an ``(N, k)`` array.

    Lin-Kernighan only proposes new edges to a city's nearest neighbours, which
    turns each move search from O(N) into O(k) and is what makes it scale.
    """
    n = distances.shape[0]
    k = max(1, min(k, n - 1))
    dm = np.asarray(distances, dtype=np.float64).copy()
    np.fill_diagonal(dm, np.inf)
    return np.argsort(dm, axis=1)[:, :k].astype(np.int64)


@njit(cache=True)
def _lk_relocate(tour, pos, s, seg_len, anchor_pos, reverse, n):
    # Move the (non-wrapping) segment tour[s .. s+seg_len-1] to sit immediately
    # after the city currently at ``anchor_pos``, optionally reversed. Rebuilds
    # the tour array in order (O(n)); only ever called on an improving move.
    seg = np.empty(seg_len, dtype=tour.dtype)
    for t in range(seg_len):
        seg[t] = tour[s + seg_len - 1 - t] if reverse else tour[s + t]
    anchor_city = tour[anchor_pos]
    new = np.empty(n, dtype=tour.dtype)
    idx = 0
    for x in range(n):
        if s <= x <= s + seg_len - 1:
            continue
        new[idx] = tour[x]
        idx += 1
        if tour[x] == anchor_city:
            for t in range(seg_len):
                new[idx] = seg[t]
                idx += 1
    for x in range(n):
        tour[x] = new[x]
        pos[tour[x]] = x


@njit(cache=True)
def _lk_kernel(distances, tour, cand, max_passes):
    """Lin-Kernighan-style local search on a cyclic tour (mutated in place).

    A variable-neighbourhood descent over two move families, both restricted to
    near-neighbour candidates: (1) **2-opt** reversals — realize a new edge
    ``(a, c)`` for a near neighbour ``c`` — and (2) **Or-opt** relocations of
    length-1/2/3 segments to sit beside a near neighbour (forward or reversed).
    The combined neighbourhood escapes the pure-2-opt local optima that trap
    ``TwoOptTSP``/``ThreeOptTSP``, so LK typically returns shorter tours.
    Returns the number of improving moves applied.
    """
    n = tour.shape[0]
    k = cand.shape[1]
    eps = 1e-9
    pos = np.empty(n, dtype=np.int64)
    for i in range(n):
        pos[tour[i]] = i
    total_moves = 0
    for _p in range(max_passes):
        improved = False
        # ---------------- candidate-list 2-opt (both directions) -------------
        for i in range(n):
            a = tour[i]
            for ddir in range(2):
                b = tour[(i + 1) % n] if ddir == 0 else tour[(i - 1 + n) % n]
                d_ab = distances[a, b]
                for ci in range(k):
                    c = cand[a, ci]
                    d_ac = distances[a, c]
                    if d_ac >= d_ab:
                        break  # candidates sorted: no positive first-gain beyond
                    j = pos[c]
                    d = tour[(j + 1) % n] if ddir == 0 else tour[(j - 1 + n) % n]
                    if c == b or d == a:
                        continue
                    if d_ab + distances[c, d] - d_ac - distances[b, d] <= eps:
                        continue
                    # reverse the interior so edges (a,c) + (b,d) are realized
                    lo = i if ddir == 0 else (i - 1 + n) % n
                    hi = j if ddir == 0 else (j - 1 + n) % n
                    p = min(lo, hi) + 1
                    q = max(lo, hi)
                    while p < q:
                        tmp = tour[p]
                        tour[p] = tour[q]
                        tour[q] = tmp
                        pos[tour[p]] = p
                        pos[tour[q]] = q
                        p += 1
                        q -= 1
                    improved = True
                    total_moves += 1
                    break
        # ---------------- Or-opt: relocate length 1..3 segments --------------
        for seg_len in range(1, 4):
            for s in range(0, n - seg_len):
                s0 = tour[s]
                sl = tour[s + seg_len - 1]
                prev = tour[(s - 1 + n) % n]
                nxt = tour[(s + seg_len) % n]
                if prev == sl or nxt == s0:
                    continue
                removed = (
                    distances[prev, s0] + distances[sl, nxt] - distances[prev, nxt]
                )
                best_gain = eps
                best_pos = -1
                best_rev = False
                for src in range(2):
                    anchor = s0 if src == 0 else sl
                    for ci in range(cand.shape[1]):
                        c = cand[anchor, ci]
                        p = pos[c]
                        if s - 1 <= p <= s + seg_len - 1:
                            continue
                        cnext = tour[(p + 1) % n]
                        if cnext == s0:
                            continue
                        base = distances[c, cnext]
                        gain_f = removed - (
                            distances[c, s0] + distances[sl, cnext] - base
                        )
                        if gain_f > best_gain:
                            best_gain = gain_f
                            best_pos = p
                            best_rev = False
                        gain_r = removed - (
                            distances[c, sl] + distances[s0, cnext] - base
                        )
                        if gain_r > best_gain:
                            best_gain = gain_r
                            best_pos = p
                            best_rev = True
                if best_pos >= 0:
                    _lk_relocate(tour, pos, s, seg_len, best_pos, best_rev, n)
                    improved = True
                    total_moves += 1
        if not improved:
            break
    return total_moves


class TwoOptTSP(TSPBase):
    def __init__(
        self,
        config: TwoOptTSPConfig,
        network_routes: Optional[AF] = None,
        city_locations: Optional[AF] = None,
        initial_route: Optional[AI] = None,
        initial_value: Optional[F] = None,
    ):
        super().__init__(network_routes, city_locations)
        self.config = config
        self.initial_value = initial_value
        self.initial_route = initial_route

    def solve(self) -> CombinatoricsResult:
        N, new_route = self.setup_local_search()
        # Compiled 2-opt (report item #13); logic identical across backends.
        new_route = np.ascontiguousarray(new_route)
        distances = np.ascontiguousarray(self.network_routes, dtype=np.float64)
        if (
            getattr(self.config, "local_search_backend", "numba") == "cython"
            and HAS_CYTHON
        ):
            new_route, no_moves = _tsp_cython.two_opt(
                distances,
                new_route,
                self.config.num_iterations,
                self.config.nearest_neighbors,
                self.config.back_to_start,
            )
            no_moves = bool(no_moves)
        else:
            no_moves = bool(
                _two_opt_kernel(
                    distances,
                    new_route,
                    self.config.num_iterations,
                    self.config.nearest_neighbors,
                    self.config.back_to_start,
                )
            )

        history = [
            check_path_distance(
                self.network_routes, new_route, self.config.back_to_start
            )
        ]

        return CombinatoricsResult(
            optimal_path=np.array(new_route),
            optimal_value=history[-1],
            value_history=np.array(history),
            stop_reason="no_improvement" if no_moves else "max_iterations",
        )

    def setup_local_search(self) -> tuple[int, AI]:
        if self.initial_route is None or self.initial_value is None:
            # Use the nearest neighbor
            nn_config = NearestNeighborTSPConfig(
                back_to_start=self.config.back_to_start, name=self.config.name
            )
            nn_solver = NearestNeighborTSP(
                nn_config,
                network_routes=self.network_routes,
                city_locations=self.city_locations,
            )
            solution = nn_solver.solve()
            self.initial_route = solution.optimal_path
            self.initial_value = solution.optimal_value
        assert self.initial_route is not None
        new_route = self.initial_route.copy()
        N = self.network_routes.shape[0]
        return N, new_route


class ThreeOptTSP(TwoOptTSP):
    def __init__(
        self,
        config: TwoOptTSPConfig,
        initial_route: Optional[AI] = None,
        initial_value: Optional[F] = None,
        network_routes: Optional[AF] = None,
        city_locations: Optional[AF] = None,
    ):
        super().__init__(
            config,
            initial_route=initial_route,
            initial_value=initial_value,
            network_routes=network_routes,
            city_locations=city_locations,
        )

    def solve(self) -> CombinatoricsResult:
        N, new_route = self.setup_local_search()
        # Compiled 3-opt (report item #13); num_iterations=-1 is a no-op pass.
        new_route = np.ascontiguousarray(new_route)
        distances = np.ascontiguousarray(self.network_routes, dtype=np.float64)
        if (
            getattr(self.config, "local_search_backend", "numba") == "cython"
            and HAS_CYTHON
        ):
            new_route, no_moves = _tsp_cython.three_opt(
                distances,
                new_route,
                self.config.num_iterations,
                self.config.nearest_neighbors,
            )
            no_moves = bool(no_moves)
        else:
            no_moves = bool(
                _three_opt_kernel(
                    distances,
                    new_route,
                    self.config.num_iterations,
                    self.config.nearest_neighbors,
                )
            )

        history = [
            check_path_distance(
                self.network_routes, new_route, self.config.back_to_start
            )
        ]

        return CombinatoricsResult(
            optimal_path=np.array(new_route),
            optimal_value=history[-1],
            value_history=np.array(history),
            stop_reason="no_improvement" if no_moves else "max_iterations",
        )


@dataclass
class LinKernighanTSPConfig(TwoOptTSPConfig):
    candidate_k: int = 8
    """Number of nearest-neighbour candidates each move search considers."""
    max_passes: int = 1000
    """Cap on full improvement passes (LK runs to convergence well before this)."""


class LinKernighanTSP(TwoOptTSP):
    """Lin-Kernighan-style local search (2-opt + Or-opt over candidate lists).

    A stronger local optimum than ``TwoOptTSP``/``ThreeOptTSP`` because its move
    set spans both segment *reversal* (2-opt) and segment *relocation* (Or-opt),
    escaping the reversal-only local optima the others settle into. Reuses
    ``setup_local_search`` (nearest-neighbour start when no ``initial_route``);
    optimizes the closed cyclic tour over all cities.
    """

    config: LinKernighanTSPConfig

    def solve(self) -> CombinatoricsResult:
        _, new_route = self.setup_local_search()
        route = np.ascontiguousarray(new_route)
        N = self.network_routes.shape[0]
        # Work on the cyclic tour of the N cities; drop a trailing depot copy if
        # the initial route appended one (NN does, for back_to_start).
        if route.shape[0] == N + 1 and route[0] == route[-1]:
            tour = np.ascontiguousarray(route[:-1], dtype=np.int64)
        else:
            tour = np.ascontiguousarray(route[:N], dtype=np.int64)

        distances = np.ascontiguousarray(self.network_routes, dtype=np.float64)
        cand = candidate_lists(distances, self.config.candidate_k)
        max_passes = (
            self.config.num_iterations
            if self.config.num_iterations > 0
            else self.config.max_passes
        )
        if (
            getattr(self.config, "local_search_backend", "numba") == "cython"
            and HAS_CYTHON
        ):
            tour, n_moves = _tsp_cython.lin_kernighan(distances, tour, cand, max_passes)
        else:
            n_moves = _lk_kernel(distances, tour, cand, max_passes)

        # Or-opt can relocate the depot (city 0) off index 0. Rotate it back to
        # the front so the reported path is depot-first like the other solvers
        # and ``check_path_distance`` doesn't add a spurious ``distances[0, 0]``
        # closing edge (which would over-report the tour length).
        zero_pos = int(np.flatnonzero(tour == 0)[0])
        if zero_pos != 0:
            tour = np.ascontiguousarray(np.roll(tour, -zero_pos))

        out = tour
        if self.config.back_to_start:
            out = np.append(tour, tour[0])
        value = check_path_distance(self.network_routes, out, self.config.back_to_start)
        return CombinatoricsResult(
            optimal_path=np.array(out),
            optimal_value=value,
            value_history=np.array([value]),
            stop_reason="no_improvement" if n_moves == 0 else "max_iterations",
        )


@dataclass
class NearestNeighborTSPConfig(IOptimizerConfig):
    back_to_start: bool = True
    """Whether to return to the start node"""


class NearestNeighborTSP(TSPBase):
    def __init__(
        self,
        config: NearestNeighborTSPConfig,
        network_routes: Optional[AF] = None,
        city_locations: Optional[AF] = None,
    ):
        super().__init__(network_routes, city_locations)
        self.config = config

    def solve(self) -> CombinatoricsResult:
        # Start at the first node, pick the nearest neighbor. Uses a boolean
        # visited mask + argmin over the current row (report item #13) instead of
        # a Python ``set`` membership scan; argmin's first-min tie-break matches
        # the old strict-``<`` first-found behavior, so the route is identical.
        n = self.network_routes.shape[0]
        route = [0]
        visited = np.zeros(n, dtype=bool)
        total_distance = 0

        current_node = 0  # Start at first node (index 0)
        visited[current_node] = True

        while visited.sum() < n:
            # Find the nearest unvisited neighbor (visited masked to +inf).
            candidates = np.where(visited, np.inf, self.network_routes[current_node])
            nearest_neighbor = int(np.argmin(candidates))
            min_distance = candidates[nearest_neighbor]

            # Check if we found a valid neighbor
            if not np.isfinite(min_distance):
                break

            # Add to route and update distance
            route.append(nearest_neighbor)
            visited[nearest_neighbor] = True
            total_distance += min_distance
            current_node = nearest_neighbor

        # If back_to_start is True, add the final connection
        if self.config.back_to_start:
            total_distance += self.network_routes[current_node][0]
            route.append(0)

        return CombinatoricsResult(
            optimal_path=np.array(route),
            optimal_value=total_distance,
            value_history=np.array([total_distance]),
            stop_reason="none",
        )


@dataclass
class ConvexHullTSPConfig(IOptimizerConfig):
    back_to_start: bool = True
    """Whether to return to the start node"""


class ConvexHullTSP(TSPBase):
    def __init__(
        self,
        config: ConvexHullTSPConfig,
        network_routes: Optional[AF] = None,
        city_locations: Optional[AF] = None,
    ):
        super().__init__(network_routes, city_locations)
        self.config = config

    def solve(self) -> CombinatoricsResult:
        # Use the windmill method starting at point-0.
        # NOTE - This will give us the convex hull PLUS the sequence required to get there.
        current_node = 0
        total_distance = 0
        tour = [current_node]
        visited = set()
        visited.add(current_node)
        start_theta = 0.0

        def atan2pos(v: AF) -> float:
            t = np.arctan2(v[1], v[0])
            if t < 0:
                t += 2 * np.pi
            return float(t)

        assert self.city_locations is not None  # ConvexHull requires coordinates
        restarted = False
        while True:
            # Find the point which is CCW from this point by the least amount.
            min_idx = -1
            min_theta = float("inf")
            p0 = self.city_locations[current_node]
            for i in range(self.city_locations.shape[0]):
                if i != current_node:
                    p_i = self.city_locations[i]
                    dp = p_i - p0
                    theta = atan2pos(dp)
                    if min_theta > theta >= start_theta:
                        min_theta = theta
                        min_idx = i
            if min_idx == -1:
                if not restarted:
                    # Retry this once for the mod 2pi issue
                    start_theta -= 2.0 * np.pi
                    continue
                else:
                    break
            start_theta = min_theta
            total_distance += self.network_routes[current_node][min_idx]
            current_node = min_idx
            tour.append(current_node)
            if current_node in visited:
                break
            visited.add(current_node)

        return CombinatoricsResult(
            optimal_path=np.array(tour),
            optimal_value=total_distance,
            value_history=np.array([total_distance]),
            stop_reason="none",
        )
