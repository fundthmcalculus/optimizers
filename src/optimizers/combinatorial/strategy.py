import heapq
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numba import njit

from ..core import IOptimizerConfig
from .base import TSPBase, CombinatoricsResult, check_path_distance
from ..core.types import AI, F, AF


@dataclass
class TwoOptTSPConfig(IOptimizerConfig):
    back_to_start: bool = True
    """Whether to return to the start node"""
    num_iterations: int = -1
    """Number of iterations to run"""
    nearest_neighbors: int = -1
    """Only check the next nodes, which makes this O(nk), but lower chance of finding crossovers"""


def _swap_segment(ij, jk, new_route):
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
def _two_opt_kernel(distances, route, num_iterations, nearest_neighbors, back_to_start):
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
def _three_opt_kernel(distances, route, num_iterations, nearest_neighbors):
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
                    d0 = (
                        distances[a, b] + distances[c, d] + distances[e, f]
                    )
                    d1 = (
                        distances[a, e] + distances[d, c] + distances[b, f]
                    )
                    d2 = (
                        distances[a, b] + distances[c, e] + distances[d, f]
                    )
                    d3 = (
                        distances[a, c] + distances[b, d] + distances[e, f]
                    )
                    d4 = (
                        distances[a, c] + distances[b, e] + distances[d, f]
                    )
                    d5 = (
                        distances[a, e] + distances[d, b] + distances[c, f]
                    )
                    d6 = (
                        distances[a, d] + distances[e, c] + distances[b, f]
                    )
                    d7 = (
                        distances[a, d] + distances[e, b] + distances[c, f]
                    )
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
        # njit kernel (report item #13); logic identical to the old Python loop.
        new_route = np.ascontiguousarray(new_route)
        distances = np.ascontiguousarray(self.network_routes, dtype=np.float64)
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

    def setup_local_search(self) -> tuple[int, AF]:
        if self.initial_route is None or self.initial_value == None:
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
        # njit kernel (report item #13); logic identical to the old Python
        # loop, including num_iterations=-1 -> range(-1) being a no-op.
        new_route = np.ascontiguousarray(new_route)
        distances = np.ascontiguousarray(self.network_routes, dtype=np.float64)
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

        def atan2pos(v):
            t = np.arctan2(v[1], v[0])
            if t < 0:
                t += 2 * np.pi
            return t

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
