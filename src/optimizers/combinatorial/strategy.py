import heapq
from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np

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

def _swap_segment(ij: int, jk: int, new_route: AI) -> AI:
    ij += 1
    while ij < jk:
        temp = new_route[ij]
        new_route[ij] = new_route[jk]
        new_route[jk] = temp
        ij += 1
        jk -= 1
    return new_route

def _try_swap(network_routes: AF, ij: int, ij_: int, jk: int, jk_: int, new_route: AI) -> tuple[AI, bool]:
    ij = ij % len(new_route)
    jk = jk % len(new_route)
    ij_ = ij_ % len(new_route)
    jk_ = jk_ % len(new_route)
    no_moves = True
    d1 = (
            network_routes[new_route[ij], new_route[ij_]]
            + network_routes[new_route[jk], new_route[jk_]]
    )
    d2 = (
            network_routes[new_route[ij], new_route[jk]]
            + network_routes[new_route[ij_], new_route[jk_]]
    )
    if d1 > d2:
        new_route = _swap_segment(ij, jk, new_route)
        no_moves = False
    return new_route, no_moves

def _try_replace(network_routes: AF, ij: int, jk: int, new_route: AI, back_to_start: bool) -> tuple[AI, bool]:
    d0 = check_path_distance(network_routes, new_route, back_to_start)
    # Exchange cities ij and jk
    temp = new_route[ij]
    new_route[ij] = new_route[jk]
    new_route[jk] = temp
    d1 = check_path_distance(network_routes, new_route, back_to_start)
    if d1 < d0:
        return new_route, True
    else:
        # Revert the operation
        temp = new_route[ij]
        new_route[ij] = new_route[jk]
        new_route[jk] = temp
        return new_route, False


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
        no_moves = True
        for cur_iter in range(self.config.num_iterations):
            no_moves = True
            for ij in range(0, N - 2):
                k_nn = N-1
                if self.config.nearest_neighbors > 0:
                    k_nn = min(k_nn, ij + self.config.nearest_neighbors)
                for jk in range(ij + 2, k_nn):
                    ij_ = ij + 1
                    jk_ = jk + 1
                    new_route, _no_moves = _try_swap(self.network_routes, ij, ij_, jk, jk_, new_route)
                    no_moves = no_moves and _no_moves

            if no_moves:
                break

        # TODO - Show the history through each iteration?
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
        if self.initial_route is None or self.initial_value == None:
            # Use the nearest neighbor
            nn_config = NearestNeighborTSPConfig(
                back_to_start=self.config.back_to_start, name=self.config.name
            )
            nn_solver = NearestNeighborTSP(nn_config, network_routes=self.network_routes, city_locations=self.city_locations)
            solution = nn_solver.solve()
            self.initial_route = solution.optimal_path
            self.initial_value = solution.optimal_value
        new_route = self.initial_route.copy()
        N = self.network_routes.shape[0]
        if self.config.num_iterations == -1:
            self.config.num_iterations = N
        return N, new_route


@dataclass
class PriorityTwoOptTSPConfig(TwoOptTSPConfig):
    priority_depth: int = 10
    """Use a priority queue to sort the `priority_depth` worst samples"""


class PriorityTwoOptTSP(TwoOptTSP):
    def __init__(
        self,
        config: PriorityTwoOptTSPConfig,
        network_routes: Optional[AF] = None,
        city_locations: Optional[AF] = None,
        initial_route: Optional[AI] = None,
        initial_value: Optional[F] = None,
    ):
        super().__init__(config, network_routes, city_locations, initial_route, initial_value)
        self.config: PriorityTwoOptTSPConfig = config
        self.initial_value = initial_value
        self.initial_route = initial_route

    def solve(self) -> CombinatoricsResult:
        N, new_route = self.setup_local_search()
        history = [check_path_distance(self.network_routes, new_route, self.config.back_to_start)]

        # To get the worst edges, use the negative distance.
        worst_edge_heap: list[tuple[float, int]] = []
        heapq.heapify(worst_edge_heap)

        # Convenience function to insert into the heap.
        def insert_worst_edge(_p):
            nonlocal worst_edge_heap
            heapq.heappush(worst_edge_heap, (-self.network_routes[new_route[_p], new_route[_p+1]], _p))

        if self.config.back_to_start:
            # Manually insert the return-to-start here.
            insert_worst_edge(-1)
        for ij in range(0, N - 2):
            insert_worst_edge(ij)

        # Chop to the N-smallest (most negative)
        worst_edges = heapq.nsmallest(self.config.priority_depth, worst_edge_heap)
        for ij in range(len(worst_edges)):
            for jk in range(ij+1, len(worst_edges)):
                p0 = worst_edges[ij][1]
                p1 = worst_edges[jk][1]
                # _p0 = p0+1
                # _p1 = p1+1
                # new_route, _ = _try_swap(self.network_routes, p0, _p0, p1, _p1, new_route)
                new_route, _ = _try_replace(self.network_routes, p0, p1, new_route, self.config.back_to_start)


        history.append(
            check_path_distance(
                self.network_routes, new_route, self.config.back_to_start
            )
        )

        return CombinatoricsResult(
            optimal_path=np.array(new_route),
            optimal_value=history[-1],
            value_history=np.array(history),
            stop_reason="max_iterations",
        )


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
        no_moves = True
        for cur_iter in range(self.config.num_iterations):
            no_moves = True
            for ij in range(0, N - 4):
                k_nn = N - 2
                if self.config.nearest_neighbors > 0:
                    k_nn = min(k_nn, ij + self.config.nearest_neighbors)
                for jk in range(ij + 2, k_nn):
                    l_nn = N
                    if self.config.nearest_neighbors > 0:
                        l_nn = min(l_nn, jk + self.config.nearest_neighbors)
                    for kl in range(jk + 2, l_nn):
                        # Create each of the 8 cases
                        # TODO - generalize to higher dimensions?
                        A = ij
                        B = ij + 1
                        C = jk
                        D = jk + 1
                        E = kl
                        F = kl + 1
                        d = np.zeros(8)
                        d[0] = (
                            self.network_routes[new_route[A], new_route[B]]
                            + self.network_routes[new_route[C], new_route[D]]
                            + self.network_routes[new_route[E], new_route[F]]
                        )
                        d[1] = (
                            self.network_routes[new_route[A], new_route[E]]
                            + self.network_routes[new_route[D], new_route[C]]
                            + self.network_routes[new_route[B], new_route[F]]
                        )
                        d[2] = (
                            self.network_routes[new_route[A], new_route[B]]
                            + self.network_routes[new_route[C], new_route[E]]
                            + self.network_routes[new_route[D], new_route[F]]
                        )
                        d[3] = (
                            self.network_routes[new_route[A], new_route[C]]
                            + self.network_routes[new_route[B], new_route[D]]
                            + self.network_routes[new_route[E], new_route[F]]
                        )
                        d[4] = (
                            self.network_routes[new_route[A], new_route[C]]
                            + self.network_routes[new_route[B], new_route[E]]
                            + self.network_routes[new_route[D], new_route[F]]
                        )
                        d[5] = (
                            self.network_routes[new_route[A], new_route[E]]
                            + self.network_routes[new_route[D], new_route[B]]
                            + self.network_routes[new_route[C], new_route[F]]
                        )
                        d[6] = (
                            self.network_routes[new_route[A], new_route[D]]
                            + self.network_routes[new_route[E], new_route[C]]
                            + self.network_routes[new_route[B], new_route[F]]
                        )
                        d[7] = (
                            self.network_routes[new_route[A], new_route[D]]
                            + self.network_routes[new_route[E], new_route[B]]
                            + self.network_routes[new_route[C], new_route[F]]
                        )
                        # Find the shortest length
                        min_length = np.argmin(d)
                        if min_length == 0:
                            continue
                        elif min_length == 1:
                            (
                                new_route[A],
                                new_route[E],
                                new_route[D],
                                new_route[C],
                                new_route[B],
                                new_route[F],
                            ) = (
                                new_route[A],
                                new_route[B],
                                new_route[C],
                                new_route[D],
                                new_route[E],
                                new_route[F],
                            )
                        elif min_length == 2:
                            (
                                new_route[A],
                                new_route[B],
                                new_route[C],
                                new_route[E],
                                new_route[D],
                                new_route[F],
                            ) = (
                                new_route[A],
                                new_route[B],
                                new_route[C],
                                new_route[D],
                                new_route[E],
                                new_route[F],
                            )
                        elif min_length == 3:
                            (
                                new_route[A],
                                new_route[C],
                                new_route[B],
                                new_route[D],
                                new_route[E],
                                new_route[F],
                            ) = (
                                new_route[A],
                                new_route[B],
                                new_route[C],
                                new_route[D],
                                new_route[E],
                                new_route[F],
                            )
                        elif min_length == 4:
                            (
                                new_route[A],
                                new_route[C],
                                new_route[B],
                                new_route[E],
                                new_route[D],
                                new_route[F],
                            ) = (
                                new_route[A],
                                new_route[B],
                                new_route[C],
                                new_route[D],
                                new_route[E],
                                new_route[F],
                            )
                        elif min_length == 5:
                            (
                                new_route[A],
                                new_route[E],
                                new_route[D],
                                new_route[B],
                                new_route[C],
                                new_route[F],
                            ) = (
                                new_route[A],
                                new_route[B],
                                new_route[C],
                                new_route[D],
                                new_route[E],
                                new_route[F],
                            )
                        elif min_length == 6:
                            (
                                new_route[A],
                                new_route[D],
                                new_route[E],
                                new_route[C],
                                new_route[B],
                                new_route[F],
                            ) = (
                                new_route[A],
                                new_route[B],
                                new_route[C],
                                new_route[D],
                                new_route[E],
                                new_route[F],
                            )
                        elif min_length == 7:
                            (
                                new_route[A],
                                new_route[D],
                                new_route[E],
                                new_route[B],
                                new_route[C],
                                new_route[F],
                            ) = (
                                new_route[A],
                                new_route[B],
                                new_route[C],
                                new_route[D],
                                new_route[E],
                                new_route[F],
                            )

                        no_moves = False

            if no_moves:
                break

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
        # Start at the first node, pick the nearest neighbor
        route = [0]

        # Initialize variables for tracking visited nodes and total distance
        visited = set()
        total_distance = 0

        current_node = 0  # Start at first node (index 0)
        visited.add(current_node)

        while len(visited) < self.network_routes.shape[0]:
            # Find the nearest unvisited neighbor
            min_distance = float("inf")
            nearest_neighbor = -1

            for i in range(self.network_routes.shape[0]):
                if (
                    i not in visited
                    and self.network_routes[current_node][i] < min_distance
                ):
                    min_distance = self.network_routes[current_node][i]
                    nearest_neighbor = i

            # Check if we found a valid neighbor
            if nearest_neighbor == -1:
                break

            # Add to route and update distance
            route.append(nearest_neighbor)
            visited.add(nearest_neighbor)
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
