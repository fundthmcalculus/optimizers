import joblib
import numpy as np
from typing import Optional
from dataclasses import dataclass

import tqdm
from joblib import Parallel, delayed

from .base import check_path_distance, CombinatoricsResult, TSPBase
from ..core.base import IOptimizerConfig, StopReason
from ..core.types import AI, AF, F, i32, i16


@dataclass
class TwoOptTSPConfig(IOptimizerConfig):
    back_to_start: bool = True
    """Whether to return to the start node"""
    num_iterations: int = 10
    """Number of iterations to run"""
    nearest_neighbors: int = -1
    """Only check the next nodes, which makes this O(n), but lower chance of finding crossovers"""


class TwoOptTSP(TSPBase):
    def __init__(
        self,
        config: TwoOptTSPConfig,
        initial_route: Optional[AI] = None,
        initial_value: Optional[F] = None,
        network_routes: Optional[AF] = None,
        city_locations: Optional[AF] = None,
    ):
        super().__init__(network_routes, city_locations)
        self.config = config
        self.initial_value = initial_value
        self.initial_route = initial_route

    def solve(self) -> CombinatoricsResult:
        if self.initial_route is None or self.initial_value == None:
            # Use the nearest neighbor
            nn_config = NearestNeighborTSPConfig(
                back_to_start=self.config.back_to_start, name=self.config.name
            )
            nn_solver = NearestNeighborTSP(nn_config)
            solution = nn_solver.solve()
            self.initial_route = solution.optimal_path
            self.initial_value = solution.optimal_value
        new_route = self.initial_route.copy()
        N = self.network_routes.shape[0]
        for cur_iter in range(self.config.num_iterations):
            for ij in range(0, N - 2):
                k_nn = N
                if self.config.nearest_neighbors > 0:
                    k_nn = min(k_nn, ij + self.config.nearest_neighbors)
                for jk in range(ij + 2, k_nn):
                    d1 = (
                        self.network_routes[new_route[ij], new_route[ij + 1]]
                        + self.network_routes[new_route[jk], new_route[jk + 1]]
                    )
                    d2 = (
                        self.network_routes[new_route[ij], new_route[jk]]
                        + self.network_routes[new_route[ij + 1], new_route[jk + 1]]
                    )
                    if d1 > d2:
                        new_route[jk], new_route[ij + 1] = (
                            new_route[ij + 1],
                            new_route[jk],
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
            stop_reason="none",
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
            t = np.atan2(v[1], v[0])
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


def _check_stop_early(config: IOptimizerConfig, soln_history) -> StopReason:
    if len(soln_history) < config.stop_after_iterations:
        return "none"
    if np.allclose(
        soln_history[-config.stop_after_iterations],
        soln_history[-1],
        rtol=1e-2,
        atol=1e-2,
    ):
        return "no_improvement"
    return "none"


@dataclass
class AntColonyTSPConfig(IOptimizerConfig):
    rho: float = 0.5
    """Pheromone decay parameter"""
    alpha: float = 1.0
    """Pheromone deposit parameter"""
    beta: float = 1.0
    """Pheromone evaporation parameter"""
    q: float = 1.0
    """Weighting parameter for selecting better ranked solutions"""
    back_to_start: bool = True
    local_optimize: bool = False
    """Local optimization using 2OPT method"""
    """Whether to return to the start node"""
    hot_start: Optional[list[int]] = None
    """Hot start solution"""
    hot_start_length: Optional[float] = None
    """Hot start length"""


class AntColonyTSP(TSPBase):
    def __init__(
        self,
        config: AntColonyTSPConfig,
        network_routes: Optional[AF] = None,
        city_locations: Optional[AF] = None,
    ):
        super().__init__(network_routes, city_locations)
        self.config = config

    def solve(self) -> CombinatoricsResult:
        self.network_routes[self.network_routes == 0] = -1
        # TODO - Should we not cache this for memory efficiency?
        eta = 1.0 / self.network_routes
        # Pheromone matrix
        tau = np.ones(self.network_routes.shape)
        # If we have a hot start, preload it 4x
        optimal_city_order: Optional[list[int]] = None
        tour_lengths = []
        optimal_tour_length = np.inf
        if self.config.hot_start is not None:
            optimal_tour_length = self.config.hot_start_length
            optimal_city_order = self.config.hot_start
            for i in range(len(self.config.hot_start) - 1):
                tau[self.config.hot_start[i], self.config.hot_start[i + 1]] += (
                    10 * self.config.q / self.config.hot_start_length
                )

        with Parallel(n_jobs=joblib.cpu_count() // 2) as parallel:
            for generation in tqdm.trange(
                self.config.num_generations, desc="ACO-TSP Generation"
            ):
                # Compute the change in pheromone!
                delta_tau = np.zeros(tau.shape)
                optimal_ant_len = np.inf
                optimal_ant_city_order = None

                def parallel_ant(local_ant):
                    return run_ant(self.network_routes, eta, tau, self.config)

                all_results = parallel(
                    delayed(parallel_ant)(i_ant)
                    for i_ant in range(self.config.population_size)
                )

                for ant in range(self.config.population_size):
                    tour_length = all_results[ant][1]
                    city_order = all_results[ant][0]
                    # If a dead-end, skip!
                    if tour_length == np.inf:
                        continue
                    # Update the relative ant pheromone
                    if tour_length <= optimal_ant_len:
                        optimal_ant_len = tour_length
                        optimal_ant_city_order = city_order
                    for i in range(len(city_order) - 1):
                        delta_tau[city_order[i], city_order[i + 1]] += (
                            self.config.q / tour_length
                        )
                    if self.config.back_to_start:
                        delta_tau[city_order[-1], city_order[0]] += (
                            self.config.q / tour_length
                        )
                # Update the per-generation information
                if optimal_ant_len < optimal_tour_length:
                    optimal_tour_length = optimal_ant_len
                    optimal_city_order = optimal_ant_city_order
                tour_lengths.append(optimal_tour_length)
                # Once all ants are done, update the pheromone
                tau = pheromone_update(tau, delta_tau, self.config.rho)
                # Check for stopping early
                stop_reason = _check_stop_early(self.config, tour_lengths)
                if stop_reason != "none":
                    break

        # TODO - Better parameters?
        two_opt_config = TwoOptTSPConfig()
        two_opt_optimize = TwoOptTSP(
            two_opt_config,
            initial_route=optimal_city_order,
            initial_value=optimal_tour_length,
            network_routes=self.network_routes,
            city_locations=self.city_locations,
        )
        result = two_opt_optimize.solve()
        tour_lengths.append(result.optimal_value)

        return CombinatoricsResult(
            optimal_path=result.optimal_path,
            optimal_value=result.optimal_value,
            value_history=np.array(tour_lengths),
            stop_reason="max_iterations" if stop_reason == "none" else stop_reason,
        )


def pheromone_update(tau_xy, delta_tau_xy, rho):
    new_tau_xy = (1 - rho) * tau_xy + delta_tau_xy
    return new_tau_xy / new_tau_xy.max()


def p_xy(eta_xy, tau_xy, allowed_y, alpha, beta, x):
    p = (tau_xy[x, :] ** alpha) * eta_xy[x, :] ** beta
    # Remove negative probabilities, those are not allowed
    p[~allowed_y] = 0
    p[p < 0] = 0
    # Normalize the probabilities
    if np.sum(p) == 0.0:
        return 0
    p = p / np.sum(p)
    return p


def run_ant(
    network_routes: AF, eta: AF, tau_xy: AF, config: AntColonyTSPConfig
) -> tuple[AI, F]:
    # Start at city 1, and visit each city exactly once
    cur_city = 0  # Offset by 1, so we start at city 1
    eta_shape_ = eta.shape[0]
    order_len = eta_shape_
    if config.back_to_start:
        order_len += 1
    # If fewer than 32,000 cities, we can use i16
    dtype = i32
    if order_len < 32000:
        dtype = i16
    city_order = np.zeros(order_len, dtype=dtype)
    idx = 0
    total_length = 0
    allowed_cities = np.ones(eta_shape_, dtype=bool)
    choice_indexes = np.arange(eta_shape_)
    while np.any(allowed_cities):
        # Mark off the current city
        allowed_cities[cur_city] = False
        city_order[idx] = cur_city
        # Compute the probability of each city
        p = p_xy(eta, tau_xy, allowed_cities, config.alpha, config.beta, cur_city)
        # If the probability is zero, we're stuck, this is a dead end!
        if np.sum(p) == 0 or np.any(np.isnan(p)):
            # If we have hit every city, we're done! We don't need to go back to the start, since we solved in reverse.
            if np.sum(allowed_cities) != 0:
                # Invalid route!
                total_length = np.inf
            # IF back-to-start, include that option
            if config.back_to_start:
                city_order[-1] = 0
                total_length += network_routes[city_order[-2], city_order[-1]]
            break
        # Choose the next city
        cur_city = np.random.choice(choice_indexes, p=p)
        total_length += network_routes[city_order[idx], cur_city]
        idx += 1

    # Randomly permute 2 entries that aren't the first, and see if that's shorter
    r0 = np.random.randint(low=1, high=eta_shape_)
    r1 = np.random.randint(low=1, high=eta_shape_)
    c0 = city_order[r0]
    c1 = city_order[r1]
    city_order[r0] = c1
    city_order[r1] = c0
    permute_distance = check_path_distance(network_routes, city_order)
    if permute_distance < total_length:
        total_length = permute_distance
    else:
        city_order[r0] = c0
        city_order[r1] = c1

    return city_order, total_length
