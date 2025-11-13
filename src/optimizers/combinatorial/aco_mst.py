from dataclasses import dataclass
from typing import Optional

import numpy as np
from joblib import delayed

from .base import CombinatoricsResult, TSPBase, _check_stop_early
from .strategy import TwoOptTSPConfig, TwoOptTSP
from ..core.base import IOptimizerConfig, setup_for_generations
from ..core.types import AI, AF, F, i32, i16
from .aco import AntColonyTSPConfig

# NOTE - MST is the same as TSP parameters, except the "back to start" is ignored.


class AntColonyMST(TSPBase):
    def __init__(
        self,
        config: AntColonyTSPConfig,
        network_routes: Optional[AF] = None,
        city_locations: Optional[AF] = None,
    ):
        super().__init__(network_routes, city_locations)
        self.config = config

    def solve(self, start_idx: int = 0) -> CombinatoricsResult:
        self.network_routes[self.network_routes == 0] = -1
        # TODO - Should we not cache this for memory efficiency?
        eta = 1.0 / self.network_routes
        eta[eta == -1] = 0
        # Pheromone matrix
        tau = np.ones(self.network_routes.shape)
        # If we have a hot start, preload it 4x
        optimal_city_order: Optional[np.ndarray] = None
        tour_lengths = []
        optimal_tour_length = np.inf
        if self.config.hot_start is not None:
            optimal_tour_length = self.config.hot_start_length
            optimal_city_order = self.config.hot_start
            for ij in range(self.config.hot_start.shape[0]):
                tau[self.config.hot_start[ij,0], self.config.hot_start[ij,0]] += (
                    10 * self.config.q / self.config.hot_start_length
                )

        generation_pbar, individuals_per_job, n_jobs, parallel = setup_for_generations(
            self.config
        )

        with parallel:
            for generations_completed in generation_pbar:
                # Compute the change in pheromone!
                delta_tau = np.zeros(tau.shape)

                def parallel_ant(local_ant):
                    results = []
                    for _ in range(individuals_per_job):
                        results.append(
                            run_ant_mst(self.network_routes, eta, tau, self.config, start_idx)
                        )
                    return results

                all_results = parallel(
                    delayed(parallel_ant)(i_ant) for i_ant in range(n_jobs)
                )

                for ant, result_gen in enumerate(all_results):
                    for city_order, tour_length in result_gen:
                        # If a dead-end, skip!
                        if tour_length == np.inf:
                            continue
                        # Update the relative ant pheromone
                        if tour_length <= optimal_tour_length:
                            optimal_tour_length = tour_length
                            optimal_city_order = city_order
                        for i in range(len(city_order) - 1):
                            delta_tau[city_order[i], city_order[i + 1]] += (
                                self.config.q / tour_length
                            )
                        if self.config.back_to_start:
                            delta_tau[city_order[-1], city_order[0]] += (
                                self.config.q / tour_length
                            )
                tour_lengths.append(optimal_tour_length)
                generation_pbar.set_postfix(best_value=optimal_tour_length)
                # Once all ants are done, update the pheromone
                tau = pheromone_update(tau, delta_tau, self.config.rho)
                # Check for stopping early
                stop_reason = _check_stop_early(self.config, tour_lengths)
                if stop_reason != "none":
                    break

            return CombinatoricsResult(
                optimal_path=optimal_city_order,
                optimal_value=optimal_tour_length,
                value_history=np.array(tour_lengths),
                stop_reason="max_iterations" if stop_reason == "none" else stop_reason,
            )


def pheromone_update(tau_xy, delta_tau_xy, rho):
    new_tau_xy = (1 - rho) * tau_xy + delta_tau_xy
    return new_tau_xy / new_tau_xy.max()


def p_xy(eta_xy, tau_xy, allowed_y, alpha, beta):
    p = np.power(tau_xy[~allowed_y, :], alpha) * np.power(eta_xy[~allowed_y, :], beta)
    # Remove negative probabilities, those are not allowed
    p[:,~allowed_y] = 0
    p[p < 0] = 0
    # Normalize the probabilities
    if np.sum(p) == 0.0:
        return 0
    p = p / np.sum(p)
    return p


def run_ant_mst(
    network_routes: AF, eta: AF, tau_xy: AF, config: AntColonyTSPConfig, start_idx: int
) -> tuple[AI, F]:
    cur_city = start_idx
    order_len = eta.shape[0]
    # If fewer than 32,000 cities, we can use i16
    dtype = i32
    if order_len < 32000:
        dtype = i16

    # (FROM, TO) ordering
    city_order = np.zeros((order_len, 2), dtype=dtype)

    idx = 0
    total_length = 0
    allowed_cities = np.ones(order_len, dtype=bool)
    choice_indexes = np.arange(order_len)

    # Mark off the current city
    allowed_cities[cur_city] = False
    city_order[idx, 0] = cur_city

    while np.any(allowed_cities):
        # Compute the probability of each city
        p = p_xy(eta, tau_xy, allowed_cities, config.alpha, config.beta)
        # If the probability is zero, we're stuck, this is a dead end!
        if np.sum(p) == 0 or np.any(np.isnan(p)):
            # If we have hit every city, we're done! We don't need to go back to the start, since we solved in reverse.
            if np.sum(allowed_cities) != 0:
                # Invalid route!
                total_length = np.inf
            break
        # Choose the next city-pair, must flatten probability matrix first
        cum_p = np.cumsum(p.flatten())
        new_p = np.random.random()
        choice_idx = np.argmin(new_p > cum_p)
        from_row = choice_idx // order_len
        from_col = choice_idx % order_len
        city_order[idx,0] = from_row
        city_order[idx,1] = choice_indexes[from_col]
        total_length += network_routes[city_order[idx,0], city_order[idx,1]]
        allowed_cities[city_order[idx,0]] = False
        allowed_cities[city_order[idx,1]] = False

        idx += 1

    return city_order, total_length
