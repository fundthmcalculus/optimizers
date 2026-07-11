from dataclasses import dataclass
from typing import Optional

import numpy as np
from joblib import delayed

from .base import CombinatoricsResult, TSPBase, _check_stop_early
from .strategy import TwoOptTSPConfig, TwoOptTSP
from ..core.base import IOptimizerConfig, setup_for_generations
from ..core.types import AI, AF, ab8, i32, i16


@dataclass
class AntColonyTSPConfig(IOptimizerConfig):
    rho: float = 0.2  # 0.451  # 0.5
    """Pheromone decay parameter"""
    alpha: float = 0.8  # 1.88  # 1.0
    """Pheromone deposit parameter"""
    beta: float = 2  # 1.88  # 1.0
    """Pheromone evaporation parameter"""
    q: float = 1  # 2.17  # 1.0
    """Weighting parameter for selecting better ranked solutions"""
    back_to_start: bool = True
    """Whether to return to the start node"""
    local_optimize: bool = False
    """Local optimization using 2OPT method"""
    hot_start: Optional[np.ndarray] = None
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
        eta[eta == -1] = 0
        # Desirability is constant for the whole run, so raise it to ``beta``
        # exactly once here instead of billions of times inside ``p_xy`` (report
        # item #6). ``tau ** alpha`` still changes each generation and is hoisted
        # to once-per-generation below.
        eta_beta = np.power(eta, self.config.beta)
        # Pheromone matrix
        tau = np.ones(self.network_routes.shape)
        # If we have a hot start, preload it 4x
        optimal_city_order: Optional[np.ndarray] = None
        tour_lengths = []
        optimal_tour_length = np.inf
        if self.config.hot_start is not None:
            if self.config.hot_start_length is None:
                raise ValueError(
                    "hot_start_length must be provided when hot_start is set"
                )
            optimal_tour_length = self.config.hot_start_length
            optimal_city_order = self.config.hot_start
            for i in range(len(self.config.hot_start) - 1):
                tau[self.config.hot_start[i], self.config.hot_start[i + 1]] += (
                    10 * self.config.q / self.config.hot_start_length
                )

        generation_pbar, individuals_per_job, n_jobs, parallel = setup_for_generations(
            self.config
        )

        with parallel:
            for generations_completed in generation_pbar:
                # Compute the change in pheromone!
                delta_tau = np.zeros(tau.shape)
                # ``tau`` only changes once per generation, so raise it to
                # ``alpha`` here and hand the workers the precomputed matrix
                # (report item #6) instead of recomputing it per ant per step.
                tau_alpha = np.power(tau, self.config.alpha)

                def parallel_ant(local_ant):
                    results = []
                    for _ in range(individuals_per_job):
                        results.append(
                            run_ant(
                                self.network_routes, eta_beta, tau_alpha, self.config
                            )
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
                        # Deposit q/tour_length on every traversed edge at once
                        # (report item #10) instead of a per-edge Python loop.
                        deposit = self.config.q / tour_length
                        np.add.at(
                            delta_tau,
                            (city_order[:-1], city_order[1:]),
                            deposit,
                        )
                        if self.config.back_to_start:
                            delta_tau[city_order[-1], city_order[0]] += deposit
                tour_lengths.append(optimal_tour_length)
                generation_pbar.set_postfix(best_value=optimal_tour_length)
                # Once all ants are done, update the pheromone
                tau = pheromone_update(tau, delta_tau, self.config.rho)
                # Check for stopping early
                stop_reason = _check_stop_early(self.config, tour_lengths)
                if stop_reason != "none":
                    break

        if self.config.local_optimize:
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
        else:
            return CombinatoricsResult(
                optimal_path=optimal_city_order,
                optimal_value=optimal_tour_length,
                value_history=np.array(tour_lengths),
                stop_reason="max_iterations" if stop_reason == "none" else stop_reason,
            )


def pheromone_update(tau_xy: AF, delta_tau_xy: AF, rho: float) -> AF:
    new_tau_xy = (1 - rho) * tau_xy + delta_tau_xy
    return new_tau_xy / new_tau_xy.max()


def p_xy(eta_beta_xy: AF, tau_alpha_xy: AF, allowed_y: ab8, x: int) -> AF:
    # ``eta_beta``/``tau_alpha`` are already raised to beta/alpha upstream
    # (report item #6), so this is a single elementwise product per step.
    p = tau_alpha_xy[x, :] * eta_beta_xy[x, :]
    # Remove disallowed / negative probabilities, those are not allowed
    p[~allowed_y] = 0
    p[p < 0] = 0
    # Normalize the probabilities
    total = p.sum()
    if total == 0.0:
        # all-zero (un-normalized) array; callers test np.sum(p) == 0
        return p
    p /= total
    return p


def run_ant(
    network_routes: AF, eta_beta: AF, tau_alpha: AF, config: AntColonyTSPConfig
) -> tuple[AI, float]:
    # Start at city 1, and visit each city exactly once
    cur_city = 0  # Offset by 1, so we start at city 1
    eta_shape_ = eta_beta.shape[0]
    order_len = eta_shape_
    # If fewer than 32,000 cities, we can use i16
    dtype = i32
    if order_len < 32000:
        dtype = i16
    city_order = np.zeros(order_len, dtype=dtype)
    idx = 0
    total_length = 0
    allowed_cities = np.ones(eta_shape_, dtype=bool)
    while np.any(allowed_cities):
        # Mark off the current city
        allowed_cities[cur_city] = False
        city_order[idx] = cur_city
        # Compute the probability of each city
        p = p_xy(eta_beta, tau_alpha, allowed_cities, cur_city)
        # If the probability is zero, we're stuck, this is a dead end!
        if np.sum(p) == 0 or np.any(np.isnan(p)):
            # If we have hit every city, we're done! We don't need to go back to the start, since we solved in reverse.
            if np.sum(allowed_cities) != 0:
                # Invalid route!
                total_length = np.inf
            # IF back-to-start, include that option
            if config.back_to_start:
                total_length += network_routes[city_order[-1], city_order[0]]
            break
        # Choose the next city via inverse-CDF sampling (report item #10):
        # cheaper than np.random.choice(..., p=p), which re-validates and
        # re-cumsums p on every call and takes a global lock.
        cur_city = int(np.searchsorted(np.cumsum(p), np.random.random()))
        if cur_city >= eta_shape_:
            cur_city = eta_shape_ - 1
        total_length += network_routes[city_order[idx], cur_city]
        idx += 1

    return city_order, total_length
