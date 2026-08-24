import numpy as np
from sklearn.metrics import pairwise_distances

from ..core.types import AF, AI, F
from ..core.base import BaseOptimizer, StopReason, IOptimizerConfig


def check_path_distance(
    distances: AF, order_path: AI, return_to_start: bool = False
) -> F:
    # Sum every consecutive edge in one fancy-indexed gather instead of a scalar
    # Python loop (report item #10). Equivalent to the old loop: the closing edge
    # returns to city 0 (order_path[0] is always the depot in these solvers).
    order_path = np.asarray(order_path)
    if order_path.shape[0] < 2:
        total_dist = 0.0
    else:
        total_dist = float(distances[order_path[:-1], order_path[1:]].sum())
    if return_to_start and order_path.shape[0] >= 1:
        total_dist += float(distances[order_path[-1], 0])
    return total_dist


def _check_stop_early(config: IOptimizerConfig, soln_history: list[F]) -> StopReason:
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


class TSPBase(BaseOptimizer):
    # ``network_routes`` is always populated by ``set_network_routes`` during
    # __init__ (the None branch asserts city_locations is present), so it is a
    # non-Optional distance matrix for the lifetime of the solver.
    city_locations: AF | None
    network_routes: AF

    def __init__(
        self,
        *,
        config: IOptimizerConfig,
        network_routes: AF | None = None,
        city_locations: AF | None = None,
    ):
        self.config = config
        self.city_locations = None
        self.set_network_routes(network_routes, city_locations)

    def set_network_routes(
        self, network_routes: AF | None = None, city_locations: AF | None = None
    ) -> None:
        """Set the network routes for the TSP solver"""
        # If we have network routes, use that, otherwise, use the city locations
        if network_routes is None:
            if city_locations is None:
                raise ValueError(
                    "Either network_routes or city_locations must be provided."
                )
            if len(city_locations.shape) != 2:
                raise ValueError("City locations must be a 2D array")

            # Compute pairwise distances between all cities
            self.city_locations = city_locations.copy()
            self.network_routes = pairwise_distances(city_locations)
        else:
            self.network_routes = network_routes.copy()
