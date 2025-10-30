from abc import ABC, abstractmethod
from sklearn.metrics import pairwise_distances
from typing import Optional

from ..core.types import AF, AI, F
from ..core.base import StopReason
from dataclasses import dataclass


@dataclass
class CombinatoricsResult:
    optimal_path: AI
    optimal_value: F
    value_history: AF
    stop_reason: StopReason


def check_path_distance(distances: AF, order_path: AI, return_to_start=False) -> F:
    total_dist = 0.0
    for ij, p0 in enumerate(order_path):
        if ij == len(order_path) - 1:
            if return_to_start:
                total_dist += distances[p0, 0]
        else:
            p1 = order_path[ij + 1]
            total_dist += distances[p0, p1]
    return total_dist


class TSPBase(ABC):
    def __init__(self,
                 network_routes: Optional[AF] = None,
                 city_locations: Optional[AF] = None,
                 ):
        self.city_locations = None
        self.network_routes = None
        self.set_network_routes(network_routes, city_locations)

    def set_network_routes(self,
        network_routes: Optional[AF] = None,
        city_locations: Optional[AF] = None):
        """Set the network routes for the TSP solver"""
        # If we have network routes, use that, otherwise, use the city locations
        if network_routes is None:
            assert city_locations is not None

            # Compute pairwise distances between all cities
            assert len(city_locations.shape) == 2, "City locations must be a 2D array"
            self.city_locations = city_locations.copy()
            self.network_routes = pairwise_distances(city_locations)
        else:
            self.network_routes = network_routes.copy()

    @abstractmethod
    def solve(self) -> CombinatoricsResult:
        raise NotImplementedError("This method should be implemented in the base classes")