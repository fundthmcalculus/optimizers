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
