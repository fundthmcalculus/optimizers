"""Combinatorial (city-permutation / TSP-family) optimizers: GA, ACO, local search."""

from .base import TSPBase, check_path_distance
from .aco import AntColonyTSP, AntColonyTSPConfig
from .aco_mst import AntColonyMST
from .ga import GeneticAlgorithmTSP, GeneticAlgorithmTSPConfig
from .mtsp import AntColonyMTSP, AntColonyMTSPConfig, ClusterMethod
from .strategy import (
    LocalSearchBackend,
    TwoOptTSP,
    TwoOptTSPConfig,
    ThreeOptTSP,
    LinKernighanTSP,
    LinKernighanTSPConfig,
    NearestNeighborTSP,
    NearestNeighborTSPConfig,
    ConvexHullTSP,
    ConvexHullTSPConfig,
)
from .compare import compare_tsp_heuristics, format_comparison_table, HeuristicResult

__all__ = [
    "TSPBase",
    "check_path_distance",
    "AntColonyTSP",
    "AntColonyTSPConfig",
    "AntColonyMST",
    "GeneticAlgorithmTSP",
    "GeneticAlgorithmTSPConfig",
    "AntColonyMTSP",
    "AntColonyMTSPConfig",
    "ClusterMethod",
    "LocalSearchBackend",
    "TwoOptTSP",
    "TwoOptTSPConfig",
    "ThreeOptTSP",
    "LinKernighanTSP",
    "LinKernighanTSPConfig",
    "NearestNeighborTSP",
    "NearestNeighborTSPConfig",
    "ConvexHullTSP",
    "ConvexHullTSPConfig",
    "compare_tsp_heuristics",
    "format_comparison_table",
    "HeuristicResult",
]
