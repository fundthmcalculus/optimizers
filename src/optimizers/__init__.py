"""Top-level convenience surface: everything a typical user needs is importable
directly from ``optimizers``. For scoped imports, the same names (plus the
family-specific base classes) are also available from the submodules:
``optimizers.continuous``, ``optimizers.combinatorial``, and ``optimizers.archive``.
"""

# Continuous (real-valued decision-vector) optimizers
from optimizers.continuous import (
    AntColonyOptimizer,
    AntColonyOptimizerConfig,
    GeneticAlgorithmOptimizer,
    GeneticAlgorithmOptimizerConfig,
    GradientDescentOptimizer,
    GradientDescentOptimizerConfig,
    ParticleSwarmOptimizer,
    ParticleSwarmOptimizerConfig,
    StepWiseOptimizer,
    StepWiseOptimizerConfig,
    MultiTypeOptimizer,
    GroupedVariableOptimizer,
    GroupedVariableOptimizerConfig,
)

# Combinatorial (city-permutation / TSP-family) optimizers
from optimizers.combinatorial import (
    AntColonyTSP,
    AntColonyTSPConfig,
    AntColonyMST,
    GeneticAlgorithmTSP,
    GeneticAlgorithmTSPConfig,
    AntColonyMTSP,
    AntColonyMTSPConfig,
    TwoOptTSP,
    TwoOptTSPConfig,
    ThreeOptTSP,
    LinKernighanTSP,
    LinKernighanTSPConfig,
    NearestNeighborTSP,
    NearestNeighborTSPConfig,
    ConvexHullTSP,
    ConvexHullTSPConfig,
    compare_tsp_heuristics,
    format_comparison_table,
)

# Quality-diversity / MAP-Elites archive (see docs/history/QD_PARETO_PLAN.md)
from optimizers.archive import CVTArchive, QDReport, qd_score, pareto_front, hypervolume

# Shared config/result/base types and the solution archive every solver uses
from optimizers.core.base import IOptimizerConfig, OptimizerResult, BaseOptimizer
from optimizers.solution_deck import SolutionDeck

# Checkpointing
from optimizers.checkpoint import (
    CheckpointConfig,
    save_checkpoint,
    load_checkpoint,
    run_multiple,
)

# Plotting
from optimizers.plot import (
    plot_convergence,
    plot_cities_and_route,
    plot_run_statistics,
    plot_pareto_front,
    plot_map_elites,
    plot_benchmark_timings,
    set_show_plots,
    show_plots_enabled,
)

from optimizers.core.random import set_seed, get_seed

__all__ = [
    # Continuous
    "AntColonyOptimizer",
    "AntColonyOptimizerConfig",
    "GeneticAlgorithmOptimizer",
    "GeneticAlgorithmOptimizerConfig",
    "GradientDescentOptimizer",
    "GradientDescentOptimizerConfig",
    "ParticleSwarmOptimizer",
    "ParticleSwarmOptimizerConfig",
    "StepWiseOptimizer",
    "StepWiseOptimizerConfig",
    "MultiTypeOptimizer",
    "GroupedVariableOptimizer",
    "GroupedVariableOptimizerConfig",
    # Combinatorial
    "AntColonyTSP",
    "AntColonyTSPConfig",
    "AntColonyMST",
    "GeneticAlgorithmTSP",
    "GeneticAlgorithmTSPConfig",
    "AntColonyMTSP",
    "AntColonyMTSPConfig",
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
    # Quality-diversity / MAP-Elites
    "CVTArchive",
    "QDReport",
    "qd_score",
    "pareto_front",
    "hypervolume",
    # Shared types
    "IOptimizerConfig",
    "OptimizerResult",
    "BaseOptimizer",
    "SolutionDeck",
    # Checkpointing
    "CheckpointConfig",
    "save_checkpoint",
    "load_checkpoint",
    "run_multiple",
    # Plotting
    "plot_convergence",
    "plot_cities_and_route",
    "plot_run_statistics",
    "plot_pareto_front",
    "plot_map_elites",
    "plot_benchmark_timings",
    "set_show_plots",
    "show_plots_enabled",
    # RNG
    "set_seed",
    "get_seed",
]
