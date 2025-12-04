from dataclasses import dataclass

from .base import IOptimizer
from ..core import OptimizerResult
from ..core.base import (
    IOptimizerConfig,
    GoalFcn,
    InputArguments,
)
from ..core.variables import InputVariables
from ..solution_deck import (
    SolutionDeck,
)


@dataclass
class BlockAntColonyTSPConfig(IOptimizerConfig):
    rho: float = 0.2  # 0.451  # 0.5
    """Pheromone decay parameter"""
    alpha: float = 0.8  # 1.88  # 1.0
    """Pheromone deposit parameter"""
    beta: float = 2  # 1.88  # 1.0
    """Pheromone evaporation parameter"""
    q: float = 1  # 2.17  # 1.0
    num_partitions: int | list[int] = 5
    """Number of partitions for continuous input variables. If list, it is per-variable partitions"""


class BlockAntColonyOptimizer(IOptimizer):
    def __init__(
        self,
        config: IOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: InputArguments | None = None,
        existing_soln_deck: SolutionDeck | None = None,
    ):
        super().__init__(
            config,
            fcn,
            variables,
            args,
            existing_soln_deck,
        )
        self.config: BlockAntColonyTSPConfig = BlockAntColonyTSPConfig(
            **{**config.__dict__}
        )

    def solve(self, preserve_percent: float = 0.0) -> OptimizerResult:
        # 1. Partition each input variable