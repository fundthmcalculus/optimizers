from typing import Literal

from optimizer_base import IOptimizer, IOptimizerConfig
import random
from abc import ABC, abstractmethod

from solution_deck import GoalFcn, InputArguments, InputVariables

StochasticOptimType = Literal["aco", "pso", "ga", "gd"]

class IOptimizerSelection(ABC):
    @abstractmethod
    def select(self) -> StochasticOptimType:
        pass

class RandomOptimizerSelection(IOptimizerSelection):
    def select(self) -> StochasticOptimType:
        # TODO - Find a better way to select optimizers
        return random.choice(["aco", "pso", "ga", "gd"])


class MultiTypeOptimizer(IOptimizer):
    def __init__(
        self,
        name: str,
        config: IOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: InputArguments | None = None,
        initial_optimizer: StochasticOptimType = "aco",
        optimizer_selector: IOptimizerSelection = RandomOptimizerSelection(),
    ):
        super().__init__(name, config, fcn, variables, args)
        self.initial_optimizer = initial_optimizer
        self.optimizer_selector = optimizer_selector

    def solve(self) -> OptimizerResult: