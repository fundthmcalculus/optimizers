from typing import Literal

StochasticOptimType = Literal["aco", "pso", "ga", "gd"]

class IOptimizerStrategy:
    def __init__(self, strategy_type: StochasticOptimType):
        self.strategy_type = strategy_type