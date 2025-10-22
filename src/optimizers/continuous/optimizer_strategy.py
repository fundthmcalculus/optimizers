import logging
import random
from typing import Literal
from abc import ABC, abstractmethod

from optimizers.core.base import IOptimizerConfig, OptimizerResult
from .base import OptimizerBase
from optimizers.solution_deck import GoalFcn, InputArguments, InputVariables
from .aco import AntColonyOptimizer, AntColonyOptimizerConfig
from .pso import ParticleSwarmOptimizer, ParticleSwarmOptimizerConfig
from .ga import GeneticAlgorithmOptimizer, GeneticAlgorithmOptimizerConfig
from .gd import GradientDescentOptimizer, GradientDescentOptimizerConfig

StochasticOptimType = Literal["aco", "pso", "ga", "gd"]


def config_to_type(
    config: IOptimizerConfig, to_type: StochasticOptimType
) -> (
    AntColonyOptimizerConfig
    | ParticleSwarmOptimizerConfig
    | GeneticAlgorithmOptimizerConfig
    | GradientDescentOptimizerConfig
):
    if to_type == "aco":
        if isinstance(config, AntColonyOptimizerConfig):
            return config
        else:
            # If you want to use default meta-parameters, you can just instantiate without extra fields
            return AntColonyOptimizerConfig(**{**config.__dict__})
    elif to_type == "pso":
        if isinstance(config, ParticleSwarmOptimizerConfig):
            return config
        else:
            return ParticleSwarmOptimizerConfig(**{**config.__dict__})
    elif to_type == "ga":
        if isinstance(config, GeneticAlgorithmOptimizerConfig):
            return config
        else:
            return GeneticAlgorithmOptimizerConfig(**{**config.__dict__})
    elif to_type == "gd":
        if isinstance(config, GradientDescentOptimizerConfig):
            return config
        else:
            return GradientDescentOptimizerConfig(**{**config.__dict__})
    else:
        raise ValueError(f"Unknown optimizer type: {to_type}")


class IOptimizerSelection(ABC):
    @abstractmethod
    def select(self) -> StochasticOptimType:
        pass


class RandomOptimizerSelection(IOptimizerSelection):
    def select(self) -> StochasticOptimType:
        # TODO - Find a better way to select optimizers
        return random.choice(["aco", "pso", "ga", "gd"])


class MultiTypeOptimizer(OptimizerBase):
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
        self.fcn = fcn

    def solve(
        self,
        preserve_percent: float = 0.0,
        restart_count: int = 0,
        max_restart: int = 5,
        generations_completed: int = 0,
    ) -> OptimizerResult:
        selected_type = (
            self.optimizer_selector.select()
            if restart_count > 0
            else self.initial_optimizer
        )
        logging.info(f"Selected optimizer: {selected_type}")
        converted_config = config_to_type(self.config, selected_type)
        # Ensure we do not exceed the total number of generations
        converted_config.num_generations = max(
            1, converted_config.num_generations - generations_completed
        )

        optimizer: OptimizerBase

        if selected_type == "aco":
            optimizer = AntColonyOptimizer(
                self.name, converted_config, self.fcn, self.variables, self.args
            )
        elif selected_type == "pso":
            optimizer = ParticleSwarmOptimizer(
                self.name, converted_config, self.fcn, self.variables, self.args
            )
        elif selected_type == "ga":
            optimizer = GeneticAlgorithmOptimizer(
                self.name, converted_config, self.fcn, self.variables, self.args
            )
        elif selected_type == "gd":
            optimizer = GradientDescentOptimizer(
                self.name, converted_config, self.fcn, self.variables, self.args
            )
        else:
            raise ValueError(f"Unknown optimizer type: {selected_type}")

        result = optimizer.solve(preserve_percent=0.0 if restart_count == 0 else 0.1)
        if result.stopped_early and restart_count < max_restart:
            # If the optimizer stopped early, we can try another optimizer
            logging.warning(
                f"Optimizer {selected_type} stopped early, selecting a new optimizer."
            )

            return result + self.solve(
                restart_count=restart_count + 1,
                generations_completed=generations_completed
                + result.generations_completed,
            )
        return result
