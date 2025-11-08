import logging
import random
from dataclasses import dataclass
from typing import Literal, get_args, Optional, List
from abc import ABC, abstractmethod

import numpy as np

from ..core.base import (
    IOptimizerConfig,
    OptimizerResult,
    create_from_dict,
    ensure_literal_choice,
    literal_options,
    GoalFcn,
    InputArguments,
)
from .base import IOptimizer
from ..solution_deck import InputVariables
from .aco import AntColonyOptimizer, AntColonyOptimizerConfig
from .pso import ParticleSwarmOptimizer, ParticleSwarmOptimizerConfig
from .ga import GeneticAlgorithmOptimizer, GeneticAlgorithmOptimizerConfig
from .gd import GradientDescentOptimizer, GradientDescentOptimizerConfig
from ..core.types import AF

OptimizationType = Literal["aco", "pso", "ga", "gd"]


def config_to_type(
    config: IOptimizerConfig, to_type: OptimizationType
) -> (
    AntColonyOptimizerConfig
    | ParticleSwarmOptimizerConfig
    | GeneticAlgorithmOptimizerConfig
    | GradientDescentOptimizerConfig
):
    ensure_literal_choice("to_type", to_type, OptimizationType)
    if to_type == "aco":
        # If you want to use default meta-parameters, you can just instantiate without extra fields
        return create_from_dict(config.__dict__, AntColonyOptimizerConfig)
    elif to_type == "pso":
        return create_from_dict(config.__dict__, ParticleSwarmOptimizerConfig)
    elif to_type == "ga":
        return create_from_dict(config.__dict__, GeneticAlgorithmOptimizerConfig)
    elif to_type == "gd":
        return create_from_dict(config.__dict__, GradientDescentOptimizerConfig)
    else:
        # Should be unreachable due to ensure_literal_choice above, but keep as safety
        allowed = ", ".join(repr(x) for x in literal_options(OptimizationType))
        raise ValueError(f"Invalid to_type={to_type!r}. Allowed options: {allowed}")


class IOptimizerSelection(ABC):
    @abstractmethod
    def select(
        self, existing_optim: Optional[OptimizationType] = None
    ) -> OptimizationType:
        pass


class RandomOptimizerSelection(IOptimizerSelection):
    def select(
        self, existing_optim: Optional[OptimizationType] = None
    ) -> OptimizationType:
        choices = list(get_args(OptimizationType))
        if existing_optim is not None:
            choices.remove(existing_optim)
        # TODO - Find a better way to select optimizers
        return random.choice(choices)


class MultiTypeOptimizer(IOptimizer):
    def __init__(
        self,
        config: IOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: InputArguments | None = None,
        initial_optimizer: OptimizationType = "aco",
        optimizer_selector: IOptimizerSelection = RandomOptimizerSelection(),
    ):
        super().__init__(config, fcn, variables, args)
        self.initial_optimizer = initial_optimizer
        self.optimizer_selector = optimizer_selector
        self.fcn = fcn
        self.optimizer_choice_history = []

    def solve(
        self,
        preserve_percent: float = 0.0,
        restart_count: int = 0,
        max_restart: int = 5,
        generations_completed: int = 0,
    ) -> OptimizerResult:
        selected_type = (
            self.optimizer_selector.select(self.optimizer_choice_history[-1])
            if restart_count > 0
            else self.initial_optimizer
        )
        # Validate selection
        ensure_literal_choice("selected_type", selected_type, OptimizationType)
        self.optimizer_choice_history.append(selected_type)
        logging.info(f"Selected optimizer: {selected_type}")
        converted_config = config_to_type(self.config, selected_type)
        # Ensure we do not exceed the total number of generations
        converted_config.num_generations = max(
            1, converted_config.num_generations - generations_completed
        )

        # TODO - Make sure we share the solution deck
        optimizer: IOptimizer
        if selected_type == "aco":
            optimizer = AntColonyOptimizer(
                converted_config, self.fcn, self.variables, self.args
            )
        elif selected_type == "pso":
            optimizer = ParticleSwarmOptimizer(
                converted_config, self.fcn, self.variables, self.args
            )
        elif selected_type == "ga":
            optimizer = GeneticAlgorithmOptimizer(
                converted_config, self.fcn, self.variables, self.args
            )
        elif selected_type == "gd":
            optimizer = GradientDescentOptimizer(
                converted_config, self.fcn, self.variables, self.args
            )
        else:
            # Should be unreachable due to ensure_literal_choice
            allowed = ", ".join(repr(x) for x in literal_options(OptimizationType))
            raise ValueError(
                f"Invalid selected_type={selected_type!r}. Allowed options: {allowed}"
            )

        result = optimizer.solve(preserve_percent=0.0 if restart_count == 0 else 0.1)
        if result.stop_reason == "no_improvement" and restart_count < max_restart:
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


@dataclass
class InputVariableGroup:
    name: str
    variables: List[str]
    optimizer_type: OptimizationType = "aco"


@dataclass
class GroupedVariableOptimizerConfig(IOptimizerConfig):
    num_rounds: int = 5
    """Number of rounds to run"""
    groups: list[InputVariableGroup] = None
    """List of input variable groups to optimize, in the order in which to optimize them"""


class GroupedVariableOptimizer(IOptimizer):
    def __init__(
        self,
        config: GroupedVariableOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: InputArguments | None = None,
    ):
        super().__init__(config, fcn, variables, args)
        self.config: GroupedVariableOptimizerConfig = config
        if config.groups is None:
            raise ValueError("Group order and groups must be provided")

    def interleave_variables(self, group: InputVariableGroup, x: AF, y: AF) -> AF:
        x_i = 0
        for i, var in enumerate(self.variables):
            if var.name in group.variables:
                y[i] = x[x_i]
                x_i += 1
        return y

    def solve(self, preserve_percent: float = 0.0) -> OptimizerResult:
        # TODO - Progress bar?
        # TODO - Pass in previous best solution deck
        # TODO - Support for check-pointing!
        default_values = [var.initial_value for var in self.variables]
        for cur_round in range(self.config.num_rounds):
            for group in self.config.groups:
                group_vars = [v for v in self.variables if v.name in group.variables]

                def new_fcn(x):
                    y = np.array(default_values)
                    y = self.interleave_variables(group, x, y)
                    return self.wrapped_fcn(y)

                config = config_to_type(self.config, group.optimizer_type)
                optimizer: IOptimizer
                if group.optimizer_type == "aco":
                    optim = AntColonyOptimizer(config, new_fcn, group_vars)
                elif group.optimizer_type == "pso":
                    optim = ParticleSwarmOptimizer(config, new_fcn, group_vars)
                elif group.optimizer_type == "ga":
                    optim = GeneticAlgorithmOptimizer(config, new_fcn, group_vars)
                elif group.optimizer_type == "gd":
                    optim = GradientDescentOptimizer(config, new_fcn, group_vars)
                else:
                    raise NotImplementedError("Optimizer not implemented")
                result = optim.solve()
                # TODO - Update the solution deck here?
                default_values = list(
                    self.interleave_variables(
                        group, result.solution_vector, default_values
                    )
                )
        return OptimizerResult(
            solution_vector=np.array(default_values),
            solution_score=self.wrapped_fcn(np.array(default_values)),
            solution_history=None,
            stop_reason="max_iterations",
        )
