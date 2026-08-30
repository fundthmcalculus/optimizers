import logging
import os
import uuid
from dataclasses import dataclass
from typing import Literal, get_args, Optional, List
from abc import ABC, abstractmethod

import numpy as np

from ..core.random import rng
from ..core.base import (
    IOptimizerConfig,
    OptimizerResult,
    create_from_dict,
    ensure_literal_choice,
    literal_options,
    GoalFcn,
    InputArguments,
    single_vector,
)
from ..checkpoint import CheckpointConfig, load_checkpoint, save_checkpoint
from .base import IOptimizer
from ..core import InputVariables
from .aco import AntColonyOptimizer, AntColonyOptimizerConfig
from .pso import ParticleSwarmOptimizer, ParticleSwarmOptimizerConfig
from .ga import GeneticAlgorithmOptimizer, GeneticAlgorithmOptimizerConfig
from .gd import GradientDescentOptimizer, GradientDescentOptimizerConfig
from ..core.types import AF, AI

OptimizationType = Literal["aco", "pso", "ga", "gd"]


def config_to_type(
    config: IOptimizerConfig, to_type: OptimizationType
) -> (
    AntColonyOptimizerConfig
    | ParticleSwarmOptimizerConfig
    | GeneticAlgorithmOptimizerConfig
    | GradientDescentOptimizerConfig
):
    ensure_literal_choice(to_type, OptimizationType)
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
        # Annotated: ``get_args`` is typed ``tuple[Any, ...]``, so without this
        # the element type is Any and the declared Literal return is unchecked.
        choices: list[OptimizationType] = list(get_args(OptimizationType))
        if existing_optim is not None:
            choices.remove(existing_optim)
        # TODO - Find a better way to select optimizers
        return choices[int(rng().integers(len(choices)))]


class MultiTypeOptimizer(IOptimizer):
    def __init__(
        self,
        *,
        config: IOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: InputArguments | None = None,
        initial_optimizer: OptimizationType = "aco",
        optimizer_selector: IOptimizerSelection | None = None,
    ):
        super().__init__(
            config=config,
            fcn=fcn,
            variables=variables,
            args=args,
        )
        self.initial_optimizer = initial_optimizer
        self.optimizer_selector = (
            optimizer_selector
            if optimizer_selector is not None
            else RandomOptimizerSelection()
        )
        self.fcn = fcn
        self.optimizer_choice_history: list[OptimizationType] = []

    def solve(self, *, preserve_percent: float = 0.0) -> OptimizerResult:
        return self._solve_with_restarts(preserve_percent=preserve_percent)

    def _solve_with_restarts(
        self,
        *,
        preserve_percent: float = 0.0,
        restart_count: int = 0,
        max_restart: int = 5,
        generations_completed: int = 0,
    ) -> OptimizerResult:
        # ``preserve_percent`` is accepted for parity with ``solve()``'s public
        # signature but not threaded through here: each restart's delegate
        # optimizer picks its own preserve_percent below (0.0 on the first
        # attempt, 0.1 on restarts, to warm-start from the previous archive).
        selected_type = (
            self.optimizer_selector.select(self.optimizer_choice_history[-1])
            if restart_count > 0
            else self.initial_optimizer
        )
        # Validate selection
        ensure_literal_choice(selected_type, OptimizationType)
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
                config=converted_config,
                fcn=self.fcn,
                variables=self.variables,
                args=self.args,
            )
        elif selected_type == "pso":
            optimizer = ParticleSwarmOptimizer(
                config=converted_config,
                fcn=self.fcn,
                variables=self.variables,
                args=self.args,
            )
        elif selected_type == "ga":
            optimizer = GeneticAlgorithmOptimizer(
                config=converted_config,
                fcn=self.fcn,
                variables=self.variables,
                args=self.args,
            )
        elif selected_type == "gd":
            # Nested local search: this strategy owns the higher-level search, so
            # GD runs serially (see GradientDescentOptimizer's ``nested`` flag).
            optimizer = GradientDescentOptimizer(
                config=converted_config,
                fcn=self.fcn,
                variables=self.variables,
                args=self.args,
                nested=True,
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

            return result + self._solve_with_restarts(
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
    groups: list[InputVariableGroup] | None = None
    """List of input variable groups to optimize, in the order in which to optimize them"""


class GroupedVariableOptimizer(IOptimizer):
    def __init__(
        self,
        *,
        config: GroupedVariableOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: InputArguments | None = None,
        checkpoint_cfg: Optional[CheckpointConfig] = None,
    ):
        super().__init__(
            config=config,
            fcn=fcn,
            variables=variables,
            args=args,
        )
        self.config: GroupedVariableOptimizerConfig = config
        if config.groups is None:
            raise ValueError("Group order and groups must be provided")
        # Opt-in checkpointing (see ``solve``'s ``resume_from``): one JSON
        # blob saved per completed round, under one run_id for the whole run.
        self.checkpoint_cfg = checkpoint_cfg
        self._checkpoint_run_id = uuid.uuid4().hex if checkpoint_cfg else None

    def interleave_variables(self, group: InputVariableGroup, x: AF | AI, y: AF) -> AF:
        x_i = 0
        for i, var in enumerate(self.variables):
            if var.name in group.variables:
                y[i] = x[x_i]
                x_i += 1
        return y

    def solve(
        self,
        *,
        preserve_percent: float = 0.0,
        resume_from: str | os.PathLike[str] | None = None,
    ) -> OptimizerResult:
        # TODO - Progress bar?
        # TODO - Pass in previous best solution deck
        default_values = [var.initial_value for var in self.variables]
        start_round = 0
        if resume_from is not None:
            checkpoint = load_checkpoint(resume_from)
            saved_result = checkpoint.get("result")
            if saved_result and saved_result.get("solution_vector") is not None:
                default_values = list(saved_result["solution_vector"])
            start_round = int(checkpoint.get("metadata", {}).get("round", -1)) + 1
            logging.info(
                f"Resuming GroupedVariableOptimizer from round {start_round} "
                f"(checkpoint {resume_from})"
            )

        assert self.config.groups is not None  # validated in __init__
        last_score: Optional[float] = None
        for cur_round in range(start_round, self.config.num_rounds):
            for group in self.config.groups:
                group_vars = [v for v in self.variables if v.name in group.variables]

                def new_fcn(x: AF) -> float:
                    y = np.array(default_values)
                    y = self.interleave_variables(group, x, y)
                    return self.wrapped_fcn(y)

                config = config_to_type(self.config, group.optimizer_type)
                optim: IOptimizer
                if group.optimizer_type == "aco":
                    optim = AntColonyOptimizer(
                        config=config, fcn=new_fcn, variables=group_vars
                    )
                elif group.optimizer_type == "pso":
                    optim = ParticleSwarmOptimizer(
                        config=config, fcn=new_fcn, variables=group_vars
                    )
                elif group.optimizer_type == "ga":
                    optim = GeneticAlgorithmOptimizer(
                        config=config, fcn=new_fcn, variables=group_vars
                    )
                elif group.optimizer_type == "gd":
                    # Nested local search: the grouped optimizer owns the outer
                    # loop over groups/rounds, so GD runs serially (see its
                    # ``nested`` flag).
                    optim = GradientDescentOptimizer(
                        config=config, fcn=new_fcn, variables=group_vars, nested=True
                    )
                else:
                    raise NotImplementedError("Optimizer not implemented")
                result = optim.solve()
                # TODO - Update the solution deck here?
                default_values = list(
                    self.interleave_variables(
                        group,
                        single_vector(result),
                        np.asarray(default_values, dtype=float),
                    )
                )

            is_last_round = cur_round == self.config.num_rounds - 1
            if self.checkpoint_cfg is not None and self.checkpoint_cfg.enabled:
                last_score = self.wrapped_fcn(np.array(default_values))
                save_checkpoint(
                    self.checkpoint_cfg,
                    optimizer_name=self.config.name or "grouped-variable",
                    config=self.config,
                    result=OptimizerResult(
                        solution_vector=np.array(default_values),
                        solution_score=last_score,
                        stop_reason="none",
                        generations_completed=cur_round + 1,
                    ),
                    run_id=self._checkpoint_run_id,
                    metadata={"round": cur_round},
                )
            elif is_last_round:
                last_score = self.wrapped_fcn(np.array(default_values))

        if last_score is None:
            # resume_from started at or past num_rounds -- nothing left to run.
            last_score = self.wrapped_fcn(np.array(default_values))
        return OptimizerResult(
            solution_vector=np.array(default_values),
            solution_score=last_score,
            solution_history=None,
            stop_reason="max_iterations",
        )
