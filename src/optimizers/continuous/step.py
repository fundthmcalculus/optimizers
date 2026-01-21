from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .base import IOptimizerConfig, IOptimizer
from .local import local_perturb_optim
from ..core import InputVariables, OptimizerResult
from ..core.base import GoalFcn, InputArguments
from ..core.types import F
from ..solution_deck import SolutionDeck


@dataclass
class StepWiseOptimizerConfig(IOptimizerConfig):
    optimize_whole_solution_deck: bool = False
    max_perturbation: float = 0.1  # Fraction of domain
    pass


class StepWiseOptimizer(IOptimizer):
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
        self.config: StepWiseOptimizerConfig = StepWiseOptimizerConfig(
            **{**config.__dict__}
        )

    def solve(self, preserve_percent: float = 0.0) -> OptimizerResult:
        best_soln_value: list[F] = list()
        # Start with the initial value from the input variables, and stepwise refine solve
        if not self.config.optimize_whole_solution_deck:
            x0 = np.array([v.initial_random_value(0.0) for v in self.variables])
            stop_reason = "max_iterations"
            for gen in tqdm(
                range(self.config.num_generations),
                desc="Stepwise optimization generations",
            ):
                x0, x0_val = local_perturb_optim(
                    self.wrapped_fcn, x0, self.variables, self.config.max_perturbation
                )
                if len(best_soln_value) == 0:
                    best_soln_value.append(x0_val)
                else:
                    best_soln_value.append(min(min(best_soln_value), x0_val))
                if gen >= 2 and np.allclose(
                    best_soln_value[-1], best_soln_value[-2], atol=1e-2, rtol=1e-2
                ):
                    stop_reason = "no_improvement"
                    break

            return OptimizerResult(
                solution_score=best_soln_value[-1],
                solution_history=np.array(best_soln_value),
                solution_vector=x0,
                stop_reason=stop_reason,
            )
        else:
            best_soln_vector = None
            for soln_idx in tqdm(
                range(self.soln_deck.archive_size), desc="Solution Deck Entry"
            ):
                cur_best_soln_value: list[F] = list()
                x0, x0_val, _ = self.soln_deck.get(soln_idx)
                stop_reason = "max_iterations"
                for gen in tqdm(
                    range(self.config.num_generations),
                    desc="Stepwise optimization generations",
                ):
                    cur_best_soln_value.append(min(min(cur_best_soln_value), x0_val))
                    if gen >= 2 and np.allclose(
                        cur_best_soln_value[-1],
                        cur_best_soln_value[-2],
                        atol=1e-2,
                        rtol=1e-2,
                    ):
                        stop_reason = "no_improvement"
                        break
                self.soln_deck.set(
                    soln_idx, x0, x0_val, stop_reason == "no_improvement"
                )
                if (
                    cur_best_soln_value[-1] < best_soln_value[-1]
                    or len(best_soln_value) == 0
                ):
                    best_soln_vector = x0
                    best_soln_value.append(cur_best_soln_value[-1])

            return OptimizerResult(
                solution_score=best_soln_value[-1],
                solution_history=np.array(best_soln_value),
                solution_vector=best_soln_vector,
                stop_reason="none",  # TODO - Is there a better method here?
            )
