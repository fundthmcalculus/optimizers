from dataclasses import dataclass

import joblib
import numpy as np

from .local import apply_local_optimization
from optimizers.core.base import (
    IOptimizerConfig,
    OptimizerResult,
)
from optimizers.continuous.base import setup_for_generations, check_stop_early, cdf
from optimizers.core.types import af64
from optimizers.solution_deck import (
    GoalFcn,
    LocalOptimType,
    SolutionDeck,
    WrappedGoalFcn,
    InputArguments,
)

from .base import OptimizerBase
from ..core.variables import InputVariables


@dataclass
class AntColonyOptimizerConfig(IOptimizerConfig):
    learning_rate: float = 0.7
    """Learning rate for updating pheromone trails"""
    q: float = 1.0
    """Weighting parameter for better ranked solutions"""
    local_grad_optim: LocalOptimType = "none"


def run_ants(
    n_ants: int,
    q_weight: float,
    learning_rate: float,
    local_optim: LocalOptimType,
    solution_archive: af64,
    variables: InputVariables,
    fcn: WrappedGoalFcn,
) -> tuple[af64, af64]:
    cp_j = cdf(q_weight, len(solution_archive))
    ant_solutions = np.zeros((n_ants, len(variables)))
    ant_values = np.zeros(n_ants)
    for ant in range(n_ants):
        new_solution = np.zeros(len(variables))
        # Generate a new solution from an existing one as a base
        p = np.random.default_rng().uniform()
        # Find the entry based upon cdf
        base_solution_idx = np.searchsorted(cp_j, p)
        base_solution = solution_archive[base_solution_idx, :]
        for i, variable in enumerate(variables):
            # Compute the weighted value for the variable
            new_solution[i] = variable.random_value(
                current_value=base_solution[i],
                other_values=solution_archive[:, i],
                learning_rate=learning_rate,
            )
        new_solution, new_value = apply_local_optimization(
            fcn, local_optim, new_solution, variables
        )

        # Store the new solution in the temporary archive.
        ant_solutions[ant, :] = new_solution
        ant_values[ant] = new_value
    return ant_solutions, ant_values


class AntColonyOptimizer(OptimizerBase):
    def __init__(
        self,
        name: str,
        config: IOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: InputArguments | None = None,
        existing_soln_deck: SolutionDeck | None = None,
    ):
        super().__init__(name, config, fcn, variables, args, existing_soln_deck)
        self.config: AntColonyOptimizerConfig = AntColonyOptimizerConfig(
            **{**config.__dict__}
        )

    def solve(self, preserve_percent: float = 0.0) -> OptimizerResult:
        self.validate_config()
        self.soln_deck.initialize_solution_deck(
            self.variables, self.wrapped_fcn, preserve_percent
        )
        self.soln_deck.sort()
        best_soln_history = np.zeros(self.config.num_generations)

        # Add the progress bar
        generation_pbar, individuals_per_job, n_jobs, parallel = setup_for_generations(
            self.config
        )
        stopped_early = False
        generations_completed = 0
        for generations_completed in generation_pbar:
            stopped_early = check_stop_early(
                self.config, best_soln_history, self.soln_deck.solution_value
            )
            if stopped_early:
                break

            job_output = parallel(
                joblib.delayed(run_ants)(
                    individuals_per_job,
                    self.config.q,
                    self.config.learning_rate,
                    self.config.local_grad_optim,
                    self.soln_deck.solution_archive,
                    self.variables,
                    self.wrapped_fcn,
                )
                for _ in range(n_jobs)
            )

            # Merge candidates into the archive
            for output in job_output:
                output_solutions = output[0]
                output_values = output[1]
                self.soln_deck.append(
                    output_solutions,
                    output_values,
                    self.config.local_grad_optim != "none",
                )
                self.soln_deck.deduplicate()
            generation_pbar.set_postfix(best_value=self.soln_deck.solution_value[0])

        # Return the best solution
        return OptimizerResult(
            solution_vector=self.soln_deck.solution_archive[0, :],
            solution_score=self.soln_deck.solution_value[0],
            solution_history=best_soln_history,
            stopped_early=stopped_early,
            generations_completed=generations_completed + 1,
        )
