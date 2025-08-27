from dataclasses import dataclass

import joblib
import numpy as np

from local import apply_local_optimization
from optimizer_base import (
    IOptimizer,
    IOptimizerConfig,
    OptimizerResult,
    InputArguments,
    setup_for_generations,
    check_stop_early,
    cdf
)
from opt_types import af64
from solution_deck import GoalFcn, LocalOptimType, InputVariables, SolutionDeck


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
    fcn: GoalFcn,
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


class AntColonyOptimizer(IOptimizer):
    def __init__(
        self,
        name: str,
        config: AntColonyOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: InputArguments | None = None,
        existing_soln_deck: SolutionDeck | None = None,
    ):
        super().__init__(name, config, fcn, variables, args, existing_soln_deck)
        # This is a rewrite for type hinting purposes
        self.config: AntColonyOptimizerConfig = config

    def solve(self, preserve_percent: float = 0.0) -> OptimizerResult:
        self.validate_config()
        self.soln_deck.initialize_solution_deck(self.variables, self.wrapped_fcn, preserve_percent)
        self.soln_deck.sort()
        best_soln_history = np.zeros(self.config.num_generations)

        # Add the progress bar
        generation_pbar, individuals_per_job, n_jobs, parallel = setup_for_generations(
            self.config
        )
        is_local_optima = np.ones(len(self.soln_deck.solution_value), dtype=bool)
        if self.config.local_grad_optim == "none":
            is_local_optima = np.zeros(len(self.soln_deck.solution_value), dtype=bool)
        stopped_early = False
        generations_completed = 0
        for generations_completed in generation_pbar:
            stopped_early = check_stop_early(self.config, best_soln_history, self.soln_deck.solution_value)
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
            # Get all the solutions, not just the first runner's worth
            for output in job_output:
                ant_solutions = output[0]
                ant_values = output[1]
                self.soln_deck.append(ant_solutions, ant_values, self.config.local_grad_optim != "none")
                self.soln_deck.deduplicate()
            generation_pbar.set_postfix(best_value=self.soln_deck.solution_value[0])

        # Return the best solution
        return OptimizerResult(
            solution_vector=self.soln_deck.solution_archive[0, :],
            solution_score=self.soln_deck.solution_value[0],
            solution_history=best_soln_history,
            stopped_early=stopped_early,
            generations_completed=generations_completed + 1
        )

    def fill_solution_archive(
        self,
        solution_archive: af64,
        solution_values: af64,
    ) -> None:
        for k in range(self.config.solution_archive_size):
            for i, variable in enumerate(self.variables):
                solution_archive[k, i] = variable.initial_random_value()
            solution_values[k] = self.wrapped_fcn(solution_archive[k])
        # insert the initial solutions to the archive
        for i, variable in enumerate(self.variables):
            solution_archive[0, i] = variable.initial_value
        solution_values[0] = self.wrapped_fcn(solution_archive[0])

    def create_solution_archive(self) -> tuple[af64, af64]:
        # Construct the solution archive
        solution_archive = np.zeros(
            (self.config.solution_archive_size, len(self.variables))
        )
        solution_values = np.zeros(self.config.solution_archive_size)
        return solution_archive, solution_values

    def validate_config(self):
        # Set the default values for the config
        if self.config.solution_archive_size < 0:
            self.config.solution_archive_size = len(self.variables) * 2
        if self.config.population_size < 0:
            self.config.population_size = self.config.solution_archive_size // 3
