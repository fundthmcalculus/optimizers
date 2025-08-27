from dataclasses import dataclass

import joblib
import numpy as np
import tqdm

from local import apply_local_optimization
from optimizer_base import (
    IOptimizer,
    OptimizerResult,
    IOptimizerConfig,
    InputArguments,
    setup_for_generations,
    check_stop_early,
)

from solution_deck import GoalFcn, LocalOptimType, InputVariables, SolutionDeck

@dataclass
class GeneticAlgorithmOptimizerConfig(IOptimizerConfig):
    mutation_rate: float = 0.1
    """Probability of mutation"""
    crossover_rate: float = 0.8
    """Probability of crossover"""
    local_grad_optim: LocalOptimType = "none"


def tournament_selection(
    population_deck: np.ndarray,
    population_fitness: np.ndarray,
    tournament_size: int = 3,
) -> np.ndarray:
    # Randomly sample row
    row_idxs = np.random.choice(
        len(population_deck), size=tournament_size, replace=False
    )
    # Take the optimal row from the option
    row_idx_sort = np.argsort(population_fitness[row_idxs])
    return population_deck[row_idxs[row_idx_sort[0]], :]


def crossover(
    parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    # Randomly pick a point in the array that the swap starts
    if np.random.rand() < crossover_rate:
        crossover_idx = np.random.choice(len(parent1))
        child1 = np.concatenate(
            [parent1[:crossover_idx], parent2[crossover_idx:]], axis=0
        )
        child2 = np.concatenate(
            [parent2[:crossover_idx], parent1[crossover_idx:]], axis=0
        )
        return child1, child2
    else:
        return parent1, parent2


def mutate(
    child: np.ndarray, mutation_rate: float, variables: InputVariables
) -> np.ndarray:
    mutant_child = np.copy(child)
    for ij, variable in enumerate(variables):
        if np.random.random() < mutation_rate:
            mutant_child[ij] = variable.perturb_value(mutant_child[ij])
    return mutant_child


def run_ga(
    n_steps: int,
    mutation_rate: float,
    crossover_rate: float,
    local_optim: LocalOptimType,
    solution_values: np.ndarray,
    solution_archive: np.ndarray,
    variables: InputVariables,
    fcn: GoalFcn,
) -> tuple[np.ndarray, np.ndarray]:
    new_population = np.zeros((n_steps, len(variables)))
    new_population_fitness = np.zeros(n_steps)
    for row in range(n_steps):
        # Take two parents
        parent_1 = tournament_selection(solution_archive, solution_values)
        parent_2 = tournament_selection(solution_archive, solution_values)
        # Perform genetic operations.
        child_1, child_2 = crossover(parent_1, parent_2, crossover_rate)
        child_1 = mutate(child_1, mutation_rate, variables)
        child_2 = mutate(child_2, mutation_rate, variables)
        # Optimize child-1, because firstborn rights.
        child_1, child_1_fitness = apply_local_optimization(
            fcn, local_optim, child_1, variables
        )
        child_2, child_2_fitness = apply_local_optimization(
            fcn, local_optim, child_2, variables
        )
        if child_1_fitness < child_2_fitness:
            new_population[row, :] = child_1
            new_population_fitness[row] = child_1_fitness
        else:
            new_population[row, :] = child_2
            new_population_fitness[row] = child_2_fitness
    return new_population, new_population_fitness


class GeneticAlgorithmOptimizer(IOptimizer):
    def __init__(
        self,
        name: str,
        config: GeneticAlgorithmOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: InputArguments | None = None,
        existing_soln_deck: SolutionDeck | None = None,
    ):
        super().__init__(name, config, fcn, variables, args, existing_soln_deck)
        self.config: GeneticAlgorithmOptimizerConfig = config

    def solve(self, preserve_percent: float = 0.0) -> OptimizerResult:
        # Optimal solution history
        best_soln_history = np.zeros(self.config.num_generations)

        self.soln_deck.initialize_solution_deck(self.variables, self.wrapped_fcn, preserve_percent)
        self.soln_deck.sort()

        generation_pbar, individuals_per_job, n_jobs, parallel = setup_for_generations(
            self.config
        )
        stopped_early = False
        generations_completed = 0
        for generations_completed in generation_pbar:
            stopped_early = check_stop_early(self.config, best_soln_history, self.soln_deck.solution_value)
            if stopped_early:
                break

            job_output = parallel(
                joblib.delayed(run_ga)(
                    individuals_per_job,
                    local_optim=self.config.local_grad_optim,
                    mutation_rate=self.config.mutation_rate,
                    crossover_rate=self.config.crossover_rate,
                    solution_values=self.soln_deck.solution_value,
                    solution_archive=self.soln_deck.solution_archive,
                    variables=self.variables,
                    fcn=self.wrapped_fcn,
                )
                for _ in range(n_jobs)
            )
            for output in job_output:
                ant_solutions = output[0]
                ant_values = output[1]
                self.soln_deck.append(ant_solutions, ant_values, self.config.local_grad_optim != "none")
                self.soln_deck.deduplicate()
            generation_pbar.set_postfix(best_value=self.soln_deck.solution_value[0])

        return OptimizerResult(
            solution_vector=self.soln_deck.solution_archive[0, :],
            solution_score=self.soln_deck.solution_value[0],
            solution_history=best_soln_history,
            stopped_early=stopped_early
            generations_completed=generations_completed + 1
        )
