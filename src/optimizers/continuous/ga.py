from dataclasses import dataclass

import joblib
import numpy as np

from .local import apply_local_optimization
from ..core.types import AF, AI
from ..core.base import (
    OptimizerResult,
    IOptimizerConfig,
    OptimizerRun,
    LocalOptimType,
    GoalFcn,
    InputArguments,
)
from .base import (
    check_stop_early,
)
from ..core.variables import InputVariables
from .base import IOptimizer

from ..solution_deck import (
    SolutionDeck,
    WrappedGoalFcn,
)


@dataclass
class GeneticAlgorithmOptimizerConfig(IOptimizerConfig):
    mutation_rate: float = 0.1
    """Probability of mutation"""
    crossover_rate: float = 0.8
    """Probability of crossover"""


def _tournament_selection(
    population_deck: AF | AI,
    population_fitness: AF,
    tournament_size: int = 3,
) -> AF | AI:
    # Randomly sample row
    row_idxs = np.random.choice(
        len(population_deck), size=tournament_size, replace=False
    )
    # Take the optimal row from the option
    row_idx_sort = np.argsort(population_fitness[row_idxs])
    return population_deck[row_idxs[row_idx_sort[0]], :]


def _crossover(
    parent1: AF | AI, parent2: AF | AI, crossover_rate: float
) -> tuple[AF | AI, AF | AI]:
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


def _mutate(child: AF | AI, mutation_rate: float, variables: InputVariables) -> AF | AI:
    mutant_child: AF | AI = np.copy(child)
    for ij, variable in enumerate(variables):
        if np.random.random() < mutation_rate:
            mutant_child[ij] = variable.perturb_value(mutant_child[ij])
    return mutant_child


def run_ga(
    n_steps: int,
    mutation_rate: float,
    crossover_rate: float,
    local_optim: LocalOptimType,
    solution_values: AF | AI,
    solution_archive: AF,
    variables: InputVariables,
    fcn: WrappedGoalFcn,
) -> OptimizerRun:
    new_population = np.zeros((n_steps, len(variables)))
    new_population_fitness = np.zeros(n_steps)
    for row in range(n_steps):
        # Take two parents
        parent_1 = _tournament_selection(solution_archive, solution_values)
        parent_2 = _tournament_selection(solution_archive, solution_values)
        # Perform genetic operations.
        child_1, child_2 = _crossover(parent_1, parent_2, crossover_rate)
        child_1 = _mutate(child_1, mutation_rate, variables)
        child_2 = _mutate(child_2, mutation_rate, variables)
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
    return OptimizerRun(
        population_solutions=new_population, population_values=new_population_fitness
    )


class GeneticAlgorithmOptimizer(IOptimizer):
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
        self.config: GeneticAlgorithmOptimizerConfig = GeneticAlgorithmOptimizerConfig(
            **{**config.__dict__}
        )

    def solve(self, preserve_percent: float = 0.0) -> OptimizerResult:
        (
            best_soln_history,
            generation_pbar,
            generations_completed,
            individuals_per_job,
            n_jobs,
            parallel,
            stopped_early,
        ) = self.initialize(preserve_percent)
        for generations_completed in generation_pbar:
            # Update runtime metadata for this generation
            self._set_phase("evolve")
            self._set_generation(generations_completed)

            stopped_early = check_stop_early(
                self.config, best_soln_history, self.soln_deck.solution_value
            )
            if stopped_early != "none":
                break

            job_output: list[OptimizerRun] = parallel(
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

            # Merge candidates into the archive
            self.update_solution_deck(generation_pbar, job_output)
            best_soln_history.append(self.soln_deck.get_best()[1])

        # Mark finalize phase
        self._set_phase("finalize")

        stopped_early = stopped_early if stopped_early != "none" else "max_iterations"
        # Return the best solution, including constraint metrics and unconstrained best
        best_x, best_val, _ = self.soln_deck.get_best()
        return OptimizerResult(
            solution_vector=best_x,
            solution_score=best_val,
            solution_history=np.array(best_soln_history),
            stop_reason=stopped_early,
            generations_completed=generations_completed + 1,
        )
