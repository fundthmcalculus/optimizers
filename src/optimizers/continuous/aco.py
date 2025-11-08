from dataclasses import dataclass

import joblib
import numpy as np

from .local import apply_local_optimization
from ..core.base import (
    IOptimizerConfig,
    OptimizerResult,
    OptimizerRun,
    LocalOptimType,
    GoalFcn,
    InputArguments,
)
from ..continuous.base import check_stop_early, cdf
from ..core.types import af64
from ..solution_deck import (
    SolutionDeck,
    WrappedGoalFcn,
)
from ..core.random import rng as global_rng

from .base import IOptimizer
from ..core.variables import InputVariables


@dataclass
class AntColonyOptimizerConfig(IOptimizerConfig):
    learning_rate: float = 0.7
    """Learning rate for updating pheromone trails"""
    q: float = 1.0
    """Weighting parameter for better ranked solutions"""


def run_ants(
    n_ants: int,
    q_weight: float,
    learning_rate: float,
    local_optim: LocalOptimType,
    solution_archive: af64,
    variables: InputVariables,
    fcn: WrappedGoalFcn,
) -> OptimizerRun:
    cp_j = cdf(q_weight, len(solution_archive))
    ant_solutions = np.zeros((n_ants, len(variables)))
    ant_values = np.zeros(n_ants)
    for ant in range(n_ants):
        new_solution = np.zeros(len(variables))
        # Generate a new solution from an existing one as a base
        p = global_rng().uniform()
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
    return OptimizerRun(
        population_values=ant_values, population_solutions=ant_solutions
    )


class AntColonyOptimizer(IOptimizer):
    def __init__(
        self,
        config: IOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: InputArguments | None = None,
        existing_soln_deck: SolutionDeck | None = None,
        inequality_constraints: list[GoalFcn] | None = None,
        equality_constraints: list[GoalFcn] | None = None,
    ):
        super().__init__(
            config,
            fcn,
            variables,
            args,
            existing_soln_deck,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )
        self.config: AntColonyOptimizerConfig = AntColonyOptimizerConfig(
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
            self.update_solution_deck(generation_pbar, job_output)
            best_soln_history.append(self.soln_deck.get_best()[1])
        # Mark finalize phase
        self._set_phase("finalize")
        stopped_early = stopped_early if stopped_early != "none" else "max_iterations"

        # Return the best solution, including constraint metrics and unconstrained best
        best_x, best_val, _ = self.soln_deck.get_best()
        ineq_vals, eq_vals, ineq_rel, eq_rel, total = (
            self.soln_deck.get_constraint_results(0)
        )
        ub_x, ub_val, _ = self.soln_deck.get_best_unconstrained()
        return OptimizerResult(
            solution_vector=best_x,
            solution_score=best_val,
            solution_history=np.array(best_soln_history),
            stop_reason=stopped_early,
            generations_completed=generations_completed + 1,
            total_constraint_violation=None if total is None else float(total),
            ineq_relative_violations=None if ineq_rel is None else ineq_rel,
            eq_relative_violations=None if eq_rel is None else eq_rel,
            ineq_values=None if ineq_vals is None else ineq_vals,
            eq_values=None if eq_vals is None else eq_vals,
            unconstrained_best_score=ub_val,
            unconstrained_best_vector=ub_x,
        )
