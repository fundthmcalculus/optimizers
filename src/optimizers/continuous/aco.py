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
    cp_j: af64 | None = None,
) -> OptimizerRun:
    # The rank CDF depends only on q and the (bounded) archive length, so it is
    # computed once per generation in solve() and passed in; recompute only as a
    # fallback. See report item #5.
    if cp_j is None:
        cp_j = cdf(q_weight, len(solution_archive))
    n_vars = len(variables)
    rng = global_rng()

    # Each ant picks a single base solution from the archive via the rank CDF
    # (the original drew one p per ant and reused it across every variable).
    base_idx = np.searchsorted(cp_j, rng.uniform(size=n_ants))
    base_idx = np.clip(base_idx, 0, solution_archive.shape[0] - 1)
    base_rows = solution_archive[base_idx]  # (n_ants, n_vars)

    # Sample each variable across ALL ants at once. n_vars is small; n_ants is
    # large, so this turns the hot inner loop into a handful of array ops.
    ant_solutions = np.empty((n_ants, n_vars))
    for i, variable in enumerate(variables):
        ant_solutions[:, i] = variable.random_values(
            current_values=base_rows[:, i],
            other_values=solution_archive[:, i],
            learning_rate=learning_rate,
            rng=rng,
        )

    # Evaluation (and optional local search) stays per-ant: the goal function is
    # a user-supplied scalar and cannot be assumed vectorizable.
    ant_values = np.empty(n_ants)
    for ant in range(n_ants):
        new_solution, new_value = apply_local_optimization(
            fcn, local_optim, ant_solutions[ant], variables
        )
        ant_solutions[ant] = new_solution
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
    ):
        super().__init__(
            config,
            fcn,
            variables,
            args,
            existing_soln_deck,
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

            # Compute the rank CDF once per generation (not once per worker).
            cp_j = cdf(self.config.q, len(self.soln_deck.solution_archive))
            job_output: list[OptimizerRun] = parallel(
                joblib.delayed(run_ants)(
                    individuals_per_job,
                    self.config.q,
                    self.config.learning_rate,
                    self.config.local_grad_optim,
                    self.soln_deck.solution_archive,
                    self.variables,
                    self.wrapped_fcn,
                    cp_j,
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
