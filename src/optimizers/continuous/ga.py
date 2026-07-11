from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from .local import apply_local_optimization
from ..core.types import AF, AI
from ..core.base import (
    OptimizerResult,
    IOptimizerConfig,
    OptimizerRun,
    GoalFcn,
    InputArguments,
)
from .base import (
    check_stop_early,
    sync_worker_meta,
)
from ..core.variables import InputVariables
from ..core.random import rng as global_rng
from ..core.parallel import GenerationRunner
from .base import IOptimizer

from ..solution_deck import (
    SolutionDeck,
)


@dataclass
class GeneticAlgorithmOptimizerConfig(IOptimizerConfig):
    mutation_rate: float = 0.1
    """Probability of mutation"""
    crossover_rate: float = 0.8
    """Probability of crossover"""


def _tournament_selection_batch(
    population_deck: AF | AI,
    population_fitness: AF,
    n: int,
    tournament_size: int = 3,
    rng: Generator | None = None,
) -> AF | AI:
    # Select ``n`` winners at once. Each winner is the best of ``tournament_size``
    # distinct random rows; distinctness comes from argsort-of-random-keys, which
    # vectorizes the per-selection np.random.choice(replace=False). See report #5.
    if rng is None:
        rng = global_rng()
    deck_len = len(population_deck)
    k = min(tournament_size, deck_len)
    candidates = np.argsort(rng.random((n, deck_len)), axis=1)[:, :k]  # (n, k)
    candidate_fitness = population_fitness[candidates]  # (n, k)
    winners = candidates[np.arange(n), np.argmin(candidate_fitness, axis=1)]
    return population_deck[winners]  # (n, n_vars)


def _crossover_batch(
    parents1: AF | AI,
    parents2: AF | AI,
    crossover_rate: float,
    rng: Generator | None = None,
) -> tuple[AF | AI, AF | AI]:
    # Single-point crossover for every pair at once. Rows where crossover does
    # not fire pass the parents through unchanged (matching the scalar version).
    if rng is None:
        rng = global_rng()
    n, n_vars = parents1.shape
    do_cross = rng.random(n) < crossover_rate
    point = rng.integers(0, n_vars, size=n)  # crossover index in [0, n_vars)
    cols = np.arange(n_vars)[None, :]
    swap = do_cross[:, None] & (cols >= point[:, None])  # (n, n_vars)
    child1 = np.where(swap, parents2, parents1)
    child2 = np.where(swap, parents1, parents2)
    return child1, child2


def _mutate_batch(
    children: AF | AI,
    mutation_rate: float,
    variables: InputVariables,
    rng: Generator | None = None,
) -> AF | AI:
    # Mutate a whole batch of children. For each variable (few), decide which
    # rows mutate and perturb those entries with one vectorized call.
    if rng is None:
        rng = global_rng()
    out = np.copy(children)
    n = out.shape[0]
    mask = rng.random((n, len(variables))) < mutation_rate
    for ij, variable in enumerate(variables):
        col_mask = mask[:, ij]
        if col_mask.any():
            perturbed = variable.perturb_values(out[:, ij], rng=rng)
            out[col_mask, ij] = perturbed[col_mask]
    return out


def run_ga(
    fixed: tuple,
    meta: InputArguments,
    solution_values: AF | AI,
    solution_archive: AF,
) -> OptimizerRun:
    # ``fixed`` is shipped to each worker once; ``meta`` is the small per-
    # generation live metadata. See core.parallel.
    (
        arg_provider,
        variables,
        fcn,
        mutation_rate,
        crossover_rate,
        local_optim,
        n_steps,
    ) = fixed
    sync_worker_meta(arg_provider, meta)
    rng = global_rng()
    # Vectorize the genetic operators across the whole batch of offspring; only
    # evaluation / local search stays per-individual (scalar goal function).
    parents1 = _tournament_selection_batch(
        solution_archive, solution_values, n_steps, rng=rng
    )
    parents2 = _tournament_selection_batch(
        solution_archive, solution_values, n_steps, rng=rng
    )
    child1, child2 = _crossover_batch(parents1, parents2, crossover_rate, rng=rng)
    child1 = _mutate_batch(child1, mutation_rate, variables, rng=rng)
    child2 = _mutate_batch(child2, mutation_rate, variables, rng=rng)

    new_population = np.empty((n_steps, len(variables)))
    new_population_fitness = np.empty(n_steps)
    for row in range(n_steps):
        # Optimize child-1, because firstborn rights.
        c1, f1 = apply_local_optimization(fcn, local_optim, child1[row], variables)
        c2, f2 = apply_local_optimization(fcn, local_optim, child2[row], variables)
        if f1 < f2:
            new_population[row, :] = c1
            new_population_fitness[row] = f1
        else:
            new_population[row, :] = c2
            new_population_fitness[row] = f2
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
        # Ship fixed data (variables, goal fn, GA hyper-parameters) to each
        # worker once; only the archive + fitness vary per generation.
        fixed = (
            self._arg_provider,
            self.variables,
            self.wrapped_fcn,
            self.config.mutation_rate,
            self.config.crossover_rate,
            self.config.local_grad_optim,
            individuals_per_job,
        )
        runner = GenerationRunner(n_jobs, self.config.joblib_prefer, fixed)
        try:
            for generations_completed in generation_pbar:
                # Update runtime metadata for this generation
                self._set_phase("evolve")
                self._set_generation(generations_completed)

                stopped_early = check_stop_early(
                    self.config, best_soln_history, self.soln_deck.solution_value
                )
                if stopped_early != "none":
                    break

                job_output: list[OptimizerRun] = runner.run(
                    run_ga,
                    (
                        self.live_meta(),
                        self.soln_deck.solution_value,
                        self.soln_deck.solution_archive,
                    ),
                )

                # Merge candidates into the archive
                self.update_solution_deck(generation_pbar, job_output)
                best_soln_history.append(self.soln_deck.get_best()[1])
        finally:
            runner.close()

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
