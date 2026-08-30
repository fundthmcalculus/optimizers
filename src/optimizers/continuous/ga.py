from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator

from .local import apply_local_optimization
from ..core.types import AF
from ..core.base import (
    OptimizerResult,
    IOptimizerConfig,
    OptimizerRun,
    GoalFcn,
    InputArguments,
    BatchGoalFcn,
)
from .base import (
    check_stop_early,
    sync_worker_meta,
)
from ..core import InputVariables
from ..core.random import rng as global_rng
from ..core.parallel import GenerationRunner
from ..archive.variation import iso_line_offspring
from .base import IOptimizer

from ..solution_deck import SolutionDeck


@dataclass
class GeneticAlgorithmOptimizerConfig(IOptimizerConfig):
    mutation_rate: float = 0.1
    """Probability of mutation"""
    crossover_rate: float = 0.8
    """Probability of crossover"""


def _tournament_selection_batch(
    population_deck: AF,
    population_fitness: AF,
    n: int,
    tournament_size: int = 3,
    rng: Generator | None = None,
) -> AF:
    # Select ``n`` winners at once. Each winner is the best of ``tournament_size``
    # random rows, drawn directly via ``rng.integers`` -- O(n*k) instead of the
    # previous O(n*deck_len) (first an O(n*deck_len log deck_len) argsort, then
    # an O(n*deck_len) argpartition; see git history for both). At a large
    # archive this previously dominated GA's wall-clock end to end (measured:
    # population 8000/archive 24000 took 251s, ~100x slower than ACO/PSO on
    # the same workload -- see docs/history/PERF_CONTINUOUS_REPORT.md §6b).
    #
    # This does not enforce distinctness within one tournament (unlike the
    # argsort/argpartition versions before it), so it is not bit-identical or
    # even guaranteed-equivalent to them -- it draws different random numbers
    # and can (rarely) pick the same row twice in one tournament. The
    # probability of any repeat among k draws is ~k*(k-1)/(2*deck_len) (e.g.
    # ~0.05% for k=3, deck_len=3000) and *shrinks* as the archive grows -- the
    # opposite of the old approach's cost, which *grew* with the archive.
    # Statistically inconsequential for a stochastic search; accepted
    # deliberately in exchange for a complexity-class fix (see
    # docs/history/PERF_CONTINUOUS_REPORT.md §6b for the discussion this implements).
    if rng is None:
        rng = global_rng()
    deck_len = len(population_deck)
    k = min(tournament_size, deck_len)
    candidates = rng.integers(0, deck_len, size=(n, k))  # (n, k)
    candidate_fitness = population_fitness[candidates]  # (n, k)
    winners = candidates[np.arange(n), np.argmin(candidate_fitness, axis=1)]
    return population_deck[winners]  # (n, n_vars)


def _crossover_batch(
    parents1: AF,
    parents2: AF,
    crossover_rate: float,
    rng: Generator | None = None,
) -> tuple[AF, AF]:
    # Single-point crossover for every pair at once. Rows where crossover does
    # not fire pass the parents through unchanged (matching the scalar version).
    if rng is None:
        rng = global_rng()
    n: int = parents1.shape[0]
    n_vars: int = parents1.shape[1]
    do_cross = rng.random(n) < crossover_rate
    point = rng.integers(0, n_vars, size=n)  # crossover index in [0, n_vars)
    cols = np.arange(n_vars)[None, :]
    swap = do_cross[:, None] & (cols >= point[:, None])  # (n, n_vars)
    child1 = np.where(swap, parents2, parents1)
    child2 = np.where(swap, parents1, parents2)
    return child1, child2


def _mutate_batch(
    children: AF,
    mutation_rate: float,
    variables: InputVariables,
    rng: Generator | None = None,
) -> AF:
    # Mutate a whole batch of children. For each variable (few), decide which
    # rows mutate and perturb those entries with one vectorized call.
    if rng is None:
        rng = global_rng()
    out = np.copy(children)
    n: int = out.shape[0]
    mask = rng.random((n, len(variables))) < mutation_rate
    for ij, variable in enumerate(variables):
        col_mask = mask[:, ij]
        if col_mask.any():
            perturbed = variable.perturb_values(out[:, ij], rng=rng)
            out[col_mask, ij] = perturbed[col_mask]
    return out


def run_ga(
    fixed: tuple[Any, ...],
    meta: InputArguments,
    solution_values: AF,
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
        qd,
        batch_fcn,
    ) = fixed
    map_elites, variation, iso_sigma, line_sigma, lower, upper = qd
    sync_worker_meta(arg_provider, meta)
    rng = global_rng()

    if map_elites and variation == "iso_line":
        # Shared Iso+LineDD variation over the diverse CVT archive (identical
        # across GA/ACO/PSO for fair comparison). See docs/history/QD_PARETO_PLAN.md §4.3.
        children = iso_line_offspring(
            solution_archive, n_steps, iso_sigma, line_sigma, lower, upper, rng
        )
        new_population = np.empty((n_steps, len(variables)))
        new_population_fitness = np.empty(n_steps)
        for row in range(n_steps):
            c, f = apply_local_optimization(fcn, local_optim, children[row], variables)
            new_population[row, :] = c
            new_population_fitness[row] = f
        return OptimizerRun(
            population_solutions=new_population,
            population_values=new_population_fitness,
            eval_count=arg_provider.eval_delta,
        )

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

    if local_optim == "none" and batch_fcn is not None:
        # No local search to interleave, and the caller supplied a batched goal
        # function: score both whole children batches in two vectorized calls
        # instead of ``2 * n_steps`` scalar ones. See docs/history/PERF_CONTINUOUS_REPORT.md.
        f1 = batch_fcn(child1)
        f2 = batch_fcn(child2)
        pick1 = f1 < f2  # ties go to child2, matching the scalar loop below
        new_population = np.where(pick1[:, None], child1, child2)
        new_population_fitness = np.where(pick1, f1, f2)
    else:
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
        population_solutions=new_population,
        population_values=new_population_fitness,
        eval_count=arg_provider.eval_delta,
    )


class GeneticAlgorithmOptimizer(IOptimizer):
    def __init__(
        self,
        *,
        config: IOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: InputArguments | None = None,
        existing_soln_deck: SolutionDeck | None = None,
        batch_fcn: BatchGoalFcn | None = None,
    ):
        super().__init__(
            config=config,
            fcn=fcn,
            variables=variables,
            args=args,
            existing_soln_deck=existing_soln_deck,
            batch_fcn=batch_fcn,
        )
        self.config: GeneticAlgorithmOptimizerConfig = GeneticAlgorithmOptimizerConfig(
            **{**config.__dict__}
        )

    def solve(self, *, preserve_percent: float = 0.0) -> OptimizerResult:
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
        # worker once; only the archive + fitness vary per generation. The ``qd``
        # payload carries the MAP-Elites variation settings (QD add-on, Phase 2).
        qd = (
            self._objective_mode == "map-elites",
            self.config.qd_variation,
            self.config.iso_sigma,
            self.config.line_sigma,
            np.array([v.lower_bound for v in self.variables], dtype=float),
            np.array([v.upper_bound for v in self.variables], dtype=float),
        )
        fixed = (
            self._arg_provider,
            self.variables,
            self.wrapped_fcn,
            self.config.mutation_rate,
            self.config.crossover_rate,
            self.config.local_grad_optim,
            individuals_per_job,
            qd,
            self.wrapped_batch_fcn,
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
