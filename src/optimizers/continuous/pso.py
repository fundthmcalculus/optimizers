from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.base import (
    IOptimizerConfig,
    OptimizerResult,
    OptimizerRun,
    GoalFcn,
    InputArguments,
)
from ..core.types import AF
from ..solution_deck import (
    InputVariables,
    SolutionDeck,
)
from .base import (
    IOptimizer,
    check_stop_early,
    sync_worker_meta,
)
from ..core.random import rng as global_rng
from ..core.parallel import GenerationRunner
from ..archive.variation import iso_line_offspring
from .local import apply_local_optimization


@dataclass
class ParticleSwarmOptimizerConfig(IOptimizerConfig):
    inertia: float = 0.5
    """Inertia weight (w) for velocity component"""
    cognitive: float = 1.5
    """Cognitive coefficient (c1) towards personal best"""
    social: float = 1.5
    """Social coefficient (c2) towards global best"""
    velocity_clamp: float = 0.5


def run_particles(
    fixed: tuple[Any, ...],
    meta: InputArguments,
    solution_archive: AF,
    solution_values: AF,
) -> OptimizerRun:
    """
    Generate new candidate solutions using a PSO-inspired step that leverages the
    current solution archive as memory for personal bests and global best.

    This implementation is stateless across generations (to align with the ACO/GA
    parallel pattern), but still moves candidates towards historically good
    solutions (p-best from archive, g-best as current best) with velocity clamping.
    """
    # Lifted from: https://en.wikipedia.org/wiki/Particle_swarm_optimization#Algorithm
    # ``fixed`` is shipped to each worker once; ``meta`` is the small per-
    # generation live metadata. See core.parallel.
    (
        arg_provider,
        variables,
        fcn,
        inertia,
        cognitive,
        social,
        velocity_clamp,
        n_particles,
        local_optim,
        qd,
    ) = fixed
    map_elites, variation, iso_sigma, line_sigma, lower, upper = qd
    sync_worker_meta(arg_provider, meta)
    rng = global_rng()
    n_vars = len(variables)

    if map_elites and variation == "iso_line":
        # Shared Iso+LineDD variation over the diverse CVT archive (same operator
        # as GA/ACO in this mode, for fair comparison). See QD_PARETO_PLAN.md §4.3.
        children = iso_line_offspring(
            solution_archive, n_particles, iso_sigma, line_sigma, lower, upper, rng
        )
        positions = np.empty((n_particles, n_vars))
        values = np.empty(n_particles)
        for k in range(n_particles):
            s, v = apply_local_optimization(fcn, local_optim, children[k], variables)
            positions[k] = s
            values[k] = v
        return OptimizerRun(
            population_values=values,
            population_solutions=positions,
            eval_count=arg_provider.eval_delta,
        )

    # Native PSO path: global best is the archive's top entry (best-first).
    global_best_position = solution_archive[0, :]
    global_best_value = solution_values[0]
    # Per-variable domains, used to clamp velocity (constant across the run).
    domains = np.array([v.domain for v in variables])

    # Vectorized initialization: fill each variable's column for all particles at
    # once (loop over the few variables, not the many particles). See report #5.
    p_best_pos = np.empty((n_particles, n_vars))  # Best Position
    p_vel = np.empty((n_particles, n_vars))  # Current Velocity
    for d, v in enumerate(variables):
        p_best_pos[:, d] = v.initial_random_values(n_particles, rng=rng)
        p_vel[:, d] = v.initial_random_velocities(n_particles, rng=rng)
    p_pos = p_best_pos.copy()  # Current Position starts at the initial best
    p_best_val = np.array([fcn(p_best_pos[k, :]) for k in range(n_particles)])

    # Get the best position
    best_idx = np.argmin(p_best_val)
    swarm_best_pos = p_best_pos[best_idx, :].copy()
    swarm_best_val = p_best_val[best_idx]
    if global_best_value < swarm_best_val:
        swarm_best_val = global_best_value
        swarm_best_pos = np.asarray(global_best_position).copy()

    n_iterations = 10
    for cur_iter in range(n_iterations):
        # One (r_p, r_g) per dimension (matching the original), applied across
        # all particles at once via broadcasting.
        r_p = rng.random(n_vars)
        r_g = rng.random(n_vars)
        p_vel = (
            inertia * p_vel
            + r_p * cognitive * (p_best_pos - p_pos)
            + social * r_g * (swarm_best_pos[None, :] - p_pos)
        )
        # Clamp the velocity (same per-dimension ratio clamp as before).
        p_vel *= np.minimum(
            np.maximum(p_vel / domains, -velocity_clamp), velocity_clamp
        )
        # Update the particle position for this time step.
        p_pos += p_vel
        # Evaluate all particles, then update personal/swarm bests vectorized.
        new_vals = np.array([fcn(p_pos[k, :]) for k in range(n_particles)])
        improved = new_vals < p_best_val
        p_best_val = np.where(improved, new_vals, p_best_val)
        p_best_pos[improved] = p_pos[improved]
        iter_best = int(np.argmin(new_vals))
        if new_vals[iter_best] < swarm_best_val:
            swarm_best_val = new_vals[iter_best]
            swarm_best_pos = p_pos[iter_best, :].copy()
    # Return the positions and values
    return OptimizerRun(
        population_values=p_best_val,
        population_solutions=p_best_pos,
        eval_count=arg_provider.eval_delta,
    )


class ParticleSwarmOptimizer(IOptimizer):
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
        self.config: ParticleSwarmOptimizerConfig = ParticleSwarmOptimizerConfig(
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
        # Ship fixed data (variables, goal fn, PSO coefficients) to each worker
        # once; only the current global best varies per generation.
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
            self.config.inertia,
            self.config.cognitive,
            self.config.social,
            self.config.velocity_clamp,
            individuals_per_job,
            self.config.local_grad_optim,
            qd,
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
                    run_particles,
                    (
                        self.live_meta(),
                        self.soln_deck.solution_archive,
                        self.soln_deck.solution_value,
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
