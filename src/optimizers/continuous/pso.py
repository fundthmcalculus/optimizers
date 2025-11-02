from dataclasses import dataclass

import joblib
import numpy as np

from ..core.base import (
    IOptimizerConfig,
    OptimizerResult,
    OptimizerRun,
    LocalOptimType,
)
from ..core.types import AF, F
from ..solution_deck import (
    WrappedGoalFcn,
    GoalFcn,
    InputArguments,
    InputVariables,
    SolutionDeck,
)
from .base import (
    IOptimizer,
    check_stop_early,
)
from ..core.types import af64


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
    n_particles: int,
    inertia: float,
    cognitive: float,
    social: float,
    velocity_clamp: float,
    global_best_position: AF,
    global_best_value: F,
    variables: InputVariables,
    fcn: WrappedGoalFcn,
) -> OptimizerRun:
    """
    Generate new candidate solutions using a PSO-inspired step that leverages the
    current solution archive as memory for personal bests and global best.

    This implementation is stateless across generations (to align with the ACO/GA
    parallel pattern), but still moves candidates towards historically good
    solutions (p-best from archive, g-best as current best) with velocity clamping.
    """
    # Lifted from: https://en.wikipedia.org/wiki/Particle_swarm_optimization#Algorithm
    # Precompute the initial vector for each particle
    p_best_pos = np.zeros((n_particles, len(variables)))  # Best Position
    p_pos = np.zeros((n_particles, len(variables)))  # Current Position
    p_vel = np.zeros((n_particles, len(variables)))  # Current Velocity
    p_best_val = np.zeros(n_particles)  # Particle best value
    # Swarm best position
    for k in range(n_particles):
        for d, v in enumerate(variables):
            p_pos[k, d] = p_best_pos[k, d] = v.initial_random_value()
            p_vel[k, d] = v.initial_random_velocity()
        p_best_val[k] = fcn(p_best_pos[k, :])
    # Get the best position
    best_idx = np.argmin(p_best_val)
    swarm_best_pos = p_best_pos[best_idx, :]
    swarm_best_val = p_best_val[best_idx]
    if global_best_value < swarm_best_val:
        swarm_best_val = global_best_value
        swarm_best_pos = global_best_position

    n_iterations = 10
    for cur_iter in range(n_iterations):
        # TODO - Vectorize this!
        for k in range(n_particles):
            for d, v in enumerate(variables):
                r_p, r_g = np.random.rand(), np.random.rand()
                # Update velocity
                p_vel[k, d] = (
                    inertia * p_vel[k, d]
                    + r_p * cognitive * (p_best_pos[k, d] - p_pos[k, d])
                    + social * r_g * (swarm_best_pos[d] - p_pos[k, d])
                )
                # Clamp the velocity
                p_vel[k, d] = min(
                    max(p_vel[k, d], -velocity_clamp), velocity_clamp
                )
        # Update the particle position for this time step.
        p_pos += p_vel
        # Do the best position for each particle
        for k in range(n_particles):
            new_val = fcn(p_pos[k, :])
            if new_val < p_best_val[k]:
                p_best_val[k] = new_val
                p_best_pos[k, :] = p_pos[k, :]
            if new_val < swarm_best_val:
                swarm_best_val = new_val
                swarm_best_pos = p_pos[k, :]
    # Return the positions and values
    return OptimizerRun(population_values=p_best_val, population_solutions=p_best_pos)


class ParticleSwarmOptimizer(IOptimizer):
    def __init__(
        self,
        config: IOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: InputArguments | None = None,
        existing_soln_deck: SolutionDeck | None = None,
    ):
        super().__init__(config, fcn, variables, args, existing_soln_deck)
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
        for generations_completed in generation_pbar:
            stopped_early = check_stop_early(
                self.config, best_soln_history, self.soln_deck.solution_value
            )
            if stopped_early != 'none':
                break

            job_output: list[OptimizerRun] = parallel(
                joblib.delayed(run_particles)(
                    individuals_per_job,
                    self.config.inertia,
                    self.config.cognitive,
                    self.config.social,
                    self.config.velocity_clamp,
                    self.soln_deck.solution_archive[0,:],
                    self.soln_deck.solution_value[0],
                    self.variables,
                    self.wrapped_fcn,
                )
                for _ in range(n_jobs)
            )

            # Merge candidates into the archive
            self.update_solution_deck(generation_pbar, job_output)

        stopped_early = stopped_early if stopped_early != "none" else "max_iterations"
        # Return the best solution
        return OptimizerResult(
            solution_vector=self.soln_deck.solution_archive[0, :],
            solution_score=self.soln_deck.solution_value[0],
            solution_history=best_soln_history,
            stop_reason=stopped_early,
            generations_completed=generations_completed + 1,
        )
