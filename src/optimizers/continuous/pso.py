from dataclasses import dataclass

import joblib
import numpy as np

from .local import apply_local_optimization
from optimizers.core.base import (
    IOptimizerConfig,
    OptimizerResult,
    OptimizerRun,
)
from .base import (
    IOptimizer,
    setup_for_generations,
    check_stop_early,
    cdf,
)
from .variables import InputDiscreteVariable
from ..core.types import af64
from optimizers.solution_deck import (
    WrappedGoalFcn,
    GoalFcn,
    LocalOptimType,
    InputArguments,
    InputVariables,
    SolutionDeck,
)


@dataclass
class ParticleSwarmOptimizerConfig(IOptimizerConfig):
    inertia: float = 0.5
    """Inertia weight (w) for velocity component"""
    cognitive: float = 1.5
    """Cognitive coefficient (c1) towards personal best"""
    social: float = 1.5
    """Social coefficient (c2) towards global best"""
    velocity_clamp: float = 0.5
    """Max velocity as a fraction of variable range"""
    q: float = 1.0
    """Weighting parameter for selecting better ranked solutions as p-best"""
    local_grad_optim: LocalOptimType = "none"


def run_particles(
    n_particles: int,
    inertia: float,
    cognitive: float,
    social: float,
    velocity_clamp: float,
    q_weight: float,
    local_optim: LocalOptimType,
    solution_archive: af64,
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
    swarm_best_pos = np.zeros(len(variables))
    swarm_best_val = 0.0
    for k in range(n_particles):
        for d, v in enumerate(variables):
            p_pos[k, d] = p_best_pos[k, d] = v.initial_random_value()
            p_vel[k, d] = v.initial_random_velocity()
        p_best_val[k] = fcn(p_best_pos[k, :])
    # Get the best position
    best_idx = np.argmin(p_best_val)
    swarm_best_pos = p_best_pos[best_idx, :]
    swarm_best_val = p_best_val[best_idx]

    # Run for a certain number of iterations, or maybe till the solution doesn't get much better?
    w = 0.5  # Inertial weight
    phi_p = 1.5  # Typically between [1,3]
    phi_g = 1.5  # Typically between [1,3]
    n_iterations = 10
    for cur_iter in range(n_iterations):
        # TODO - Vectorize this!
        for k in range(n_particles):
            for d, v in enumerate(variables):
                r_p, r_g = np.random.rand(), np.random.rand()
                # Update velocity
                p_vel[k, d] = (
                    w * p_vel[k, d]
                    + r_p * phi_p * (p_best_pos[k, d] - p_pos[k, d])
                    + phi_g * r_g * (swarm_best_pos[d] - p_pos[k, d])
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
            if stopped_early:
                break

            job_output: list[OptimizerRun] = parallel(
                joblib.delayed(run_particles)(
                    individuals_per_job,
                    self.config.inertia,
                    self.config.cognitive,
                    self.config.social,
                    self.config.velocity_clamp,
                    self.config.q,
                    self.config.local_grad_optim,
                    self.soln_deck.solution_archive,
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
