from dataclasses import dataclass

import joblib
import numpy as np

from .local import apply_local_optimization
from optimizers.core.base import (
    IOptimizerConfig,
    OptimizerResult,
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
) -> tuple[af64, af64]:
    """
    Generate new candidate solutions using a PSO-inspired step that leverages the
    current solution archive as memory for personal bests and global best.

    This implementation is stateless across generations (to align with the ACO/GA
    parallel pattern), but still moves candidates towards historically good
    solutions (p-best from archive, g-best as current best) with velocity clamping.
    """
    # Pre-compute selection CDF (rank-biased) and global best
    cp_j = cdf(q_weight, len(solution_archive))
    gbest = solution_archive[0, :]

    dim = len(variables)
    new_solutions = np.zeros((n_particles, dim))
    new_values = np.zeros(n_particles)

    rng = np.random.default_rng()

    # Precompute per-dimension velocity clamps based on variable ranges
    var_ranges = np.array([var.upper_bound - var.lower_bound for var in variables])
    v_max = np.maximum(1e-12, velocity_clamp * var_ranges)

    for k in range(n_particles):
        # Choose a base current position x and a personal best pbest from the archive (rank-biased)
        p = rng.uniform()
        base_idx = int(np.searchsorted(cp_j, p))
        x = solution_archive[base_idx, :]

        p2 = rng.uniform()
        pbest_idx = int(np.searchsorted(cp_j, p2))
        pbest = solution_archive[pbest_idx, :]

        # Ensure pbest has better fitness than x
        if p2 < x:  # Assuming minimization
            x, pbest = pbest, x

        # Random factors
        r1 = rng.uniform(size=dim)
        r2 = rng.uniform(size=dim)

        # Velocity (stateless approximation): move towards pbest and gbest
        v = (
            inertia * (pbest - x)
            + cognitive * r1 * (pbest - x)
            + social * r2 * (gbest - x)
        )

        # Clamp velocity to avoid overshoot
        v = np.clip(v, -v_max, v_max)

        # Update position and clip to bounds
        x_new = x + v
        for i, var in enumerate(variables):
            lb = var.lower_bound
            ub = var.upper_bound
            if x_new[i] < lb:
                x_new[i] = lb
            elif x_new[i] > ub:
                x_new[i] = ub

            # Handle discrete variables by rounding to the nearest valid option
            if isinstance(var, InputDiscreteVariable):
                x_new[i] = var.get_nearest_value(x_new[i])

        # Optional local refinement and evaluation
        x_new, f_new = apply_local_optimization(fcn, local_optim, x_new, variables)

        new_solutions[k, :] = x_new
        new_values[k] = f_new

    return new_solutions, new_values


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
        best_soln_history, generation_pbar, generations_completed, individuals_per_job, n_jobs, parallel, stopped_early = self.initialize(
            preserve_percent)
        for generations_completed in generation_pbar:
            stopped_early = check_stop_early(
                self.config, best_soln_history, self.soln_deck.solution_value
            )
            if stopped_early:
                break

            job_output = parallel(
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
            for output in job_output:
                output_solutions = output[0]
                output_values = output[1]
                self.soln_deck.append(
                    output_solutions,
                    output_values,
                    self.config.local_grad_optim != "none",
                )
                self.soln_deck.deduplicate()
            generation_pbar.set_postfix(best_value=self.soln_deck.solution_value[0])

        # Return the best solution
        return OptimizerResult(
            solution_vector=self.soln_deck.solution_archive[0, :],
            solution_score=self.soln_deck.solution_value[0],
            solution_history=best_soln_history,
            stopped_early=stopped_early,
            generations_completed=generations_completed + 1,
        )
