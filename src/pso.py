from dataclasses import dataclass

import joblib
import numpy as np

from local import apply_local_optimization
from optimizer_base import (
    IOptimizer,
    IOptimizerConfig,
    OptimizerResult,
    setup_for_generations,
    check_stop_early,
)
from variables import InputVariable, InputDiscreteVariable
from opt_types import af64
from solution_deck import GoalFcn, LocalOptimType, InputVariables, InputArguments


def cdf(q: float, N: int) -> af64:
    j = np.r_[1 : N + 1]
    c1 = 1 - np.exp(-q * j / N)
    return c1 / c1[-1]


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
    fcn: GoalFcn,
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
        x = np.array(solution_archive[base_idx, :], dtype=float)

        p2 = rng.uniform()
        pbest_idx = int(np.searchsorted(cp_j, p2))
        pbest = np.array(solution_archive[pbest_idx, :], dtype=float)

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
        name: str,
        config: ParticleSwarmOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: InputArguments | None = None,
    ):
        super().__init__(name, config, fcn, variables, args)
        self.config: ParticleSwarmOptimizerConfig = config

    def solve(self) -> OptimizerResult:
        self.validate_config(self.variables)
        self.soln_deck.initialize_solution_deck(self.variables, self.wrapped_fcn)
        self.soln_deck.sort()
        best_soln_history = np.zeros(self.config.num_generations)

        generation_pbar, individuals_per_job, n_jobs, parallel = setup_for_generations(
            self.config
        )
        for generation in generation_pbar:
            if check_stop_early(self.config, best_soln_history, self.soln_deck.solution_value):
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

            # Merge candidates into archive
            for output in job_output:
                part_solutions = output[0]
                part_values = output[1]
                self.soln_deck.append(part_solutions, part_values, self.config.local_grad_optim != "none")
                self.soln_deck.deduplicate()
            generation_pbar.set_postfix(best_value=self.soln_deck.solution_value[0])

        return OptimizerResult.from_solution_deck(self.soln_deck)

    def fill_solution_archive(
        self,
        fcn: GoalFcn,
        solution_archive: af64,
        solution_values: af64,
        variables: list[InputVariable],
        args: InputArguments,
    ) -> None:
        # Randomly initialize the archive and evaluate
        if args:
            wrapped_fcn = lambda x: fcn(x, args)
        else:
            wrapped_fcn = fcn
        for k in range(self.config.solution_archive_size):
            for i, variable in enumerate(variables):
                solution_archive[k, i] = variable.initial_random_value()
            solution_values[k] = wrapped_fcn(solution_archive[k])
        # Ensure the very first archive entry is the provided initial values
        for i, variable in enumerate(variables):
            solution_archive[0, i] = variable.initial_value
        solution_values[0] = wrapped_fcn(solution_archive[0])

    def create_solution_archive(
        self, variables: list[InputVariable]
    ) -> tuple[af64, af64]:
        solution_archive = np.zeros((self.config.solution_archive_size, len(variables)))
        solution_values = np.zeros(self.config.solution_archive_size)
        return solution_archive, solution_values

    def validate_config(self, variables: list[InputVariable]):
        if self.config.solution_archive_size < 0:
            self.config.solution_archive_size = len(variables) * 2
        if self.config.population_size < 0:
            self.config.population_size = max(1, self.config.solution_archive_size // 3)
