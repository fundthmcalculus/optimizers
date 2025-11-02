from dataclasses import dataclass
from typing import Optional

import numpy as np
import tqdm
from joblib import Parallel, delayed, cpu_count

from .strategy import TwoOptTSPConfig, TwoOptTSP
from ..core import IOptimizerConfig
from .base import TSPBase, CombinatoricsResult, _check_stop_early, check_path_distance
from ..core.base import setup_for_generations
from ..core.types import AF, AI, F


@dataclass
class GeneticAlgorithmTSPConfig(IOptimizerConfig):

    mutation_rate: float = 0.1
    """Probability of mutation"""
    crossover_rate: float = 0.8
    """Probability of crossover"""
    back_to_start: bool = True
    """Whether to return to the start node"""
    local_optimize: bool = False
    """Local optimization using 2OPT method"""
    hot_start: Optional[list[int]] = None
    """Hot start solution"""
    hot_start_length: Optional[float] = None
    """Hot start length"""


class GeneticAlgorithmTSP(TSPBase):
    def __init__(
        self,
        config: GeneticAlgorithmTSPConfig,
        network_routes: Optional[AF] = None,
        city_locations: Optional[AF] = None,
    ):
        super().__init__(network_routes, city_locations)
        self.config = config

    def solve(self) -> CombinatoricsResult:
        self.network_routes[self.network_routes == 0] = -1
        # Allocate the genome candidates
        genome = np.zeros(
            (self.config.solution_archive_size, self.network_routes.shape[1]),
            dtype=np.int32,
        )
        genome_value = np.zeros(self.config.solution_archive_size)
        # Create a bunch of random entries in the genome, each row is a permutation of [0, N), but 0 is always first
        permute_city = np.r_[1 : self.network_routes.shape[1]]
        for i in range(self.config.solution_archive_size):
            genome[i, 1:] = np.random.permutation(permute_city)
            genome_value[i] = check_path_distance(
                self.network_routes, genome[i], self.config.back_to_start
            )

        # Sort by value
        genome = genome[np.argsort(genome_value)]
        genome_value = np.sort(genome_value)

        if self.config.hot_start is not None:
            # Replace the worst entry with the best
            genome[-1, :] = self.config.hot_start
            # Resort
            genome = genome[np.argsort(genome_value)]
            genome_value = np.sort(genome_value)

        tour_lengths = []

        generation_pbar, individuals_per_job, n_jobs, parallel = setup_for_generations(
            self.config
        )

        with parallel:
            for generations_completed in generation_pbar:

                def parallel_ga(local_ant):
                    for _ in range(individuals_per_job):
                        yield run_ga(
                            genome, genome_value, self.network_routes, self.config
                        )

                all_results = parallel(
                    delayed(parallel_ga)(i_ant) for i_ant in range(n_jobs)
                )

                for ant, result_gen in enumerate(all_results):
                    for city_order, tour_length in result_gen:
                        # If a dead-end, skip!
                        if tour_length == np.inf:
                            continue
                        # Append to the genome
                        genome = np.vstack((genome, city_order))
                        genome_value = np.hstack((genome_value, tour_length))

                # Sort by genome and keep only the required rows
                genome = genome[np.argsort(genome_value)]
                genome_value = np.sort(genome_value)
                genome = genome[: self.config.solution_archive_size, :]
                genome_value = genome_value[: self.config.solution_archive_size]

                tour_lengths.append(genome_value[0])
                # Check for stopping early
                stop_reason = _check_stop_early(self.config, tour_lengths)
                if stop_reason != "none":
                    break

        if self.config.local_optimize:
            # TODO - Better parameters?
            two_opt_config = TwoOptTSPConfig()
            two_opt_optimize = TwoOptTSP(
                two_opt_config,
                initial_route=genome[0, :],
                initial_value=genome_value[0],
                network_routes=self.network_routes,
                city_locations=self.city_locations,
            )
            result = two_opt_optimize.solve()
            tour_lengths.append(result.optimal_value)

            return CombinatoricsResult(
                optimal_path=result.optimal_path,
                optimal_value=result.optimal_value,
                value_history=np.array(tour_lengths),
                stop_reason="max_iterations" if stop_reason == "none" else stop_reason,
            )
        else:
            return CombinatoricsResult(
                optimal_path=genome[0, :],
                optimal_value=genome_value[0],
                value_history=np.array(tour_lengths),
                stop_reason="max_iterations" if stop_reason == "none" else stop_reason,
            )


def run_ga(
    genome: AI, genome_value: AF, network_routes: AF, config: GeneticAlgorithmTSPConfig
) -> tuple[AI, F]:
    # Take two parents
    parent_1 = _tournament_selection(genome, genome_value)
    parent_2 = _tournament_selection(genome, genome_value)
    # Perform genetic operations.
    child_1, child_2 = _crossover(parent_1, parent_2, config.crossover_rate)
    child_1 = _mutate(child_1, config.mutation_rate, network_routes)
    child_2 = _mutate(child_2, config.mutation_rate, network_routes)
    # Do a bit of 2-OPT fine-tuning so everything gets a tiny bit better
    child_1 = _2opt_refine(child_1, network_routes)
    child_2 = _2opt_refine(child_2, network_routes)
    child_1_fitness = check_path_distance(network_routes, child_1, config.back_to_start)
    child_2_fitness = check_path_distance(network_routes, child_2, config.back_to_start)
    if child_1_fitness < child_2_fitness:
        return child_1, child_1_fitness
    else:
        return child_2, child_2_fitness


def _2opt_refine(new_route: AI, network_routes: AF, nearest_neighbors=10) -> AI:
    N = len(new_route)
    ij = np.random.randint(low=1, high=max(1, N - nearest_neighbors))
    k_nn = N
    if nearest_neighbors > 0:
        k_nn = min(k_nn, ij + nearest_neighbors)
    for jk in range(ij + 2, k_nn):
        d1 = (
            network_routes[new_route[ij], new_route[ij + 1]]
            + network_routes[new_route[jk], new_route[jk + 1]]
        )
        d2 = (
            network_routes[new_route[ij], new_route[jk]]
            + network_routes[new_route[ij + 1], new_route[jk + 1]]
        )
        if d1 > d2:
            new_route[jk], new_route[ij + 1] = (
                new_route[ij + 1],
                new_route[jk],
            )
    return new_route


def _mutate(child: AI, mutation_rate: F, network_routes: AF) -> AI:
    if np.random.rand() < mutation_rate:
        # Swap a percentage of the variables
        n_swaps = max(1, int(np.round(mutation_rate * len(child) / 2.0)))
        candidate_swaps = np.r_[1 : len(child) - 1]
        for _ in range(n_swaps):
            ij, jk = np.random.choice(candidate_swaps, 2, replace=False)
            # Ensure that this swap is actually better.
            d1 = (
                network_routes[child[ij], child[ij + 1]]
                + network_routes[child[jk], child[jk + 1]]
            )
            d2 = (
                network_routes[child[ij], child[jk]]
                + network_routes[child[jk + 1], child[ij + 1]]
            )
            if d1 > d2:
                child[ij], child[jk] = child[jk], child[ij]
        return child
    else:
        return child


def _tournament_selection(
    population_deck: AF | AI,
    population_fitness: AF,
    tournament_size: int = 3,
) -> AF | AI:
    # Randomly sample row
    row_idxs = np.random.choice(
        population_deck.shape[0], size=tournament_size, replace=False
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
        # Preallocate a full length sequence to ensure each child has ALL entries
        all_entries = np.random.permutation(parent1.shape[0])
        child1 = np.concatenate(
            [parent1[:crossover_idx], parent2[crossover_idx:], all_entries], axis=0
        )
        child2 = np.concatenate(
            [parent2[:crossover_idx], parent1[crossover_idx:], all_entries], axis=0
        )
        # Deduplicate entries
        child1 = np.unique(child1, sorted=False)
        child2 = np.unique(child2, sorted=False)
        # Make sure 0 is always first
        child1 = np.concatenate([np.array([0]), child1[child1 > 0]], axis=0)
        child2 = np.concatenate([np.array([0]), child2[child2 > 0]], axis=0)
        return child1, child2
    else:
        return parent1, parent2
