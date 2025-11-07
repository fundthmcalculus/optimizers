import os
import time

import numpy as np
from sklearn.metrics import pairwise_distances

from optimizers.combinatorial.aco import AntColonyTSPConfig, AntColonyTSP
from optimizers.combinatorial.ga import GeneticAlgorithmTSP, GeneticAlgorithmTSPConfig
from optimizers.combinatorial.strategy import (
    ConvexHullTSPConfig,
    ConvexHullTSP,
    TwoOptTSP,
    TwoOptTSPConfig,
    NearestNeighborTSP,
    NearestNeighborTSPConfig,
)
from optimizers.core.types import AF
from optimizers.plot import plot_cities_and_route, plot_convergence


def main():
    city_locations = project_2_data()
    results = compute_tsp_bounds(city_locations)
    trace_names = [
        "Nearest Neighbor",
        "2-OPT",
        "Convex Hull",
        "Genetic Algorithm",
        "Ant Colony",
    ]
    plot_convergence([x.value_history for x in results], trace_names)
    plot_cities_and_route(
        city_locations,
        [x.optimal_path for x in results],
        trace_names,
    )


def compute_tsp_bounds(cities: AF):
    # Compute upper bound using Nearest Neighbor
    distances: AF = pairwise_distances(cities)

    # Time Nearest Neighbor
    start_time = time.time()
    nn_config = NearestNeighborTSPConfig(name="NN TSP", back_to_start=True)
    nn_optimizer = NearestNeighborTSP(nn_config, distances)
    nn_result = nn_optimizer.solve()
    nn_time = time.time() - start_time

    # Time 2-OPT
    start_time = time.time()
    topt_config = TwoOptTSPConfig(
        name="2-OPT TSP", back_to_start=True, num_iterations=cities.shape[0]
    )
    topt_optimizer = TwoOptTSP(
        topt_config,
        initial_route=nn_result.optimal_path,
        initial_value=nn_result.optimal_value,
        network_routes=distances,
    )
    topt_result = topt_optimizer.solve()
    topt_time = time.time() - start_time

    # Time Convex Hull
    start_time = time.time()
    ch_config = ConvexHullTSPConfig(name="CH TSP", back_to_start=True)
    ch_optimizer = ConvexHullTSP(ch_config, city_locations=cities)
    ch_result = ch_optimizer.solve()
    ch_time = time.time() - start_time

    n_generations = 50
    n_ants = 50
    solution_archive_size = 100

    # Time Genetic Algorithm
    start_time = time.time()
    ga_config = GeneticAlgorithmTSPConfig(
        name="GA TSP",
        num_generations=n_generations
        * 3,  # NOTE - Anecdotally, GA runs about 3x faster than ACO, but worse.
        population_size=n_ants,
        solution_archive_size=solution_archive_size,
        joblib_prefer="threads",
        stop_after_iterations=n_generations * 3,  # No early stopping!
    )
    ga_optimizer = GeneticAlgorithmTSP(
        ga_config, network_routes=distances, city_locations=cities
    )
    ga_result = ga_optimizer.solve()
    ga_time = time.time() - start_time

    # Time Ant Colony Optimization
    start_time = time.time()
    aco_config = AntColonyTSPConfig(
        name="ACO TSP",
        num_generations=n_generations,
        solution_archive_size=solution_archive_size,
        population_size=n_ants,
        joblib_prefer="threads",
        stop_after_iterations=n_generations,  # No early stopping!
    )
    aco_optimizer = AntColonyTSP(
        aco_config, network_routes=distances, city_locations=cities
    )
    aco_result = aco_optimizer.solve()
    aco_time = time.time() - start_time

    print("\n")
    print(
        f"TSP Upper Bound (Nearest Neighbor): {nn_result.optimal_value:.2f} (Time: {nn_time:.2f}s)"
    )
    print(
        f"TSP 2-OPT Solution: {topt_result.optimal_value:.2f} (Time: {topt_time:.2f}s)"
    )
    print(
        f"TSP Lower Bound (Convex Hull): {ch_result.optimal_value:.2f} (Time: {ch_time:.2f}s)"
    )
    print(f"TSP GA Solution: {ga_result.optimal_value:.2f} (Time: {ga_time:.2f}s)")
    print(f"TSP ACO Solution: {aco_result.optimal_value:.2f} (Time: {aco_time:.2f}s)")

    return nn_result, topt_result, ch_result, ga_result, aco_result


def project_2_data() -> AF:
    """Load city coordinates from city_data.txt and validate shape.

    Returns:
        AF: Array of city coordinates with shape (50,2)
    """
    city_data = np.loadtxt(
        os.path.join(os.path.dirname(__file__), "city_data.txt"),
        delimiter=",",
        dtype=np.float64,
    )
    assert city_data.shape == (
        50,
        2,
    ), f"Expected shape (50,2) but got {city_data.shape}"
    return city_data


if __name__ == "__main__":
    main()
