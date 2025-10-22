import time

import numpy as np
from matplotlib import pyplot as plt
from optimizers.combinatorial.base import check_path_distance
from optimizers.combinatorial.tsp import AntColonyTSPConfig, AntColonyTSP
from optimizers.core.types import AF
from optimizers.plot import plot_convergence
from sklearn.metrics import pairwise_distances

N_CITIES_CLUSTER = 40
N_CLUSTERS = N_CITIES_CLUSTER // 2

N_ANTS = 10 * N_CITIES_CLUSTER
N_GENERATIONS = 5 * N_CLUSTERS

CLUSTER_DIAMETER = 3
CLUSTER_SPACING = 10 * CLUSTER_DIAMETER

HALF_CIRCLE = False


def random_cities(center_x, center_y, n_cities=-1) -> np.ndarray:
    if n_cities == -1:
        n_cities = N_CITIES_CLUSTER
    # Randomly distribute cities in a uniform circle?
    theta = np.linspace(0, 2 * np.pi, n_cities + 1, dtype=np.float32)
    theta = theta[:-1]
    city_x = np.cos(theta) * CLUSTER_DIAMETER / 2.0 + center_x
    city_y = np.sin(theta) * CLUSTER_DIAMETER / 2.0 + center_y
    return np.c_[city_x, city_y]


def circle_random_clusters(n_clusters=-1, n_cities=-1) -> np.ndarray:
    if n_clusters == -1:
        n_clusters = N_CLUSTERS
    city_locations = np.zeros(shape=(0, 2), dtype=np.float32)
    for theta in np.linspace(0, 2 * np.pi, n_clusters):
        if HALF_CIRCLE:
            theta /= 2.0
        else:
            theta *= n_clusters / (n_clusters + 1)
        cx = CLUSTER_SPACING * np.cos(theta)
        cy = CLUSTER_SPACING * np.sin(theta)
        city_locations = np.concatenate(
            (city_locations, random_cities(cx, cy, n_cities=n_cities)), axis=0
        )
    return city_locations


def poly_perimeter(n_sides, r):
    # Compute perimeter of inscribed polygon in circle of radius, r.
    return n_sides * 2 * r * np.sin(2 * np.pi / (2 * n_sides))


def test_tsp():
    print("Configuring random")
    all_cities = circle_random_clusters()
    # Compute all distances
    distances: AF = pairwise_distances(all_cities)
    print("Distance-shape", distances.shape)

    approx_optimal_dist = (
        N_CLUSTERS * poly_perimeter(N_CITIES_CLUSTER, r=CLUSTER_DIAMETER / 2.0)
        + poly_perimeter(N_CITIES_CLUSTER, r=CLUSTER_SPACING)
        - N_CLUSTERS * CLUSTER_DIAMETER
    )
    if HALF_CIRCLE:
        approx_optimal_dist /= 2.0
    rand_dist = check_path_distance(
        distances, np.random.permutation(np.arange(N_CLUSTERS * N_CITIES_CLUSTER))
    )
    # Compute TSP optimized distance
    config = AntColonyTSPConfig(
        name="Test TSP", num_generations=N_GENERATIONS, population_size=N_ANTS
    )
    optimizer = AntColonyTSP(config, distances)
    tour_lengths = optimizer.solve()

    plot_convergence(tour_lengths)
