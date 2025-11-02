from typing import List

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances

from optimizers.combinatorial.mtsp import AntColonyMTSPConfig, AntColonyMTSP
from optimizers.combinatorial.aco import (
    AntColonyTSPConfig,
    AntColonyTSP,
    NearestNeighborTSPConfig,
    NearestNeighborTSP,
    ConvexHullTSP,
    ConvexHullTSPConfig,
    TwoOptTSPConfig,
    TwoOptTSP,
    ThreeOptTSP,
)
from optimizers.core.types import AF, AI
from optimizers.plot import plot_convergence

N_CITIES_CLUSTER = 20
N_CLUSTERS = N_CITIES_CLUSTER // 2

N_ANTS = 10 * N_CITIES_CLUSTER
N_GENERATIONS = 5 * N_CLUSTERS

CLUSTER_DIAMETER = 3
CLUSTER_SPACING = 1 * CLUSTER_DIAMETER

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


def plot_cities_and_route(cities: AF, route: AI | List[AI]):
    fig = go.Figure()

    # Plot cities
    fig.add_trace(
        go.Scatter(
            x=cities[:, 0],
            y=cities[:, 1],
            mode="markers",
            name="Cities",
            marker=dict(size=8, color="blue"),
        )
    )

    if not isinstance(route, list):
        route = [route]

    # Plot route
    for ir, route in enumerate(route):
        route_cities = np.vstack(
            (cities[route], cities[route[0]])
        )  # Connect back to start
        fig.add_trace(
            go.Scatter(
                x=route_cities[:, 0],
                y=route_cities[:, 1],
                mode="lines",
                name=f"Route-{ir+1}",
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title="TSP Route",
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=True,
        template="plotly_white",
    )

    fig.show()


def test_aco_tsp():
    all_cities = circle_random_clusters()
    # Compute all distances
    distances: AF = pairwise_distances(all_cities)
    # Compute TSP optimized distance
    config = AntColonyTSPConfig(
        name="Test TSP",
        num_generations=N_GENERATIONS,
        population_size=N_ANTS,
        stop_after_iterations=5,
        joblib_prefer="threads",
    )
    optimizer = AntColonyTSP(
        config, network_routes=distances, city_locations=all_cities
    )
    result = optimizer.solve()
    plot_convergence(result.value_history)
    plot_cities_and_route(all_cities, result.optimal_path)


def test_nn_tsp():
    all_cities = circle_random_clusters()
    # Compute all distances
    distances: AF = pairwise_distances(all_cities)
    # Compute TSP optimized distance
    config = NearestNeighborTSPConfig(name="Test NN", back_to_start=True)
    optimizer = NearestNeighborTSP(config, distances)
    result = optimizer.solve()
    topt_config = TwoOptTSPConfig(name="2opt TSP", back_to_start=True)
    topt_optimizer = TwoOptTSP(
        topt_config,
        initial_route=result.optimal_path,
        initial_value=result.optimal_value,
        network_routes=distances,
    )
    result2 = topt_optimizer.solve()
    topt_config2 = TwoOptTSPConfig(
        name="2opt TSP", back_to_start=True, nearest_neighbors=5
    )
    topt_optimizer2 = ThreeOptTSP(
        topt_config2,
        initial_route=result.optimal_path,
        initial_value=result.optimal_value,
        network_routes=distances,
    )
    result3 = topt_optimizer2.solve()
    plot_cities_and_route(
        all_cities,
        [result.optimal_path, result2.optimal_path, result3.optimal_path],
    )


def test_convex_hull_tsp():
    all_cities = circle_random_clusters()
    # Compute all distances
    distances: AF = pairwise_distances(all_cities)
    # Compute TSP optimized distance
    config = ConvexHullTSPConfig(name="Test Convex Hull", back_to_start=True)
    optimizer = ConvexHullTSP(config, city_locations=all_cities)
    result = optimizer.solve()
    plot_cities_and_route(all_cities, result.optimal_path)


def test_mtsp():
    all_cities = circle_random_clusters()
    # Compute all distances
    # Compute TSP optimized distance
    config = AntColonyMTSPConfig(
        name="Test TSP",
        num_generations=N_GENERATIONS,
        population_size=N_ANTS,
        n_clusters=N_CLUSTERS,
        clustering_method="kmeans",
        stop_after_iterations=5,
        local_optimize=True,
    )
    optimizer = AntColonyMTSP(config, all_cities)
    result = optimizer.solve()
    plot_convergence(result.value_history)
    plot_cities_and_route(all_cities, result.optimal_path)
