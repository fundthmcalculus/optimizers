import gc
import os
import time
import numpy as np
import optuna
import pandas as pd
import plotly.graph_objs as go
from optuna.importance import get_param_importances
from sklearn.metrics import pairwise_distances


from optimizers.combinatorial.aco import AntColonyTSPConfig, AntColonyTSP
from optimizers.combinatorial.ga import GeneticAlgorithmTSP, GeneticAlgorithmTSPConfig
from optimizers.combinatorial.strategy import ConvexHullTSPConfig, ConvexHullTSP, TwoOptTSP, TwoOptTSPConfig, \
    NearestNeighborTSP, NearestNeighborTSPConfig
from optimizers.core.types import AF
from optimizers.plot import plot_cities_and_route, plot_convergence

def main():
    city_locations = demo_data()
    # Now let's do parameter tuning on the GA and ACO.
    n_trials=50
    study: optuna.study.Study = aco_parameter_tuning(city_locations=city_locations,n_trials=n_trials)
    print("Best ACO parameters:", study.best_params)

    study: optuna.study.Study = ga_parameter_tuning(city_locations=city_locations,n_trials=n_trials)
    print("Best GA parameters:", study.best_params)

    # Baseline case
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

def aco_parameter_tuning(city_locations, n_trials):
    def aco_objective(trial):
        params = {
            # "alpha": trial.suggest_float("alpha", 0.2, 2),
            # "beta": trial.suggest_float("beta", 0.2, 2),
            # "rho": trial.suggest_float("rho", 0.1, 0.9),
            # "q": trial.suggest_float("q", 1.0, 10.0),
            "population_size": trial.suggest_int("population_size", 10, 100),
            "num_generations": trial.suggest_int("num_generations", 50, 500),
            "solution_archive_size": trial.suggest_int("solution_archive_size", 50, 500)
        }
        aco_config = AntColonyTSPConfig(
            name="ACO TSP",
            back_to_start=True,
            joblib_prefer="processes",
            **params
        )
        aco = AntColonyTSP(config=aco_config, city_locations=city_locations)
        result = aco.solve()

        gc.collect() # avoids some RAM problems

        return result.optimal_value

    study = optuna.create_study(direction="minimize")
    study.optimize(aco_objective, n_trials=n_trials, show_progress_bar=False)

    plot_study_factors(study)

    return study

def ga_parameter_tuning(city_locations, n_trials):
    def ga_objective(trial):
        params = {
            # "mutation_rate": trial.suggest_float("mutation_rate", 0.1, 1.0, log=True),
            # "crossover_rate": trial.suggest_float("crossover_rate", 0.1, 1.0, log=True),
            "population_size": trial.suggest_int("population_size", 10, 100),
            "num_generations": trial.suggest_int("num_generations", 50, 500),
            "solution_archive_size": trial.suggest_int("solution_archive_size", 50, 500)
        }
        aco_config = GeneticAlgorithmTSPConfig(
            name="GA TSP",
            back_to_start=True,
            joblib_prefer="processes",
            **params
        )
        aco = GeneticAlgorithmTSP(config=aco_config, city_locations=city_locations)
        result = aco.solve()

        gc.collect() # avoids some RAM problems

        return result.optimal_value

    study = optuna.create_study(direction="minimize")
    study.optimize(ga_objective, n_trials=n_trials, show_progress_bar=False)

    plot_study_factors(study)

    return study


def plot_study_factors(study: optuna.study.Study):
    # Our own importance plot
    importances = get_param_importances(study)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(importances.values()),
        y=list(importances.keys()),
        orientation='h'
    ))

    # Create annotation text
    annotation_text = f"Best Score: {study.best_value:.4f}<br>"
    for key, value in study.best_params.items():
        annotation_text += f"{key}: {value:.4f}<br>"

    fig.add_annotation(
        x=1.1,
        y=1.1,
        xref="paper",
        yref="paper",
        text=annotation_text,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )

    fig.update_layout(
        title="Hyperparameter Importance",
        xaxis_title="Importance",
        yaxis_title="Hyperparameter",
        height=400,
        width=800
    )
    
    fig.show()


def compute_tsp_bounds(cities: AF, aco_params: dict | None = None, ga_params: dict | None = None):
    aco_params = aco_params or dict()
    ga_params = ga_params or dict()
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
        num_generations=n_generations * 3,  # NOTE - Anecdotally, GA runs about 3x faster than ACO, but worse.
        population_size=n_ants,
        solution_archive_size=solution_archive_size,
        joblib_prefer="processes",
        stop_after_iterations=n_generations * 3,  # No early stopping!
        **ga_params
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
        joblib_prefer="processes",
        stop_after_iterations=n_generations,  # No early stopping!
        **aco_params
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


def demo_data(name: str = "berlin52") -> AF:
    """Load city coordinates from TSP instance dataset.

    Args:
        name: Name of the TSP instance to load

    Returns:
        AF: Array of city coordinates with shape (n_cities, 2)
    """
    # Load the TSP instance data
    city_data = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "tsp_instances_dataset.csv"),
    )
    city_row = city_data[city_data['TSP_Instance'] == name]

    # Find X/Y coordinate columns
    x_cols = [col for col in city_row.columns if col.startswith('City_') and col.endswith('_X')]
    y_cols = [col for col in city_row.columns if col.startswith('City_') and col.endswith('_Y')]

    # Extract coordinates, dropping any NaN values
    coords = []
    for x_col, y_col in zip(x_cols, y_cols):
        x = city_row[x_col].iloc[0]
        y = city_row[y_col].iloc[0]
        if pd.notna(x) and pd.notna(y):
            coords.append([x, y])

    coords = np.array(coords)
    return coords

if __name__ == "__main__":
    main()