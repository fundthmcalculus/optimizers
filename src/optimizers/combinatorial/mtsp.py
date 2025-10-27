from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.cluster import KMeans

from .base import CombinatoricsResult
from .tsp import AntColonyTSPConfig, AntColonyTSP
from ..continuous.aco import AntColonyOptimizer, AntColonyOptimizerConfig
from ..continuous.variables import InputDiscreteVariable
from ..core.base import create_from_dict
from ..core.types import AF

# TODO - Handle other clustering methods.
# TODO - Handle GA method of clustering.
ClusterMethod = Literal["kmeans", "TSP"]


@dataclass
class AntColonyMTSPConfig(AntColonyTSPConfig):
    n_clusters: int = 10
    """Number of clusters to split the cities into"""
    clustering_method: ClusterMethod = "kmeans"


class AntColonyMTSP:
    def __init__(self, config: AntColonyMTSPConfig, city_locations: AF):
        self.config = config
        self.city_locations: AF = city_locations.copy()

    def solve(self) -> CombinatoricsResult:
        # TODO - Handle the number of processors based upon parallel clusters?
        if self.config.clustering_method == "kmeans":
            return self.solve_kmeans()
        elif self.config.clustering_method == "TSP":
            # Create n_cities DiscreteVariable with the options being each cluster
            cluster_ids = np.arange(self.config.n_clusters)
            variables = [
                InputDiscreteVariable(f"cluster_{i}", cluster_ids)
                for i in range(self.city_locations.shape[0])
            ]
            solver_config = create_from_dict(
                self.config.__dict__, AntColonyOptimizerConfig
            )
            tsp_config = create_from_dict(self.config.__dict__, AntColonyTSPConfig)
            # Because this is a multi-level optimization, don't parallelize here.
            solver_config.joblib_num_procs = 1

            def goal_fcn(x: AF) -> float:
                x = np.int32(x)
                # Iterate over the number of clusters, and do each cluster's TSP optimization separately.
                total_value = 0.0
                for cluster_id in range(self.config.n_clusters):
                    cluster_cities = self.city_locations[x == cluster_id, :]
                    if len(cluster_cities) == 0:
                        continue
                    tsp_solve = AntColonyTSP(
                        f"tsp-cluster-{cluster_id+1}",
                        tsp_config,
                        city_locations=cluster_cities,
                    )
                    tsp_result = tsp_solve.solve()
                    total_value += tsp_result.optimal_value
                return total_value

            solver = AntColonyOptimizer(
                name="TSP cluster optimizer",
                config=solver_config,
                variables=variables,
                fcn=goal_fcn,
            )
            result = solver.solve()

            raise ValueError("TSP clustering is not yet supported")
        else:
            raise ValueError(
                f"Unknown clustering method: {self.config.clustering_method}"
            )

    def solve_kmeans(self) -> CombinatoricsResult:
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=self.config.n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.city_locations)
        cluster_centers = kmeans.cluster_centers_
        # Group cities by cluster
        clusters = [[] for _ in range(self.config.n_clusters)]
        for i, label in enumerate(cluster_labels):
            clusters[label].append(i)

        results = []
        for cluster_id, cluster in enumerate(clusters):
            cluster_cities = self.city_locations[cluster, :]
            cluster_config = create_from_dict(self.config.__dict__, AntColonyTSPConfig)
            cluster_config.name = f"{self.config.name}-{cluster_id + 1}"
            # Not relevant, but for clarity
            cluster_config.n_clusters = 1
            tsp_solve = AntColonyTSP(cluster_config, city_locations=cluster_cities)
            cluster_result = tsp_solve.solve()
            # Map cluster indices back to original indices
            cluster_result.optimal_path = np.array(
                [cluster[i] for i in cluster_result.optimal_path]
            )

            results.append(cluster_result)
        return CombinatoricsResult(
            value_history=np.vstack([result.value_history for result in results]),
            optimal_value=np.sum([result.optimal_value for result in results]),
            stop_reason="max_iterations",
            optimal_path=np.vstack([result.optimal_path for result in results]),
        )
