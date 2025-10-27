from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import tqdm
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

from .base import check_path_distance, CombinatoricsResult
from .tsp import AntColonyTSPConfig, AntColonyTSP
from ..core.base import IOptimizerConfig, create_from_dict
from ..core.types import AI, AF, F, i32, i16


# TODO - Handle other clustering methods.
# TODO - Handle GA method of clustering.
ClusterMethod = Literal["kmeans"]


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
                cluster_config = create_from_dict(self.config.__dict__,AntColonyTSPConfig)
                cluster_config.name = f"{self.config.name}-{cluster_id+1}"
                # Not relevant, but for clarity
                cluster_config.n_clusters = 1
                tsp_solve = AntColonyTSP(cluster_config, city_locations=cluster_cities)
                cluster_result = tsp_solve.solve()
                # Map cluster indices back to original indices
                cluster_result.optimal_path = np.array([cluster[i] for i in cluster_result.optimal_path])

                results.append(cluster_result)
            return CombinatoricsResult(
                value_history=np.vstack([result.value_history for result in results]),
                optimal_value=np.sum([result.optimal_value for result in results]),
                stop_reason="max_iterations",
                optimal_path=np.vstack([result.optimal_path for result in results])
            )
        else:
            raise ValueError("Only kmeans clustering is supported")
