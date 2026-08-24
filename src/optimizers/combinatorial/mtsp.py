from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering

from .base import TSPBase
from .aco import AntColonyTSPConfig, AntColonyTSP
from ..core.base import OptimizerResult, create_from_dict, literal_options
from ..core.random import get_seed
from ..core.types import AF

# NOTE: "FCM" is accepted but always raises NotImplementedError (see
# do_clustering) -- the fuzzy c-means dependency was unreliable enough that the
# implementation was pulled, but the option is left here (rather than removed)
# as a documented placeholder for whoever restores it.
ClusterMethod = Literal["kmeans", "spectral", "FCM"]


@dataclass
class AntColonyMTSPConfig(AntColonyTSPConfig):
    n_clusters: int = 10
    """Number of clusters to split the cities into"""
    clustering_method: ClusterMethod = "kmeans"


class AntColonyMTSP(TSPBase):
    """Multiple-TSP: cluster cities, then solve an independent ACO tour per cluster."""

    config: AntColonyMTSPConfig

    def __init__(self, *, config: AntColonyMTSPConfig, city_locations: AF):
        super().__init__(config=config, city_locations=city_locations)

    def solve(self, *, preserve_percent: float = 0.0) -> OptimizerResult:
        # Each cluster's ACO run is independent and could in principle run in
        # parallel, but every solver here draws from the seeded global RNG via
        # core.random.spawn_streams()/rng() -- and spawn_streams' docstring is
        # explicit that it must be called from a single thread (its counter
        # isn't synchronized). Running clusters concurrently would race on that
        # counter and break the reproducibility guarantee the RNG-determinism
        # work (core/random.py) established. Parallelizing this safely needs a
        # dedicated per-cluster stream handed in up front (spawned once, here,
        # before dispatch) rather than each cluster spawning its own -- left
        # sequential until that's done.
        clusters = self.do_clustering()

        results = []
        for cluster_id, cluster in enumerate(clusters):
            cluster_cities = self.city_locations[cluster, :]
            cluster_config = create_from_dict(self.config.__dict__, AntColonyTSPConfig)
            cluster_config.name = f"{self.config.name}-{cluster_id + 1}"
            tsp_solve = AntColonyTSP(
                config=cluster_config, city_locations=cluster_cities
            )
            cluster_result = tsp_solve.solve()
            # Map cluster indices back to original indices
            cluster_result.solution_vector = np.array(
                [cluster[i] for i in cluster_result.solution_vector]
            )

            results.append(cluster_result)

        optimal_paths = [result.solution_vector for result in results]

        return OptimizerResult(
            solution_history=[result.solution_history for result in results],
            solution_score=np.sum([result.solution_score for result in results]),
            stop_reason="max_iterations",
            solution_vector=optimal_paths,
        )

    def do_clustering(self) -> list[list[int]]:
        if self.config.clustering_method == "kmeans":
            kmeans = KMeans(n_clusters=self.config.n_clusters, random_state=get_seed())
            cluster_labels = kmeans.fit_predict(self.city_locations)
            # Group cities by cluster
            clusters: list[list[int]] = [[] for _ in range(self.config.n_clusters)]
            for i, label in enumerate(cluster_labels):
                clusters[label].append(i)
            return clusters
        elif self.config.clustering_method == "FCM":
            # Perform the fuzzy c-means clustering
            # fcm = FCM(n_clusters=self.config.n_clusters)
            # fcm.fit(self.city_locations)
            # cluster_labels = fcm.predict(self.city_locations)
            # clusters: list[list[int]] = [[] for _ in range(self.config.n_clusters)]
            # for i, label in enumerate(cluster_labels):
            #     clusters[label].append(i)
            # return clusters
            raise NotImplementedError("FCM package is unreliable")
        elif self.config.clustering_method == "spectral":
            sc = SpectralClustering(
                n_clusters=self.config.n_clusters,
                assign_labels="discretize",
                random_state=get_seed(),
            )
            cluster_labels = sc.fit_predict(self.city_locations)
            clusters = [[] for _ in range(self.config.n_clusters)]
            for i, label in enumerate(cluster_labels):
                clusters[label].append(i)
            return clusters
        else:
            allowed = ", ".join(repr(x) for x in literal_options(ClusterMethod))
            raise ValueError(
                f"Invalid clustering_method={self.config.clustering_method!r}. Allowed options: {allowed}"
            )
