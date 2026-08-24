from dataclasses import dataclass
from typing import Literal

import numpy as np
from joblib import Parallel, cpu_count, delayed
from sklearn.cluster import KMeans, SpectralClustering

from .base import TSPBase
from .aco import AntColonyTSPConfig, AntColonyTSP
from ..core.base import OptimizerResult, create_from_dict, literal_options
from ..core.random import get_seed, spawn_stream_roots, use_stream_root
from ..core.types import AF

# NOTE: "FCM" is accepted but always raises NotImplementedError (see
# do_clustering) -- the fuzzy c-means dependency was unreliable enough that the
# implementation was pulled, but the option is left here (rather than removed)
# as a documented placeholder for whoever restores it.
ClusterMethod = Literal["kmeans", "spectral", "FCM"]


def _solve_cluster(
    cluster_id: int,
    cluster: list[int],
    city_locations: AF,
    base_config: "AntColonyMTSPConfig",
    n_jobs: int,
    stream_root: np.random.SeedSequence,
) -> OptimizerResult:
    """Solve one cluster's independent ACO tour. Module-level so it's picklable
    for joblib's ``processes`` backend; ``stream_root`` isolates this cluster's
    internal RNG draws from every other concurrently-running cluster."""
    with use_stream_root(stream_root):
        cluster_cities = city_locations[cluster, :]
        cluster_config = create_from_dict(base_config.__dict__, AntColonyTSPConfig)
        cluster_config.name = f"{base_config.name}-{cluster_id + 1}"
        cluster_config.n_jobs = n_jobs
        tsp_solve = AntColonyTSP(config=cluster_config, city_locations=cluster_cities)
        cluster_result = tsp_solve.solve()
        # Map cluster indices back to original indices
        cluster_result.solution_vector = np.array(
            [cluster[i] for i in cluster_result.solution_vector]
        )
        return cluster_result


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
        clusters = self.do_clustering()
        n_clusters = len(clusters)

        # Split the configured processor budget between cluster-level and
        # per-cluster ant-level parallelism instead of giving every cluster
        # the full budget (which would oversubscribe once clusters run
        # concurrently) -- this is what the old TODO here ("handle the number
        # of processors based upon parallel clusters") was asking for.
        total_jobs = self.config.n_jobs if self.config.n_jobs > 0 else cpu_count() - 1
        outer_jobs = max(1, min(n_clusters, total_jobs))
        inner_jobs = max(1, total_jobs // outer_jobs)

        # Each cluster's ACO run draws from the seeded RNG via
        # core.random.spawn_streams()/rng() internally, and spawn_streams'
        # counter isn't itself synchronized across threads. Spawn one
        # independent stream *root* per cluster up front, here, from the
        # single (calling) thread, then have each cluster task stand in its
        # own root for the duration of its run (use_stream_root) so its
        # internal spawn_streams() calls draw from that root instead of
        # racing on the shared global one -- see core/random.py.
        stream_roots = spawn_stream_roots(n_clusters)

        results = Parallel(n_jobs=outer_jobs, prefer=self.config.joblib_prefer)(
            delayed(_solve_cluster)(
                cluster_id,
                cluster,
                self.city_locations,
                self.config,
                inner_jobs,
                stream_roots[cluster_id],
            )
            for cluster_id, cluster in enumerate(clusters)
        )

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
