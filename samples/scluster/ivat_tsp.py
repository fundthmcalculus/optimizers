# Generate city clusters
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances

from cluster import compute_ivat
from optimizers.combinatorial.strategy import TwoOptTSPConfig, TwoOptTSP
from test_cluster import identify_ivat_blocks
from test_combinatorics import circle_random_clusters

# n_clusters = 512
# n_cities = 64
# total_points = n_clusters * n_cities
# all_cities = circle_random_clusters(n_clusters=n_clusters, n_cities=n_cities, cluster_spacing=int(n_clusters/5), cluster_diameter=0.5)

file_name = "att532"  # "att532"
optimal_length = None
with open(f"./{file_name}.txt") as f:
    lines = [l.strip() for l in f.readlines()]

# Look for solution in header
for line in lines[:10]:
    if "SOLUTION" in line:
        optimal_length = int(line.split(":")[1])
        break

# Remove the lines until we hit the first entry, and remove the EOF line
lines = [line for line in lines if line.strip() and "EOF" not in line]
start_idx = [ij for ij, l in enumerate(lines) if l.strip().startswith("1")][0]
lines = lines[start_idx:]
lines = [x.split()[1:] for x in lines]
print(f"Solving TSP for {len(lines)} nodes")

all_cities = np.array([[float(s) for s in l] for l in lines])
n_cities = len(all_cities)
n_clusters = 1
total_points = n_cities * n_clusters

# Scramble the order to ensure we sort it!
cols = np.arange(len(all_cities), dtype="int")
rand_col_order = np.random.permutation(cols)
scramble_cities = all_cities[rand_col_order, :]

# Compute pairwise distances
matrix_of_pairwise_distance = pairwise_distances(scramble_cities, n_jobs=-1).astype(
    np.float32
)
scrambled_matrix = matrix_of_pairwise_distance.copy()

# Get VAT-optimized order using compute_ivat
t0 = time.time()
ivat_mst, vat_mst, ivat_order, vat_order = compute_ivat(scrambled_matrix)
t1 = time.time()

# Pick the blocks to optimize
(
    abrupt_change_indices,
    cluster_city_ids,
    diagonal_values,
    max_diff_index,
    peaks_threshold,
    sorted_diagonal,
) = identify_ivat_blocks(all_cities, ivat_mst, vat_order)

# TODO - Try optimizing the end-points between each cluster


def optimize_cluster(cluster_idx, cluster_cities):
    """Worker function to optimize a single cluster using 2-opt."""
    if len(cluster_cities) < 3:
        return cluster_idx, cluster_cities

    # Get distances for this cluster
    cluster_distances = matrix_of_pairwise_distance[
        np.ix_(cluster_cities, cluster_cities)
    ]

    # Set up 2-opt configuration (don't force return to start)
    two_opt_config = TwoOptTSPConfig(
        name=f"Cluster {cluster_idx} 2opt", back_to_start=False
    )

    # Create initial route (just sequential indices for the cluster)
    initial_route = list(range(len(cluster_cities)))

    # Initialize optimizer with cluster distances and initial route
    two_opt_optimizer = TwoOptTSP(
        two_opt_config,
        initial_route=initial_route,
        initial_value=np.sum(
            [
                cluster_distances[
                    initial_route[i], initial_route[(i + 1) % len(initial_route)]
                ]
                for i in range(len(initial_route))
            ]
        ),
        network_routes=cluster_distances,
    )

    # Solve 2-opt
    two_opt_result = two_opt_optimizer.solve()

    # Get optimized order within cluster (as indices)
    optimized_cluster_indices = two_opt_result.optimal_path

    # Map back to original city indices
    return cluster_idx, cluster_cities[optimized_cluster_indices]


# Parallelize cluster optimization using ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    # Submit all cluster optimization tasks
    futures = {
        executor.submit(optimize_cluster, cluster_idx, cluster_cities): cluster_idx
        for cluster_idx, cluster_cities in enumerate(cluster_city_ids)
    }

    # Collect results as they complete
    for future in as_completed(futures):
        cluster_idx, optimized_cities = future.result()
        cluster_city_ids[cluster_idx] = optimized_cities

# Get optimized city order from VAT ordering
vat_ordered_cities = scramble_cities[vat_order]

# Placeholder for 2-OPT optimized order (just use VAT order for now)
vat_2opt_order = np.concatenate(cluster_city_ids)
two_opt_ordered_cities = scramble_cities[vat_2opt_order]

t2 = time.time()

original_distance = np.sum(np.sqrt(np.sum(np.diff(all_cities, axis=0) ** 2, axis=1)))
scramble_distance = np.sum(
    np.sqrt(np.sum(np.diff(scramble_cities, axis=0) ** 2, axis=1))
)
vat_distance = np.sum(np.sqrt(np.sum(np.diff(vat_ordered_cities, axis=0) ** 2, axis=1)))
two_opt_distance = np.sum(
    np.sqrt(np.sum(np.diff(two_opt_ordered_cities, axis=0) ** 2, axis=1))
)

print(f"\n=== Scenario Statistics ===")
print(f"N Clusters:       {n_clusters}")
print(f"N Cities/Cluster: {n_cities}")
print(f"N Total:          {total_points}")
print(f"\n=== Runtime Statistics ===")
print(f"VAT Time: {(t1 - t0):.2f} seconds")
print(f"2-opt Time: {(t2 - t1):.2f} seconds")
print(f"Total Time: {(t2 - t0):.2f} seconds")
print(f"\n=== Distance Statistics ===")
print(f"Original Distance: {original_distance:.2f}")
print(f"Scrambled Distance: {scramble_distance:.2f}")
print(f"VAT Distance: {vat_distance:.2f}")
print(f"2-opt VAT Distance: {two_opt_distance:.2f}")
print(f"\n=== Improvements ===")
print(
    f"VAT vs Scrambled: {((scramble_distance - vat_distance) / scramble_distance * 100):.2f}% improvement"
)
print(
    f"2-opt vs VAT: {((vat_distance - two_opt_distance) / vat_distance * 100):.2f}% improvement"
)
print(
    f"2-opt vs Scrambled: {((scramble_distance - two_opt_distance) / scramble_distance * 100):.2f}% improvement"
)

# Create 2x2 subplot figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("City Order Comparisons: Original, Scrambled, and VAT")

# Plot 1: Original random city order
axes[0, 0].plot(all_cities[:, 0], all_cities[:, 1], c="blue", alpha=0.7)
axes[0, 0].text(
    0.05,
    0.95,
    f"Distance: {original_distance:.2f}",
    transform=axes[0, 0].transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
axes[0, 0].set_title("Original Order")
axes[0, 0].set_xlabel("X Coordinate")
axes[0, 0].set_ylabel("Y Coordinate")

# Plot 2: VAT optimized city order
axes[0, 1].plot(scramble_cities[:, 0], scramble_cities[:, 1], c="green", alpha=0.7)
axes[0, 1].text(
    0.05,
    0.95,
    f"Distance: {scramble_distance:.2f}",
    transform=axes[0, 1].transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
axes[0, 1].set_title("Scrambled Order")
axes[0, 1].set_xlabel("X Coordinate")
axes[0, 1].set_ylabel("Y Coordinate")

# Plot 3: Placeholder for 2-OPT optimized order (using VAT order)
axes[1, 0].plot(
    vat_ordered_cities[:, 0], vat_ordered_cities[:, 1], c="red", alpha=0.7, label="VAT"
)
axes[1, 0].plot(
    two_opt_ordered_cities[:, 0],
    two_opt_ordered_cities[:, 1],
    c="black",
    alpha=0.7,
    label="2-opt VAT",
)
axes[1, 0].text(
    0.05,
    0.95,
    f"VAT Dist: {vat_distance:.2f}\n" f"2-opt Dist: {two_opt_distance:.2f}",
    transform=axes[1, 0].transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
axes[1, 0].text(
    0.95,
    0.05,
    f"VAT Time: {(t1-t0):.2f}\n" f"2-opt Time: {(t2-t1):.2f}",
    transform=axes[1, 0].transAxes,
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
axes[1, 0].set_title("VAT-Order")
axes[1, 0].set_xlabel("X Coordinate")
axes[1, 0].set_ylabel("Y Coordinate")

# Plot 4: VAT matrix visualization for reference
im = axes[1, 1].imshow(vat_mst, cmap="viridis")
axes[1, 1].set_title("VAT Matrix")
axes[1, 1].text(
    0.05,
    0.05,
    f"N_clusters: {n_clusters}\n"
    f"N_cities: {n_cities}\n"
    f"N_total: {total_points}\n",
    transform=axes[1, 1].transAxes,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plt.show()
