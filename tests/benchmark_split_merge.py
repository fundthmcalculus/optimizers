import time
import numpy as np
import matplotlib.pyplot as plt
from src.cluster.mergevat import vat_prim_mst_custom, vat_prim_mst_split_merge

def verify_split_merge():
    n_samples = 5
    # Generate some synthetic data
    N = 20000
    np.random.seed(42)

    # Create 7 distinct clusters
    n_clusters = 7
    points_per_cluster = N // n_clusters
    data = []

    for i in range(n_clusters):
        # Generate cluster centers spread out in 2D space
        center_x = (i % 3) * 3.0 + np.random.rand() * 0.5
        center_y = (i // 3) * 3.0 + np.random.rand() * 0.5

        # Generate points around each center with small variance
        cluster_points = np.random.randn(points_per_cluster, 2) * 0.3
        cluster_points[:, 0] += center_x
        cluster_points[:, 1] += center_y
        data.append(cluster_points)

    # Handle remaining points (if N is not divisible by n_clusters)
    remaining = N - points_per_cluster * n_clusters
    if remaining > 0:
        center_x = np.random.rand() * 9.0
        center_y = np.random.rand() * 9.0
        remaining_points = np.random.randn(remaining, 2) * 0.3
        remaining_points[:, 0] += center_x
        remaining_points[:, 1] += center_y
        data.append(remaining_points)

    data = np.vstack(data).astype(np.float32)
    
    # Compute pairwise distance matrix
    diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    adj = np.sqrt(np.sum(np.square(diff), axis=-1)).astype(np.float32)

    print(f"Verifying with N={N}...")

    _, _ = vat_prim_mst_custom(adj)
    _, _ = vat_prim_mst_split_merge(adj)

    # Compute both
    h_seq_orig, p_seq_orig = vat_prim_mst_custom(adj)
    h_seq_split, p_seq_split = vat_prim_mst_split_merge(adj)

    # Note: Sequences might not be identical because of split/merge logic and different starting points
    # But the resulting MST should be "valid".
    # More importantly, let's see if it runs and produces a valid permutation.
    
    assert len(np.unique(h_seq_split)) == N, "Split-merge did not produce a valid permutation"
    print("Verification: Valid permutation produced.")

    # Benchmark
    start = time.time()
    for _ in range(n_samples):
        vat_prim_mst_custom(adj)
    custom_time = (time.time() - start) / n_samples
    print(f"Custom time: {custom_time:.6f}s")

    start = time.time()
    for _ in range(n_samples):
        vat_prim_mst_split_merge(adj)
    split_merge_time = (time.time() - start) / n_samples
    print(f"Split-merge time: {split_merge_time:.6f}s")

    print(f"Speedup: {custom_time / split_merge_time:.2f}x")

    # Create visualization of sorted arrays
    sorted_adj_orig = adj[h_seq_orig, :][:, h_seq_orig]
    sorted_adj_split = adj[h_seq_split, :][:, h_seq_split]

    fig, axes = plt.subplots(2, 1, figsize=(8, 12))

    im0 = axes[0].imshow(sorted_adj_orig, cmap='viridis')
    axes[0].set_title('Custom VAT - Sorted Adjacency Matrix')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Index')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(sorted_adj_split, cmap='viridis')
    axes[1].set_title('Split-Merge VAT - Sorted Adjacency Matrix')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Index')
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    verify_split_merge()
