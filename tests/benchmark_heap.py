# Plot scaling comparison
import matplotlib.pyplot as plt
import time
import numpy as np

from cluster.mergevat import shuffle_array
from src.cluster.mergevat import vat_prim_mst, vat_prim_mst_custom

def benchmark_vat_mst():
    # Generate some synthetic data and benchmark across different sizes
    np.random.seed(42)

    # Define size ranges
    sample_sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]

    # Storage for results
    results = {
        'n_samples': [],
        'n_passes': [],
        'original_time': [],
        'custom_time': [],
        'speedup': []
    }

    for n_samples in sample_sizes:
        # Scale passes inversely with size: more passes for smaller sizes
        # This keeps total computation roughly constant
        n_passes = max(1, int(10000 / n_samples))

        print(f"\nBenchmarking with N={n_samples}, passes={n_passes}...")

        data = np.random.rand(n_samples, 2).astype(np.float32)

        # Compute pairwise distance matrix
        diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
        adj = np.sqrt(np.sum(np.square(diff), axis=-1)).astype(np.float32)

        # Print matrix size in MB
        matrix_size_mb = adj.nbytes / (1024 * 1024)
        print(f"Adjacency matrix size: {matrix_size_mb:.2f} MB")

        # Benchmark original
        start_time = time.time()
        for _ in range(n_passes):
            h_seq1, p_seq1 = vat_prim_mst(adj)
        end_time = time.time()
        original_time = (end_time - start_time) / n_passes
        print(f"Original vat_prim_mst (heapq) average time: {original_time:.3f} seconds")

        # Benchmark custom
        ts= []
        start_time = time.time()
        for _ in range(n_passes):
            t0 = time.time_ns()
            h_seq2, p_seq2 = vat_prim_mst_custom(adj)
            t1 = time.time_ns()
            shuffle_array(True, adj, p_seq2)
            t2 = time.time_ns()
            ts.append([t0, t1, t2])
        end_time = time.time()
        custom_time = (end_time - start_time) / n_passes
        print(f"Custom vat_prim_mst (binary heap) average time: {custom_time:.3f} seconds")
        ts = np.array(ts)
        dt = np.diff(ts, axis=1)
        dt = np.mean(dt, axis=0)/1E9
        print(f"Average vat_prim_mst: {dt[0]:.2f}, shuffle: {dt[1]:.2f}")

        speedup = original_time / custom_time
        print(f"Speedup: {speedup:.2f}x")

        # Verify results are identical
        np.testing.assert_array_equal(h_seq1, h_seq2)
        np.testing.assert_array_equal(p_seq1, p_seq2)
        print("Verification successful: Results are identical.")

        # Store results
        results['n_samples'].append(n_samples)
        results['n_passes'].append(n_passes)
        results['original_time'].append(original_time)
        results['custom_time'].append(custom_time)
        results['speedup'].append(speedup)

    plot_results(results)

def plot_results(results):

    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Plot 1: Runtime vs N
    ax1.loglog(results['n_samples'], results['original_time'], 'o-', label='Original (heapq)', linewidth=2)
    ax1.loglog(results['n_samples'], results['custom_time'], 's-', label='Custom (binary heap)', linewidth=2)
    ax1.set_xlabel('Number of Samples (N)')
    ax1.set_ylabel('Average Runtime (seconds)')
    ax1.set_title('Runtime Scaling Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Speedup vs N
    ax2.semilogx(results['n_samples'], results['speedup'], 'o-', linewidth=2, color='green')
    ax2.axhline(y=1.0, color='r', linestyle='--', label='No speedup')
    ax2.set_xlabel('Number of Samples (N)')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Speedup vs Problem Size')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('benchmark_scaling.png', dpi=150)
    plt.show()

    print(f"\n{'=' * 60}")
    print(f"Summary of scaling results:")
    print(f"{'=' * 60}")
    for i in range(len(results['n_samples'])):
        print(f"N={results['n_samples'][i]:5d}, passes={results['n_passes'][i]:4d}, "
              f"speedup={results['speedup'][i]:.2f}x")

if __name__ == "__main__":
    benchmark_vat_mst()
