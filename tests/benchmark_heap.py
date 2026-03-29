import time
import numpy as np
from src.cluster.mergevat import vat_prim_mst, vat_prim_mst_custom

def benchmark_vat_mst():
    # Generate some synthetic data
    N = 20000
    np.random.seed(42)
    data = np.random.rand(N, 2).astype(np.float32)
    
    # Compute pairwise distance matrix
    diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    adj = np.sqrt(np.sum(np.square(diff), axis=-1)).astype(np.float32)

    print(f"Benchmarking with N={N}...")

    # Warm up
    _ = vat_prim_mst(adj)
    _ = vat_prim_mst_custom(adj)

    # Benchmark original
    start_time = time.time()
    for _ in range(10):
        h_seq1, p_seq1 = vat_prim_mst(adj)
    end_time = time.time()
    original_time = (end_time - start_time) / 10
    print(f"Original vat_prim_mst (heapq) average time: {original_time:.6f} seconds")

    # Benchmark custom
    start_time = time.time()
    for _ in range(10):
        h_seq2, p_seq2 = vat_prim_mst_custom(adj)
    end_time = time.time()
    custom_time = (end_time - start_time) / 10
    print(f"Custom vat_prim_mst (binary heap) average time: {custom_time:.6f} seconds")

    print(f"Speedup: {original_time / custom_time:.2f}x")

    # Verify results are identical
    np.testing.assert_array_equal(h_seq1, h_seq2)
    np.testing.assert_array_equal(p_seq1, p_seq2)
    print("Verification successful: Results are identical.")

if __name__ == "__main__":
    benchmark_vat_mst()
