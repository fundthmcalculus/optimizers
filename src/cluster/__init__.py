import heapq
import numba
import numpy as np


@numba.jit(cache=True)
def compute_ordered_dis_njit_merge(matrix_of_pairwise_distance: np.ndarray) -> tuple[np.ndarray, list[int]]:
    N = matrix_of_pairwise_distance.shape[0]
    ordered_matrix: np.ndarray = np.zeros(matrix_of_pairwise_distance.shape)
    p: list[int] = vat_prim_mst(matrix_of_pairwise_distance)
    # Step 3 - since this is symmetric, we only have to do half
    for ij in range(N):
        for jk in range(ij,N):
            ordered_matrix[ij, jk] = ordered_matrix[jk, ij] = matrix_of_pairwise_distance[p[ij], p[jk]]

    # Step 4 - since this is symmetric, we only have to do half
    return ordered_matrix, p


@numba.jit(cache=True)
def vat_prim_mst(adj: np.ndarray) -> np.ndarray:
    N = len(adj)

    # Find the column of the maximum value.
    max_adj = np.argmax(adj)
    src = max_adj // N
    src_key = np.max(adj)

    # Create a list for keys and initialize all keys as infinite (INF)
    key: np.ndarray = np.full(N, float("inf"))

    # To store the parent array which, in turn, stores MST
    parent: np.ndarray = np.full(N, -1)

    # To keep track of vertices included in MST
    in_mst = np.full(N, False)

    # Insert the source itself into the priority queue and initialize its key as 0
    pq: list[tuple[float, int]] = [
        (src_key, src)
    ]  # Priority queue to store vertices that are being processed
    key[src] = src_key

    # The final sequence of vertices in MST
    heap_seq: np.ndarray = np.zeros(N, dtype=np.int32)
    heap_seq_idx = 0

    # Loop until the priority queue becomes empty
    while pq:
        # The first vertex in the pair is the minimum key vertex
        # Extract it from the priority queue
        # The vertex label is stored in the second of the pair
        u = heapq.heappop(pq)[1]

        # Different key values for the same vertex may exist in the priority queue.
        # The one with the least key value is always processed first.
        # Therefore, ignore the rest.
        if in_mst[u]:
            continue

        in_mst[u] = True  # Include the vertex in MST
        heap_seq[heap_seq_idx] = u
        heap_seq_idx += 1

        # Iterate through all adjacent vertices of a vertex
        # Parallel processing of adjacent vertices
        vertices = np.arange(N)
        mask = (vertices != u) & ~in_mst & (key[vertices] > adj[u, vertices])
        key[mask] = adj[u, mask]
        for v in vertices[mask]:
            heapq.heappush(pq, (key[v], v))
            parent[v] = u

    return heap_seq
