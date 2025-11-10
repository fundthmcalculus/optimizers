import heapq
import numba
import numpy as np


@numba.jit
def compute_ordered_dis_njit_merge(matrix_of_pairwise_distance: np.ndarray):
    N = matrix_of_pairwise_distance.shape[0]
    ordered_matrix = np.zeros(matrix_of_pairwise_distance.shape)
    p: list[int] = vat_prim_mst(matrix_of_pairwise_distance)
    # Replace the only "-1" with the index of the maximum value.
    p = np.array(p).astype(np.int32)
    # Step 3:
    for column_index_of_maximum_value in range(N):
        for j in range(N):
            ordered_matrix[column_index_of_maximum_value, j] = (
                matrix_of_pairwise_distance[p[column_index_of_maximum_value], p[j]]
            )

    # Step 4 :
    return ordered_matrix, p


@numba.jit
def vat_prim_mst(adj: np.ndarray) -> list[int]:
    N = len(adj)

    # Find the column of the maximum value.
    max_adj = np.argmax(adj)
    src = max_adj // N
    src_key = np.max(adj)

    # Create a list for keys and initialize all keys as infinite (INF)
    key: list[float] = [float("inf")] * N

    # To store the parent array which, in turn, stores MST
    parent: list[int] = [-1] * N

    # To keep track of vertices included in MST
    in_mst: list[bool] = [False] * N

    # Insert the source itself into the priority queue and initialize its key as 0
    pq: list[tuple[float, int]] = [
        (src_key, src)
    ]  # Priority queue to store vertices that are being processed
    # heapq.heappush(pq, (src_key, src))
    key[src] = src_key

    # The final sequence of vertices in MST
    heap_seq: list[int] = [-1]

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
        heap_seq.append(u)

        # Iterate through all adjacent vertices of a vertex
        for v in range(N):
            if v == u:
                continue
            weight = adj[u, v]
            # If v is not in MST and the weight of (u, v) is smaller than the current key of v
            if not in_mst[v] and key[v] > weight:
                # Update the key of v
                key[v] = weight
                heapq.heappush(pq, (key[v], v))
                parent[v] = u

    # Remove the preallocated sequence entry of `-1`
    return heap_seq[1:]
