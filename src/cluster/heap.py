import numpy as np
from numba import njit, int32, float32, void, boolean

# Define the structure of the heap entry
# w: float32, u: int32, v: int32

@njit(void(float32[:], int32[:], int32[:], int32, float32, int32, int32), cache=True, nogil=True)
def heappush(heap_w, heap_u, heap_v, size, w, u, v):
    """Push a (w, u, v) tuple onto the heap."""
    # Add new element at the end
    idx = size
    heap_w[idx] = w
    heap_u[idx] = u
    heap_v[idx] = v
    
    # Sift up
    while idx > 0:
        parent = (idx - 1) >> 1
        if heap_w[idx] < heap_w[parent]:
            # Swap
            tmp_w = heap_w[idx]
            heap_w[idx] = heap_w[parent]
            heap_w[parent] = tmp_w
            
            tmp_u = heap_u[idx]
            heap_u[idx] = heap_u[parent]
            heap_u[parent] = tmp_u
            
            tmp_v = heap_v[idx]
            heap_v[idx] = heap_v[parent]
            heap_v[parent] = tmp_v
            
            idx = parent
        else:
            break

@njit(void(float32[:], int32[:], int32[:], int32), cache=True, nogil=True)
def heappop(heap_w, heap_u, heap_v, size):
    """Pop the minimum (w, u, v) tuple from the heap."""
    if size <= 0:
        return

    # Replace root with the last element
    last_idx = size - 1
    heap_w[0] = heap_w[last_idx]
    heap_u[0] = heap_u[last_idx]
    heap_v[0] = heap_v[last_idx]
    
    # Sift down
    idx = 0
    new_size = size - 1
    while True:
        left = (idx << 1) + 1
        right = (idx << 1) + 2
        smallest = idx
        
        if left < new_size and heap_w[left] < heap_w[smallest]:
            smallest = left
        if right < new_size and heap_w[right] < heap_w[smallest]:
            smallest = right
            
        if smallest != idx:
            # Swap
            tmp_w = heap_w[idx]
            heap_w[idx] = heap_w[smallest]
            heap_w[smallest] = tmp_w
            
            tmp_u = heap_u[idx]
            heap_u[idx] = heap_u[smallest]
            heap_u[smallest] = tmp_u
            
            tmp_v = heap_v[idx]
            heap_v[idx] = heap_v[smallest]
            heap_v[smallest] = tmp_v
            
            idx = smallest
        else:
            break
