import numpy as np
from numba import njit, void, float32, int32

# Define the structured type for the heap
heap_dt = np.dtype([('w', np.float32), ('u', np.int32), ('v', np.int32)])

@njit(void(float32[:, :], int32, float32, int32, int32), cache=True, nogil=True, inline='always')
def heappush(heap, size, w, u, v):
    """Push a (w, u, v) tuple onto the heap."""
    # Add new element at the end
    idx = size
    
    # Sift up
    while idx > 0:
        parent = (idx - 1) >> 1
        pw = heap[parent, 0]
        if w < pw:
            heap[idx, 0] = pw
            heap[idx, 1] = heap[parent, 1]
            heap[idx, 2] = heap[parent, 2]
            idx = parent
        else:
            break
    heap[idx, 0] = w
    heap[idx, 1] = u
    heap[idx, 2] = v

@njit(void(float32[:, :], int32), cache=True, nogil=True, inline='always')
def heappop(heap, size):
    """Pop the minimum (w, u, v) tuple from the heap."""
    if size <= 0:
        return

    # Replace root with the last element
    last_idx = size - 1
    w = heap[last_idx, 0]
    u = heap[last_idx, 1]
    v = heap[last_idx, 2]
    
    # Sift down
    idx = 0
    new_size = size - 1
    while True:
        left = (idx << 1) + 1
        right = (idx << 1) + 2
        smallest = idx
        
        if left < new_size:
            lw = heap[left, 0]
            if lw < w:
                smallest = left
                if right < new_size:
                    rw = heap[right, 0]
                    if rw < lw:
                        smallest = right
            elif right < new_size:
                rw = heap[right, 0]
                if rw < w:
                    smallest = right
            
        if smallest != idx:
            heap[idx, 0] = heap[smallest, 0]
            heap[idx, 1] = heap[smallest, 1]
            heap[idx, 2] = heap[smallest, 2]
            idx = smallest
        else:
            break
    heap[idx, 0] = w
    heap[idx, 1] = u
    heap[idx, 2] = v
