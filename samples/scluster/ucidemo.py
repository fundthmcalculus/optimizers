import time

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo
from cluster import compute_ordered_dis_njit_merge, compute_ivat

print("\n")


def pairwise_distances2(data, dtype=np.float32) -> np.ndarray:
    dist_arr = np.zeros((data.shape[0], data.shape[0]), dtype=dtype)
    for i in tqdm(range(len(data))):
        dist_arr[i, i + 1:] = np.linalg.norm(data[i] - data[i + 1:], axis=1)
        dist_arr[i + 1:, i] = dist_arr[i, i + 1:]
    return dist_arr

def to_np_array(data: pd.DataFrame) -> np.ndarray:
    y = data.to_numpy(np.float32)
    return y

# fetch dataset
# 59 is letter recognition
# 827 is sepsis survival (allocates 80+ GB RAM)
# 148 is shuttle stat log (allocates 50 GB RAM)
dataset_id = 59
letter_recognition = fetch_ucirepo(id=dataset_id)

# data (as pandas dataframes)
X = letter_recognition.data.features
l_x = len(X)
X = to_np_array(X)

# metadata
print(f"Metadata: {letter_recognition.metadata}")

# variable information
print(f"Variable Information: {letter_recognition.variables}")

# Compute the pairwise distances - float32 for space-saving.
matrix_of_pairwise_distance = np.log(pairwise_distances2(X) + 1).astype(np.float32)
del X
matrix_of_pairwise_distance = (
        matrix_of_pairwise_distance / matrix_of_pairwise_distance.max()
)
print(f"Pairwise distance matrix shape: {matrix_of_pairwise_distance.shape}")
t0 = time.time()
# ordered_matrix = compute_ordered_dis_njit(matrix_of_pairwise_distance)
ordered_matrix, path_merge = compute_ordered_dis_njit_merge(
    matrix_of_pairwise_distance, inplace=True
)
t1 = time.time()

print(f"Elapsed time for {l_x} data points: {t1 - t0:.02f}")

# Save the ordered matrix as an image
img_array = (ordered_matrix * 255).astype(np.uint8)
img = Image.fromarray(img_array)
img.save(f"ordered_matrix_{dataset_id}.png")
t2 = time.time()
print(f"Elapsed time for {l_x} data points image saving: {t2 - t1:.02f}")
