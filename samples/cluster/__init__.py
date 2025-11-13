import time

import numpy as np
from PIL import Image
from pyclustertend.visual_assessment_of_tendency import compute_ordered_dis_njit
from sklearn.metrics import pairwise_distances

from cluster import compute_ordered_dis_njit_merge

print("\n")
from ucimlrepo import fetch_ucirepo

# fetch dataset
# 59 is letter recognition
# 827 is sepsis survival (allocates 80+ GB RAM)
# 148 is shuttle stat log (allocates 50 GB RAM)
dataset_id = 148
letter_recognition = fetch_ucirepo(id=dataset_id)

# data (as pandas dataframes)
X = letter_recognition.data.features

# metadata
print(f"Metadata: {letter_recognition.metadata}")

# variable information
print(f"Variable Information: {letter_recognition.variables}")

# Compute the pairwise distances - float32 for space-saving.
matrix_of_pairwise_distance = np.log(pairwise_distances(X)+1).astype(np.float32)
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

print(f"Elapsed time for {len(X)} data points: {t1 - t0:.02f}")

# Save the ordered matrix as an image
img_array = (ordered_matrix * 255).astype(np.uint8)
img = Image.fromarray(img_array)
img.save(f'ordered_matrix_{dataset_id}.png')
t2 = time.time()
print(f"Elapsed time for {len(X)} data points image saving: {t2 - t1:.02f}")