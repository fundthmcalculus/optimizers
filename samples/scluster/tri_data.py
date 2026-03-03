import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from PIL import Image

from cluster import compute_ordered_dis_njit_merge

input_file = "data_preprocessed1.csv"

df = pd.read_csv(input_file, dtype=np.float32)
# Take the log of the age column and the really big column.
df.drop(columns=["LENGTH_OF_STAY_MINS"], inplace=True)
df['AGE_ENCOUNTER'] = df['AGE_ENCOUNTER'].apply(lambda x: np.log(x))
dist = pairwise_distances(df)

ordered_dist, p_order, q_order = compute_ordered_dis_njit_merge(dist, inplace=True)

# Normalize the distance matrix to 0-255 range for greyscale image
normalized = ((ordered_dist - ordered_dist.min()) / (ordered_dist.max() - ordered_dist.min()) * 255).astype(np.uint8)

# Create and save greyscale image using PIL
img = Image.fromarray(normalized, mode='L')
img.save('ordered_distance_matrix.png')
