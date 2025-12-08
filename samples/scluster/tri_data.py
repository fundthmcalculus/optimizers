import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

from cluster import compute_ordered_dis_njit_merge

input_file = "data_preprocessed1.csv"

df = pd.read_csv(input_file, nrows=10000, dtype=np.float64)
# Take the log of the age column and the really big column.
df.drop(columns=["LENGTH_OF_STAY_MINS"], inplace=True)
# df['AGE_ENCOUNTER'] = df['AGE_ENCOUNTER'].apply(lambda x: np.log(x))
dist = pairwise_distances(df)

ordered_dist, p_order = compute_ordered_dis_njit_merge(dist, inplace=True)

# Create figure with two subplots
fit = plt.figure()
im2 = plt.imshow(ordered_dist)
plt.title('Ordered Distance Matrix')

plt.tight_layout()
plt.show()
