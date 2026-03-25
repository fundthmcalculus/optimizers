from numba import prange, njit, set_num_threads
import numpy as np
import pandas as pd
import os
from PIL import Image
from numba_progress import ProgressBar

from cluster import compute_ordered_dis_njit_merge

input_file = "data_preprocessed1.csv"
default_dtype = np.float32


@njit(cache=True, parallel=True, nogil=True)
def pairwise_distances2(data, progress_proxy, dtype=default_dtype) -> np.ndarray:
    dist_arr = np.zeros((data.shape[0], data.shape[0]), dtype=dtype)
    for i in prange(len(data)):
        dist_arr[i, i + 1 :] = np.linalg.norm(data[i, :] - data[i + 1 :, :])
        dist_arr[i + 1 :, i] = dist_arr[i, i + 1 :]
        progress_proxy.update(1)
    return dist_arr


def main():
    set_num_threads(int((os.cpu_count() or 1.2) * 0.9))
    # Load the first 10 rows to identify column locations
    df_sample = pd.read_csv(input_file, nrows=10, dtype=default_dtype)
    print("Column names and their positions:")
    for idx, col in enumerate(df_sample.columns):
        print(f"  {idx}: {col}")

    los_idx = age_idx = 0

    if not os.path.exists("distance_matrix-uint8.npy"):
        # Identify specific column locations
        if "LENGTH_OF_STAY_MINS" in df_sample.columns:
            los_idx = df_sample.columns.get_loc("LENGTH_OF_STAY_MINS")

        if "AGE_ENCOUNTER" in df_sample.columns:
            age_idx = df_sample.columns.get_loc("AGE_ENCOUNTER")

        # Check if distance matrix already exists
        if os.path.exists("distance_matrix.npy"):
            print("Loading existing distance matrix...")
            dist = np.load("distance_matrix.npy")
        else:
            print("Computing distance matrix...")
            data = np.loadtxt(
                input_file, dtype=default_dtype, skiprows=1, delimiter=","
            )
            # Take the log of the age column and zero the really big column.
            data[:, los_idx] = 0
            # data[:, age_idx] = 0 # np.log10(data[:, age_idx])
            with ProgressBar(total=len(data)) as progress:
                dist = pairwise_distances2(data, progress)
            np.save("distance_matrix.npy", dist)
            del data

        with ProgressBar(total=len(dist)) as progress:
            dist, p_order, q_order = compute_ordered_dis_njit_merge(
                dist, inplace=True, progress_bar=progress
            )

        # Normalize the distance matrix to 0-255 range for greyscale image
        min_val = dist.min()
        max_val = dist.max()
        dist -= min_val
        dist *= 255 / (max_val - min_val)
        dist = dist.astype(np.uint8)
        np.save("distance_matrix-uint8.npy", dist)
        ordered_dist = dist
    else:
        ordered_dist = np.load("distance_matrix-uint8.npy")

    print("Performing VAT/IVAT...")

    np.save("vat_col_sequence.npy", p_order)
    np.save("ivat_col_sequence.npy", q_order)

    print(
        f"VAT/IVAT completed. Saved column sequences to vat_col_sequence.npy and ivat_col_sequence.npy"
    )

    # Create and save greyscale image using PIL
    img = Image.fromarray(ordered_dist, mode="L")
    img.save("ordered_distance_matrix.png")

    print("Done.")


if __name__ == "__main__":
    main()
