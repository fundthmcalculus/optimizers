import time

import numpy as np
from cluster import compute_ordered_dis_njit_merge
from matplotlib import pyplot as plt
from pyclustertend.visual_assessment_of_tendency import compute_ordered_dis_njit
from sklearn.metrics import pairwise_distances

from test_combinatorics import circle_random_clusters


def test_vat_scaling():
    city_count: list[int] = []
    merge_time: list[float] = []
    lib_time: list[float] = []
    o1 = 3
    o2 = 10
    n = 5
    for group_count in np.logspace(o1, o2, n, base=2, dtype="int"):
        city_count.append(group_count)
        # print(f"City count: {group_count}")
        all_cities = circle_random_clusters(n_clusters=group_count, n_cities=1)
        matrix_of_pairwise_distance = pairwise_distances(all_cities)
        # Scramble the order to ensure we sort it!
        cols = np.arange(len(all_cities), dtype="int")
        rand_col_order = np.random.permutation(cols)
        matrix_of_pairwise_distance = matrix_of_pairwise_distance[:, rand_col_order][
            rand_col_order, :
        ]
        # Cluster using our IVAT
        t0 = time.time()
        ordered_matrix2, path_merge = compute_ordered_dis_njit_merge(
            matrix_of_pairwise_distance
        )
        t1 = time.time()
        # Cluster using the library IVAT
        ordered_matrix = compute_ordered_dis_njit(matrix_of_pairwise_distance)
        # ordered_matrix = 0*ordered_matrix2
        t2 = time.time()

        # Print the results
        merge_time.append(t1 - t0)
        lib_time.append(t2 - t1)

    # Prepare data for regression
    log_x = np.log(city_count)
    log_lib = np.log(lib_time)
    log_merge = np.log(merge_time)

    print("\n")
    print(f"Merge-VAT O(n)=n^{np.round(np.mean(log_merge/log_x))}")
    print(f"Lib-VAT O(n)=n^{np.round(np.mean(log_lib/log_x))}")

    plt.figure()
    plt.loglog(city_count, merge_time, "o", label="Merge VAT")
    plt.loglog(city_count, lib_time, "o", label="Lib VAT")
    plt.xlabel("Number of Cities")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.title("VAT Scaling Test")
    plt.show()

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 10))
    fig.suptitle(f"VAT Scaling Test (N={group_count})")
    im1 = ax1.imshow(ordered_matrix, cmap="viridis")
    ax1.set_title(f"Library VAT Result: {t2-t1:.2f} seconds")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(ordered_matrix2, cmap="viridis")
    ax2.set_title(f"Merge Sort VAT Result: {t1-t0:.2f} seconds")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.show()
