import time

import numpy as np
from cluster import compute_ordered_dis_njit_merge, vat_prim_mst_seq
from matplotlib import pyplot as plt

from pyclustertend.visual_assessment_of_tendency import compute_ordered_dis_njit
from sklearn.metrics import pairwise_distances
from PIL import Image

from test_combinatorics import circle_random_clusters


def test_cluster_uci_letter_reco():
    print("\n")
    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    # 59 is letter recognition
    # 827 is sepsis survival (allocates 80+ GB RAM)
    # 148 is shuttle stat log (allocates 50 GB RAM)
    letter_recognition = fetch_ucirepo(id=148)

    # data (as pandas dataframes)
    X = letter_recognition.data.features
    y = letter_recognition.data.targets

    # metadata
    print(f"Metadata: {letter_recognition.metadata}")

    # variable information
    print(f"Variable Information: {letter_recognition.variables}")

    # Compute the pairwise distances
    matrix_of_pairwise_distance = np.log(pairwise_distances(X))
    matrix_of_pairwise_distance = (
        matrix_of_pairwise_distance / matrix_of_pairwise_distance.max()
    )
    print(f"Pairwise distance matrix shape: {matrix_of_pairwise_distance.shape}")
    t0 = time.time()
    ordered_matrix = compute_ordered_dis_njit(matrix_of_pairwise_distance)
    # ordered_matrix, path_merge = compute_ordered_dis_njit_merge(
    #     matrix_of_pairwise_distance, inplace=True
    # )
    t1 = time.time()

    print(f"Elapsed time for {len(X)} data points: {t1-t0:.02f}")


    # Save the ordered matrix as an image
    # img_array = (ordered_matrix * 255).astype(np.uint8)
    # img = Image.fromarray(img_array)
    # img.save('ordered_matrix.png')


def test_cluster_sequencing():
    print("\n")
    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    # 59 is letter recognition
    # 827 is sepsis survival (allocates 80+ GB RAM)
    # 148 is shuttle stat log (allocates 50 GB RAM)
    letter_recognition = fetch_ucirepo(id=59)

    # data (as pandas dataframes)
    X = np.array(letter_recognition.data.features)

    # metadata
    print(f"Metadata: {letter_recognition.metadata}")

    # variable information
    print(f"Variable Information: {letter_recognition.variables}")

    # Compute the pairwise distances
    t0 = time.time()
    ordered_matrix = vat_prim_mst_seq(X)
    t1 = time.time()

    print(f"Elapsed time for {len(X)} data points: {t1-t0:.02f}")


def test_vat_scaling():
    city_count: list[int] = []
    merge_time: list[float] = []
    lib_time: list[float] = []
    o1 = 7
    o2 = 11
    n = 2*(o2-o1+1)
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
        # Cluster using our VAT
        t0 = time.time()
        ordered_matrix2, path_merge = compute_ordered_dis_njit_merge(
            matrix_of_pairwise_distance, inplace=False
        )
        t1 = time.time()
        # Cluster using the library VAT
        ordered_matrix = compute_ordered_dis_njit(matrix_of_pairwise_distance.copy())
        # ordered_matrix = 0 * ordered_matrix2
        t2 = time.time()

        # Print the results
        merge_time.append(t1 - t0)
        lib_time.append(t2 - t1)

    # Chop the smallest entry off - that has the njit compilation time baked in.
    city_count = city_count[1:]
    merge_time = merge_time[1:]
    lib_time = lib_time[1:]

    # Prepare data for regression
    city_array = np.array(city_count)
    merge_array = np.array(merge_time)
    lib_array = np.array(lib_time)
    city_count_scl = city_array[:] / city_array[0]
    merge_count_scl = merge_array[:] / merge_array[0]
    lib_count_scl = lib_array[:] / lib_array[0]

    # Plot scaling variance
    plt.figure()
    plt.semilogy(city_count_scl, merge_count_scl, 'o-', label='Merge VAT')
    plt.semilogy(city_count_scl, lib_count_scl, 'o-', label='Lib VAT')
    plt.semilogy(city_count_scl, city_count_scl**2, '-', label='$O(N)=N^2$')
    plt.semilogy(city_count_scl, city_count_scl**2*np.log(city_count_scl+1), '-', label='$O(N)=N^2*log(N)$')
    plt.semilogy(city_count_scl, city_count_scl**3, '-', label='$O(N)=N^3$')
    plt.xlabel("N Scaling")
    plt.ylabel("Time Scaling")
    plt.legend()
    plt.title("VAT Scaling Test")
    plt.show()


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
