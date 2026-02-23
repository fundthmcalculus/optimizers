# Load a dataset for TSP
import time

import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

from cluster import compute_ordered_dis_njit_merge, compute_ivat
from optimizers.combinatorial.aco import AntColonyTSP, AntColonyTSPConfig
from optimizers.combinatorial.strategy import TwoOptTSPConfig, TwoOptTSP

optim = "two-opt" # "aco"
file_name = "burma14" # "att532"
optimal_length = None
with open(f"./{file_name}.txt") as f:
    lines = [l.strip() for l in f.readlines()]

# Look for solution in header
for line in lines[:10]:
    if "SOLUTION" in line:
        optimal_length = int(line.split(":")[1])
        break

# Remove the lines until we hit the first entry, and remove the EOF line
lines = [line for line in lines if line.strip() and "EOF" not in line]
start_idx = [ij for ij, l in enumerate(lines) if l.strip().startswith("1")][0]
lines = lines[start_idx:]
lines = [x.split()[1:] for x in lines]
print(f"Solving TSP for {len(lines)} nodes")

cities = np.array([[float(s) for s in l] for l in lines])
cities_dist = pairwise_distances(cities)

# Compute the VAT order, determine how good a result we get in what time.
print("Solving VAT order for TSP")
t0 = time.time()
d_p_star, d_star, ivat_seq, city_sequence = compute_ivat(cities_dist)
t1 = time.time()

# Since this is a loop, the start location really doesn't matter!
total_distance = 0
for ij, city in enumerate(city_sequence):
    # Initial round will close the loop (-1 -> 0)
    total_distance += cities_dist[city_sequence[ij - 1], city_sequence[ij]]
print(f"VAT order: {city_sequence}")
print(f"Total time: {t1-t0}")
print(f"Total distance: {total_distance:.0f}")


# city_sequence = np.append(city_sequence, city_sequence[0])
start_time = time.time()
if optim == "aco":
    topt_config = AntColonyTSPConfig(
        name="ACO TSP",
        back_to_start=True,
        hot_start=city_sequence,
        hot_start_length=total_distance,
        local_optimize=True,
    )
    topt_optimizer = AntColonyTSP(
        topt_config,
        network_routes=cities_dist,
    )
elif optim == "two-opt":
    topt_config = TwoOptTSPConfig(
        name="ACO TSP",
        back_to_start=True,
    )
    topt_optimizer = TwoOptTSP(
        topt_config,
        network_routes=cities_dist,
        initial_route=city_sequence,
        initial_value=total_distance,
    )
else:
    raise ValueError("optim must be aco or two-opt")

topt_result = topt_optimizer.solve()
topt_time = time.time() - start_time
# TODO - Try the actual MST!


opt_city_sequence = topt_result.optimal_path
print(f"ACO-TSP time: {topt_time:.02f}, distance={topt_result.optimal_value:.0f}")
print(f"ACO-TSP order: {opt_city_sequence}")

# Plot the convergence
# plt.figure(figsize=(10, 5))
# plt.plot(topt_result.value_history)
# plt.title("ACO Convergence History")
# plt.xlabel("Generation")
# plt.ylabel("Tour Length")
# plt.grid(True)
# plt.show()

# Plot the cities and route
plt.figure(figsize=(10, 10))
x_coords = cities[:, 0]
y_coords = cities[:, 1]

# Plot all cities
plt.scatter(x_coords, y_coords, c="red", s=50)


def get_plot_seq(city_seq):
    _plot_x = []
    _plot_y = []
    for ij in range(len(city_seq)):
        start = city_seq[ij - 1]
        end = city_seq[ij]
        _plot_x.append(x_coords[start])
        _plot_x.append(x_coords[end])
        _plot_y.append(y_coords[start])
        _plot_y.append(y_coords[end])
    return _plot_x, _plot_y


# Plot the route
plot_x, plot_y = get_plot_seq(city_sequence)
plt.plot(plot_x, plot_y, "b-", label="VAT-TSP")
plot_x, plot_y = get_plot_seq(opt_city_sequence)
plt.plot(plot_x, plot_y, "r-", label="Optimized-TSP")

plt.title(
    f"TSP Route through {file_name} Cities:\n"
    f"$L_{{VAT}}$={total_distance:.0f}\n"
    f"$L_{{2-OPT}}$={topt_result.optimal_value:.0f}\n"
    f"$L_{{optimal}}$={optimal_length}\n"
    f"$T_{{VAT}}$={t1-t0:.02f}s\n"
    f"$T_{{2-OPT}}$={topt_time:.02f}s"
)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.tight_layout()
plt.legend()
# plt.savefig('tsp_route.eps', format='eps')
# plt.close()

# Show both the VAT image and the ACO-TSP optimized image.
fig = plt.figure(figsize=(8, 8))
fig.subplots(2, 2)
plt.subplot(2, 2, 1)
plt.imshow(d_star, cmap="viridis")
plt.colorbar()
plt.title(f"VAT-{file_name}")
plt.subplot(2, 2, 2)
plt.imshow(d_star[:, opt_city_sequence][opt_city_sequence, :], cmap="viridis")
plt.colorbar()
plt.title(f"Optimized VAT-{file_name}")
plt.subplot(2, 2, 3)
plt.imshow(d_p_star, cmap="viridis")
plt.colorbar()
plt.title(f"IVAT-{file_name}")
plt.subplot(2, 2, 4)
plt.imshow(d_p_star[:, opt_city_sequence][opt_city_sequence, :], cmap="viridis")
plt.colorbar()
plt.title(f"Optimized IVAT-{file_name}")
# plt.savefig('vat_comparison2.eps', format='eps')
# plt.close()
plt.show()
