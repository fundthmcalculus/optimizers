"""Wall-clock vs. population-size scaling check for ACO/GA/PSO.

Complements ``run_benchmark.py`` (which fixes population/generations and
sweeps eval mode + seeds): this sweeps population size instead, at a fixed
eval mode, to catch complexity-class regressions (an O(n^2) hot path hiding
behind a small default benchmark) rather than just constant-factor ones. See
PERF_CONTINUOUS_REPORT.md §6 for the investigation this script grew out of
(ACO's sampling was O(population x archive_size) per variable; fixed to
O(log archive_size)).

Usage::

    python benchmarks/run_scaling_benchmark.py [--seeds N] [--quick]

Writes ``benchmarks/results/scaling_timings.{csv,png}`` (log-log wall-clock
vs. population, mean +/- stdev across seeds). A roughly straight line on the
log-log plot is the signature of polynomial scaling; a line that visibly
steepens as population grows is worth profiling.
"""

import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("OPTIMIZERS_NO_SHOW", "1")

from optimizers.benchmarks.functions import ackley, ackley_batch  # noqa: E402
from optimizers.benchmarks.harness import OPTIMIZERS  # noqa: E402
from optimizers.continuous.variables import InputContinuousVariable  # noqa: E402
from optimizers.core.random import set_seed  # noqa: E402


def _time_one(optimizer_name, population_size, n_dim, num_generations, seed):
    optimizer_cls, config_cls = OPTIMIZERS[optimizer_name]
    set_seed(seed)
    variables = [InputContinuousVariable(f"x{i}", -5.0, 5.0) for i in range(n_dim)]
    config = config_cls(
        name=f"{optimizer_name}-scaling-{population_size}-{seed}",
        population_size=population_size,
        num_generations=num_generations,
        solution_archive_size=population_size * 3,
        n_jobs=1,
        joblib_prefer="threads",
        local_grad_optim="none",
        stop_after_iterations=10_000,  # disable early stopping for consistent timing
    )
    optimizer = optimizer_cls(
        config=config, variables=variables, fcn=ackley, batch_fcn=ackley_batch
    )
    start = time.perf_counter()
    optimizer.solve()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--populations", type=int, nargs="+", default=[100, 250, 500, 1000, 2000]
    )
    parser.add_argument("--n-dim", type=int, default=20)
    parser.add_argument("--num-generations", type=int, default=20)
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=list(OPTIMIZERS),
        default=list(OPTIMIZERS),
        help="Subset of optimizers to run (default: all). Useful for a "
        "targeted large-population trial of just one solver.",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Small sweep for a fast sanity check."
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "results"),
    )
    args = parser.parse_args()

    populations = [100, 500] if args.quick else args.populations
    seeds = range(2 if args.quick else args.seeds)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for optimizer_name in args.optimizers:
        for population_size in populations:
            times = [
                _time_one(
                    optimizer_name, population_size, args.n_dim, args.num_generations, s
                )
                for s in seeds
            ]
            mean_t = statistics.mean(times)
            std_t = statistics.stdev(times) if len(times) > 1 else 0.0
            rows.append((optimizer_name, population_size, mean_t, std_t))
            print(
                f"{optimizer_name:<5} pop={population_size:5d}  mean={mean_t:8.4f}s  stdev={std_t:8.4f}s"
            )

    csv_path = os.path.join(args.out_dir, "scaling_timings.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["optimizer", "population", "mean_s", "stdev_s"])
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for optimizer_name in args.optimizers:
        opt_rows = [r for r in rows if r[0] == optimizer_name]
        pops = [r[1] for r in opt_rows]
        means = [r[2] for r in opt_rows]
        stds = [r[3] for r in opt_rows]
        ax.errorbar(pops, means, yerr=stds, marker="o", capsize=4, label=optimizer_name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("population size (archive = 3x population)")
    ax.set_ylabel("wall-clock time (s)")
    ax.set_title("GA / ACO / PSO wall-clock vs. population size (log-log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, "scaling_timings.png")
    fig.savefig(png_path, dpi=150)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
