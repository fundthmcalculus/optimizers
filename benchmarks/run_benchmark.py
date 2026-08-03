"""Standalone benchmark runner for the GA/ACO/PSO continuous solvers.

Usage::

    python benchmarks/run_benchmark.py [--seeds N] [--quick]

Runs every (optimizer x test-function x eval-mode x seed) combination via
``optimizers.benchmarks.harness``, prints a summary table, writes the raw
results to ``benchmarks/results/timings.csv`` and a grouped bar chart with
error bars to ``benchmarks/results/timings.png``.

``local_grad_optim`` is fixed at ``"none"`` throughout -- this script is
scoped to the population-based continuous solvers themselves. See
PERF_CONTINUOUS_REPORT.md for the write-up this plot supports.
"""

import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("OPTIMIZERS_NO_SHOW", "1")

from optimizers.benchmarks import BenchmarkSpec, run_benchmark_grid  # noqa: E402
from optimizers.plot import plot_benchmark_timings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, default=8, help="Number of seeds per cell."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small population/generations for a fast sanity-check run.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory to write timings.csv / timings.png into.",
    )
    args = parser.parse_args()

    spec = BenchmarkSpec(seeds=tuple(range(args.seeds)))
    if args.quick:
        spec.population_size = 20
        spec.num_generations = 15
        spec.solution_archive_size = 60

    print(
        f"Running {len(spec.optimizers)} optimizers x {len(spec.functions)} functions x "
        f"{len(spec.modes)} eval-modes x {len(spec.seeds)} seeds = "
        f"{len(spec.optimizers) * len(spec.functions) * len(spec.modes) * len(spec.seeds)} runs..."
    )
    results = run_benchmark_grid(spec)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "timings.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["optimizer", "function", "mode", "seed", "wall_time", "best_value"]
        )
        for r in results:
            writer.writerow(
                [r.optimizer, r.function, r.mode, r.seed, r.wall_time, r.best_value]
            )
    print(f"Wrote {csv_path}")

    # Summary table: mean +/- stdev wall time per (function, optimizer, mode).
    print(
        f"\n{'function':<12}{'optimizer':<8}{'mode':<8}{'mean(s)':>10}{'stdev(s)':>10}{'speedup':>10}"
    )
    functions = sorted({r.function for r in results})
    optimizers = sorted({r.optimizer for r in results})
    # "scalar" first so its mean is available as the speedup baseline when
    # "batch" is printed.
    all_modes = {r.mode for r in results}
    modes = [m for m in ("scalar", "batch") if m in all_modes]
    for function_name in functions:
        for optimizer_name in optimizers:
            scalar_mean = None
            for mode in modes:
                times = [
                    r.wall_time
                    for r in results
                    if r.function == function_name
                    and r.optimizer == optimizer_name
                    and r.mode == mode
                ]
                if not times:
                    continue
                mean_t = statistics.mean(times)
                std_t = statistics.stdev(times) if len(times) > 1 else 0.0
                if mode == "scalar":
                    scalar_mean = mean_t
                speedup = (scalar_mean / mean_t) if scalar_mean else float("nan")
                print(
                    f"{function_name:<12}{optimizer_name:<8}{mode:<8}"
                    f"{mean_t:>10.4f}{std_t:>10.4f}{speedup:>10.2f}x"
                )

    png_path = os.path.join(args.out_dir, "timings.png")
    fig = plot_benchmark_timings(results)
    fig.savefig(png_path, dpi=150)
    print(f"\nWrote {png_path}")


if __name__ == "__main__":
    main()
