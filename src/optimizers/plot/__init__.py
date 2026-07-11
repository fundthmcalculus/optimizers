import os
from typing import Any

import matplotlib

_TRUTHY = {"1", "true", "yes", "on"}


def _env_no_show() -> bool:
    """Whether the ``OPTIMIZERS_NO_SHOW`` environment variable disables display."""
    return os.environ.get("OPTIMIZERS_NO_SHOW", "").strip().lower() in _TRUTHY


# Select a non-interactive backend *before* pyplot is imported when display is
# disabled, so that importing this module never tries to reach a screen/browser.
# This is what makes the library safe for headless/agentic/CI runs.
if _env_no_show():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from ..core.types import AF, AI  # noqa: E402

# Global runtime toggle. Defaults to the env var, but can be flipped at runtime
# via ``set_show_plots`` (e.g. from a pytest fixture) without touching the env.
_show_plots = not _env_no_show()


def set_show_plots(enabled: bool) -> None:
    """Globally enable or disable interactive display of plots.

    When disabled, plotting functions build and return the figure but do not
    call ``plt.show()`` (the figure is closed to free memory). This is the
    switch tests and agents use to run fully offline.
    """
    global _show_plots
    _show_plots = enabled


def show_plots_enabled() -> bool:
    """Return whether plots are currently displayed when created."""
    return _show_plots


def _finish(fig: Figure) -> Figure:
    """Show the figure if display is enabled, otherwise close it. Returns the figure."""
    if _show_plots:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_convergence(
    tour_lengths: np.ndarray | list[np.ndarray], trace_names: list[str] | None = None
) -> Figure:
    if not isinstance(tour_lengths, list):
        tour_lengths = tour_lengths.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i_t, trace in enumerate(tour_lengths):
        ax.plot(
            np.arange(len(trace)),
            trace,
            marker="o",
            markersize=4,
            linewidth=2,
            label=trace_names[i_t] if trace_names else f"Run-{i_t + 1}",
        )

    ax.set_title("Convergence")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Output Value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    return _finish(fig)


def plot_cities_and_route(
    cities: AF, route: AI | list[AI], trace_names: list[str] | None = None
) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot cities
    ax.scatter(cities[:, 0], cities[:, 1], s=64, c="blue", label="Cities", zorder=3)

    if not isinstance(route, list):
        route = [route]

    # Plot route(s)
    for ir, r in enumerate(route):
        # If 1D, this is a tour; if 2D, this is a route/MST.
        if r.ndim == 1:
            route_cities = np.vstack((cities[r], cities[r[0]]))  # Connect back to start
            ax.plot(
                route_cities[:, 0],
                route_cities[:, 1],
                linewidth=2,
                label=trace_names[ir] if trace_names else f"Route-{ir + 1}",
            )
        elif r.ndim == 2:
            x_route: list[float] = list()
            y_route: list[float] = list()
            for seg in r:
                x_route.extend(cities[seg, 0][:])
                y_route.extend(cities[seg, 1][:])
                # NaN breaks the line between disjoint segments (matplotlib
                # equivalent of plotly's ``None`` gap markers).
                x_route.append(np.nan)
                y_route.append(np.nan)
            ax.plot(
                x_route,
                y_route,
                linewidth=2,
                label=trace_names[ir] if trace_names else f"MST-{ir + 1}",
            )

    ax.set_title("TSP Route")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    fig.tight_layout()

    return _finish(fig)


def plot_run_statistics(
    summary_or_scores: dict[str, Any] | np.ndarray | list[float],
    runtimes: np.ndarray | list[float] | None = None,
    title_prefix: str = "Run Statistics",
) -> Figure:
    """Render box-and-whisker plots for final scores and total runtimes across runs.

    Parameters
    ----------
    summary_or_scores: dict | array-like
        - If dict (from optimizers.checkpoint.run_multiple), expects keys 'scores' and 'runtimes'.
        - If array-like, treated as the list of final scores; then provide `runtimes` separately.
    runtimes: array-like | None
        List of runtimes in seconds (only required when `summary_or_scores` is not a dict).
    title_prefix: str
        Custom prefix for figure titles.
    """
    if isinstance(summary_or_scores, dict):
        scores = np.asarray(summary_or_scores.get("scores", []), dtype=float)
        rtimes = np.asarray(summary_or_scores.get("runtimes", []), dtype=float)
    else:
        scores = np.asarray(summary_or_scores, dtype=float)
        rtimes = np.asarray(runtimes if runtimes is not None else [], dtype=float)

    fig, (ax_scores, ax_rtimes) = plt.subplots(1, 2, figsize=(10, 5))

    # Box for scores
    if scores.size:
        ax_scores.boxplot(scores, showmeans=True, tick_labels=["Final Score"])
    ax_scores.set_ylabel("Final Score")

    # Box for runtimes (independent axis/scale)
    if rtimes.size:
        ax_rtimes.boxplot(rtimes, showmeans=True, tick_labels=["Runtime (s)"])
    ax_rtimes.set_ylabel("Runtime (seconds)")

    fig.suptitle(f"{title_prefix}: Final Score and Runtime")
    fig.tight_layout()

    return _finish(fig)


def plot_pareto_front(
    objectives: AF,
    objective_names: list[str] | None = None,
) -> Figure:
    """Scatter the tracked objectives with the Pareto-non-dominated set highlighted.

    Quality-diversity add-on (QD_PARETO_PLAN.md §4.5). Handles 2 or 3 objectives;
    for more, the first three are shown. Returns the figure.
    """
    from ..archive.metrics import non_dominated_mask

    f = np.atleast_2d(np.asarray(objectives, dtype=float))
    m = f.shape[1]
    mask = non_dominated_mask(f)
    names = objective_names or [f"objective {i + 1}" for i in range(m)]

    if m == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            f[~mask, 0],
            f[~mask, 1],
            s=36,
            c="lightgray",
            label="dominated",
            zorder=2,
        )
        # Order the front by the first objective so the connecting line is monotone.
        front = f[mask][np.argsort(f[mask, 0])]
        ax.plot(
            front[:, 0],
            front[:, 1],
            marker="o",
            markersize=8,
            linewidth=2,
            color="crimson",
            label="Pareto front",
            zorder=3,
        )
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
    else:
        cols = [0, 1, 2] if m >= 3 else [0, 1]
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            f[~mask, cols[0]],
            f[~mask, cols[1]],
            f[~mask, cols[2]],
            s=12,
            c="lightgray",
            label="dominated",
        )
        ax.scatter(
            f[mask, cols[0]],
            f[mask, cols[1]],
            f[mask, cols[2]],
            s=36,
            c="crimson",
            label="Pareto front",
        )
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.set_zlabel(names[2] if len(names) > 2 else "objective 3")

    ax.set_title("Pareto front")
    ax.legend(loc="best")
    fig.tight_layout()

    return _finish(fig)


def plot_map_elites(archive, objective_name: str = "fitness") -> Figure:
    """Scatter a CVT MAP-Elites archive's cells in 2-D, colored by elite fitness.

    Occupied cells are colored by their elite's value (lower=better); empty cells
    are shown faintly. Uses the first two descriptor dimensions. Returns the
    figure.
    """
    centroids, values, occupied = archive.cell_data()
    centroids = np.asarray(centroids, dtype=float)
    values = np.asarray(values, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))

    empty = ~occupied
    if np.any(empty):
        ax.scatter(
            centroids[empty, 0],
            centroids[empty, 1],
            s=25,
            c="lightgray",
            label="empty cell",
            zorder=2,
        )
    if np.any(occupied):
        scatter = ax.scatter(
            centroids[occupied, 0],
            centroids[occupied, 1],
            s=64,
            c=values[occupied],
            cmap="viridis",
            label="elite",
            zorder=3,
        )
        fig.colorbar(scatter, ax=ax, label=objective_name)

    ax.set_title("MAP-Elites archive")
    ax.set_xlabel("descriptor 1")
    ax.set_ylabel("descriptor 2")
    ax.legend(loc="best")
    fig.tight_layout()

    return _finish(fig)
