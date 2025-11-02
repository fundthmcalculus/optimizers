from typing import List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..core.types import AF, AI


def plot_convergence(
    tour_lengths: np.ndarray | list[np.ndarray], trace_names: list[str] | None = None
):
    # Create the figure
    fig = go.Figure()

    if not isinstance(tour_lengths, list):
        tour_lengths = tour_lengths.reshape(1, -1)

    # Add the line trace
    for i_t, trace in enumerate(tour_lengths):
        fig.add_trace(
            go.Scatter(
                x=np.r_[0 : len(trace)],
                y=trace,
                mode="lines+markers",
                name=trace_names[i_t] if trace_names else f"Tour Length-{i_t+1}",
                line=dict(width=2),
                marker=dict(size=6),
            )
        )

    # Update layout
    fig.update_layout(
        title="ACO Convergence",
        xaxis_title="Generation",
        yaxis_title="Tour Length",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Show the figure
    fig.show()


def plot_cities_and_route(
    cities: AF, route: AI | List[AI], trace_names: list[str] | None = None
):
    fig = go.Figure()

    # Plot cities
    fig.add_trace(
        go.Scatter(
            x=cities[:, 0],
            y=cities[:, 1],
            mode="markers",
            name="Cities",
            marker=dict(size=8, color="blue"),
        )
    )

    if not isinstance(route, list):
        route = [route]

    # Plot route
    for ir, route in enumerate(route):
        route_cities = np.vstack(
            (cities[route], cities[route[0]])
        )  # Connect back to start
        fig.add_trace(
            go.Scatter(
                x=route_cities[:, 0],
                y=route_cities[:, 1],
                mode="lines",
                name=trace_names[ir] if trace_names else f"Route-{ir+1}",
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title="TSP Route",
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=True,
        template="plotly_white",
    )

    fig.show()


def plot_run_statistics(
    summary_or_scores,
    runtimes: np.ndarray | list[float] | None = None,
    title_prefix: str = "Run Statistics",
):
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

    # Create two subplots vertically using specifications
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "box"}, {"type": "box"}]])

    # Box for Scores
    fig.add_trace(
        go.Box(
            y=scores,
            name="Final Score",
            boxmean=True,
            marker_color="#1f77b4",
        ),
        row=1,
        col=1,
    )

    # Create a second independent y-axis for runtimes by adding as another trace with different x name
    fig.add_trace(
        go.Box(
            y=rtimes,
            name="Runtime (s)",
            boxmean=True,
            marker_color="#ff7f0e",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"{title_prefix}: Final Score and Runtime",
        template="plotly_white",
        showlegend=False,
        yaxis_title="Final Score",
        yaxis2_title="Runtime (seconds)",
    )

    fig.show()
