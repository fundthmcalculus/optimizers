import numpy as np
import plotly.graph_objects as go


def plot_convergence(tour_lengths: np.ndarray | list[np.ndarray]):
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
                name=f"Tour Length-{i_t+1}",
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


def plot_run_statistics(summary_or_scores, runtimes: np.ndarray | list[float] | None = None, title_prefix: str = "Run Statistics"):
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
    fig = go.Figure()

    # Box for Scores
    fig.add_trace(
        go.Box(
            y=scores,
            name="Final Score",
            boxmean=True,
            marker_color="#1f77b4",
        )
    )

    # Create a second independent y-axis for runtimes by adding as another trace with different x name
    fig.add_trace(
        go.Box(
            y=rtimes,
            name="Runtime (s)",
            boxmean=True,
            marker_color="#ff7f0e",
        )
    )

    fig.update_layout(
        title=f"{title_prefix}: Final Score and Runtime",
        yaxis_title="Value",
        template="plotly_white",
        showlegend=False,
    )

    fig.show()
