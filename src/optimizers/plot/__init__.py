import numpy as np
import plotly.graph_objects as go


def plot_convergence(tour_lengths: np.ndarray):
    # Create the figure
    fig = go.Figure()

    if len(tour_lengths.shape) == 1:
        tour_lengths = tour_lengths.reshape(1, -1)

    # Add the line trace
    for trace in range(tour_lengths.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=np.r_[0 : tour_lengths.shape[-1]],
                y=tour_lengths[trace, :],
                mode="lines+markers",
                name=f"Tour Length-{trace+1}",
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
