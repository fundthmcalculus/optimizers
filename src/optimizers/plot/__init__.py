import plotly.graph_objects as go


def plot_convergence(tour_lengths):
    # Create the figure
    fig = go.Figure()

    # Add the line trace
    fig.add_trace(
        go.Scatter(
            x=np.r_[0 : len(tour_lengths)],
            y=tour_lengths,
            mode="lines+markers",
            name="Tour Length",
            line=dict(color="royalblue", width=2),
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
