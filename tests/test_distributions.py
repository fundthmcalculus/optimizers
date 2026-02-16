from typing import Callable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optimizers.solution_deck import spiral_points


# TODO - Demo function to approximate using fuzzy
def griewank_fcn(x):
    p_i = [np.cos(x[:, ij] / np.sqrt(ij + 1)) for ij in range(x.shape[1])]
    return 1 + 1.0 / 4000.0 * np.sum(np.square(x), axis=1) - np.prod(p_i, axis=0)


def plot_fcn(fcn: Callable, x_domain, n=2) -> None:
    x_s = np.linspace(x_domain[0], x_domain[1], 100)
    x = [x_s for ij in range(n)]
    x_grid = np.meshgrid(*x)
    x_flat = np.stack([grid.flatten() for grid in x_grid], axis=1)
    z = fcn(x_flat)
    z = z.reshape(x_grid[0].shape)

    fig = go.Figure(data=[go.Surface(z=z)])
    fig.show()


def get_rule_idx(s: int, n_base: int, n_dim: int) -> np.ndarray:
    idxes_rev = list()
    for idim in range(n_dim):
        idxes_rev.append(np.mod(s, n_base))
        s //= n_base
    idxes_rev.reverse()
    # NOTE - It really doesn't matter that this is backwards from logic.
    return np.array(idxes_rev)


def test_membership_fcn_distribution():
    n_rules = 20
    n_mu = 5
    n_vars = 4
    l_norm = 0
    max_rules = n_mu ** n_vars
    print(f"\nmax_rules: {max_rules}")
    print(f"n_rules: {n_rules}  ({(n_rules/max_rules):.2%} coverage)")
    print(f"n_mu: {n_mu}")
    print(f"n_vars: {n_vars}")
    # I need to distribute
    mu_selects = spiral_points(n_rules, n_vars, r_scale=1.0)
    # Now map those points to the natural numbers interval
    mu_selects = np.int32(np.round(mu_selects * (n_mu - 1), 0))
    all_norms = get_selected_rule_dists(mu_selects, l_norm)
    possible_norms = get_all_rule_dists(max_rules, n_mu, n_vars, l_norm)

    # Convert sampled rules to linear indices in the full rule space
    sampled_linear_indices = []
    for rule_idx in range(n_rules):
        linear_idx = 0
        for dim in range(n_vars):
            linear_idx = linear_idx * n_mu + mu_selects[rule_idx, dim]
        sampled_linear_indices.append(linear_idx)
    sampled_linear_indices = np.array(sampled_linear_indices)

    # Plot the norms as an image
    # Create subplots using plotly
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'L-{l_norm} Norms Between {n_rules}-Sampled Rules',
                        f'L-{l_norm} Norms Between All Rules'),
        vertical_spacing=0.15
    )

    # First heatmap - sampled rules norms
    fig.add_trace(
        go.Heatmap(
            z=all_norms,
            colorscale='Viridis',
            colorbar=dict(title=f'L-{l_norm} Norm', y=0.75, len=0.4),
            showscale=True
        ),
        row=1, col=1
    )

    # Second heatmap - all possible rules norms
    fig.add_trace(
        go.Heatmap(
            z=possible_norms,
            colorscale='Viridis',
            colorbar=dict(title='L-1 Norm', y=0.25, len=0.4),
            showscale=True
        ),
        row=2, col=1
    )

    # Add scatter markers for sampled rules on second plot
    fig.add_trace(
        go.Scatter(
            x=sampled_linear_indices,
            y=sampled_linear_indices,
            mode='markers',
            marker=dict(color='red', size=8, symbol='circle'),
            name='Sampled Rules',
            showlegend=True
        ),
        row=2, col=1
    )

    # Add vertical and horizontal lines for sampled rules
    for idx in sampled_linear_indices:
        # Vertical line
        fig.add_trace(
            go.Scatter(
                x=[idx, idx],
                y=[0, max_rules - 1],
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        # Horizontal line
        fig.add_trace(
            go.Scatter(
                x=[0, max_rules - 1],
                y=[idx, idx],
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_xaxes(title_text='Rule Index', row=2, col=1)
    fig.update_yaxes(title_text='Rule Index', row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)

    fig.update_layout(
        height=1000,
        width=600,
        showlegend=True
    )

    fig.show()

    # Report statistics (excluding diagonal which is always 0)
    non_diag_norms = all_norms[np.triu_indices(n_rules, k=1)]
    print(f"Min norm: {np.min(non_diag_norms):.2f}")
    print(f"Median norm: {np.median(non_diag_norms):.2f}")
    print(f"Mean norm: {np.mean(non_diag_norms):.2f}")
    print(f"Max norm: {np.max(non_diag_norms):.2f}")


def get_selected_rule_dists(mu_selects, l_norm: int = 1):
    # Vectorized L-norm computation using broadcasting
    diff = mu_selects[:, np.newaxis, :] - mu_selects[np.newaxis, :, :]
    if l_norm == 0:
        all_norms = np.count_nonzero(np.abs(diff), axis=-1)
    else:
        all_norms = np.pow(np.sum(np.abs(diff) ** l_norm, axis=-1), 1/l_norm).round(0)
    return all_norms.astype(np.int32)


def get_all_rule_dists(max_rules, n_mu, n_vars, l_norm: int = 1):
    # Do the entire domain of possible rules
    possible_norms = np.zeros((max_rules, max_rules), dtype=np.int32)
    for ij in range(max_rules):
        idx_ij = get_rule_idx(ij, n_mu, n_vars)
        for jk in range(ij, max_rules):
            idx_jk = get_rule_idx(jk, n_mu, n_vars)
            if l_norm > 0:
                possible_norms[ij, jk] = np.pow(np.sum(np.abs(idx_ij - idx_jk) ** l_norm), 1/l_norm).round(0).astype(np.int32)
            elif l_norm == 0:
                possible_norms[ij, jk] = np.count_nonzero(idx_ij - idx_jk, axis=-1).astype(np.int32)
            possible_norms[jk, ij] = possible_norms[ij, jk]
    return possible_norms
