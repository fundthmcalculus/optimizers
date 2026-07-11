import matplotlib
import numpy as np
from matplotlib.figure import Figure

from optimizers.plot import (
    plot_cities_and_route,
    plot_convergence,
    plot_run_statistics,
    set_show_plots,
    show_plots_enabled,
)


def test_backend_is_headless():
    # conftest forces the non-interactive Agg backend for the suite.
    assert matplotlib.get_backend().lower() == "agg"
    assert show_plots_enabled() is False


def test_plot_convergence_returns_figure_without_showing():
    fig = plot_convergence(np.array([10.0, 8.0, 7.0, 6.5]))
    assert isinstance(fig, Figure)


def test_plot_convergence_multiple_traces():
    fig = plot_convergence(
        [np.array([5.0, 4.0, 3.0]), np.array([9.0, 6.0, 4.0])],
        trace_names=["A", "B"],
    )
    assert isinstance(fig, Figure)


def test_plot_cities_and_route_tour():
    cities = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    route = np.array([0, 1, 2, 3])
    fig = plot_cities_and_route(cities, route)
    assert isinstance(fig, Figure)


def test_plot_cities_and_route_2d_segments():
    cities = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    mst = np.array([[0, 1], [2, 3]])
    fig = plot_cities_and_route(cities, mst)
    assert isinstance(fig, Figure)


def test_plot_run_statistics_from_dict():
    summary = {"scores": [3.0, 2.5, 2.7, 2.6], "runtimes": [1.1, 1.3, 1.2, 1.0]}
    fig = plot_run_statistics(summary)
    assert isinstance(fig, Figure)


def test_plot_run_statistics_empty_is_safe():
    # Empty inputs must not raise (matplotlib boxplot rejects empty arrays).
    fig = plot_run_statistics({"scores": [], "runtimes": []})
    assert isinstance(fig, Figure)


def test_set_show_plots_toggles_flag():
    original = show_plots_enabled()
    try:
        set_show_plots(True)
        assert show_plots_enabled() is True
        set_show_plots(False)
        assert show_plots_enabled() is False
    finally:
        set_show_plots(original)
