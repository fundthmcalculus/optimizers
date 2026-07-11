"""Phase-3 quality-diversity add-on: metrics, run report, plots (QD_PARETO_PLAN.md §4.5)."""

import numpy as np

from optimizers.archive import (
    non_dominated_mask,
    pareto_front,
    hypervolume,
    qd_score,
    QDReport,
)
from optimizers.plot import plot_pareto_front, plot_map_elites
from optimizers.continuous.ga import (
    GeneticAlgorithmOptimizer,
    GeneticAlgorithmOptimizerConfig,
)
from optimizers.continuous.variables import InputContinuousVariable
from optimizers.core.random import set_seed


def _vars(n=6):
    return [
        InputContinuousVariable(f"x{i}", lower_bound=-3.0, upper_bound=3.0)
        for i in range(n)
    ]


def _biobj(x):
    x = np.asarray(x)
    f1 = float(np.sum((x - 1.0) ** 2))
    f2 = float(np.sum((x + 1.0) ** 2))
    return f1, np.array([f1, f2])  # (fitness, [obj1, obj2])


def _sphere(x):
    return float(np.sum((np.asarray(x) - 1.0) ** 2))


def test_non_dominated_mask_minimization():
    f = np.array([[1, 4], [2, 2], [4, 1], [3, 3], [5, 5]], dtype=float)
    assert non_dominated_mask(f).tolist() == [True, True, True, False, False]


def test_non_dominated_all_when_tradeoff():
    f = np.array([[0, 3], [1, 1], [3, 0]], dtype=float)
    assert non_dominated_mask(f).all()


def test_pareto_front_indices_sorted_by_first_objective():
    f = np.array([[4, 1], [1, 4], [2, 2], [3, 3]], dtype=float)
    front = pareto_front(f)
    assert np.array_equal(front, np.array([1, 2, 0]))  # (1,4),(2,2),(4,1)


def test_hypervolume_2d_exact():
    f = np.array([[1, 4], [2, 2], [4, 1]], dtype=float)
    assert np.isclose(hypervolume(f, np.array([6.0, 6.0])), 20.0)


def test_hypervolume_1d_and_empty():
    assert np.isclose(hypervolume(np.array([[2.0], [5.0]]), np.array([10.0])), 8.0)
    assert hypervolume(np.empty((0, 2)), np.array([1.0, 1.0])) == 0.0


def test_hypervolume_dominated_point_ignored():
    # adding a dominated point must not change the hypervolume
    a = np.array([[1, 4], [2, 2], [4, 1]], dtype=float)
    b = np.vstack([a, [3.0, 3.0]])  # dominated
    ref = np.array([6.0, 6.0])
    assert np.isclose(hypervolume(a, ref), hypervolume(b, ref))


def test_qd_score():
    assert np.isclose(qd_score(np.array([1.0, 2.0, 5.0])), 7.0)  # ref=5: 4+3+0
    assert qd_score(np.array([])) == 0.0


def test_qd_report_map_elites_no_objectives():
    set_seed(6)
    cfg = GeneticAlgorithmOptimizerConfig(
        name="me",
        num_generations=12,
        population_size=40,
        n_jobs=2,
        joblib_prefer="threads",
        objective_mode="map-elites",
        descriptor_dim=2,
        archive_cells=32,
        stop_after_iterations=999,
    )
    opt = GeneticAlgorithmOptimizer(cfg, _sphere, _vars(), args={})
    opt.solve()
    rep = opt.qd_report()
    assert isinstance(rep, QDReport)
    assert rep.num_elites > 0
    assert rep.coverage is not None and rep.coverage > 0.2
    assert rep.qd_score >= 0.0
    assert rep.pareto_objectives is None  # no objectives tracked
    assert rep.hypervolume is None


def test_qd_report_with_tracked_objectives_has_pareto_front():
    set_seed(4)
    cfg = GeneticAlgorithmOptimizerConfig(
        name="me",
        num_generations=18,
        population_size=50,
        n_jobs=3,
        joblib_prefer="threads",
        objective_mode="map-elites",
        descriptor_dim=2,
        archive_cells=40,
        n_outputs=2,
        stop_after_iterations=999,
    )
    opt = GeneticAlgorithmOptimizer(cfg, _biobj, _vars(8), args={})
    opt.solve()
    rep = opt.qd_report()
    assert rep.all_objectives is not None
    assert rep.all_objectives.shape[1] == 2
    assert rep.pareto_objectives is not None and len(rep.pareto_objectives) >= 1
    assert rep.pareto_solutions.shape[0] == rep.pareto_objectives.shape[0]
    assert rep.hypervolume is not None and rep.hypervolume > 0.0
    # the reported front is genuinely non-dominated
    assert non_dominated_mask(rep.pareto_objectives).all()


def test_qd_plots_build():
    objs = np.array([[1, 4], [2, 2], [4, 1], [3, 3]], dtype=float)
    fig = plot_pareto_front(objs, ["cost", "risk"])
    # matplotlib Figure: the pareto plot draws a dominated-points scatter
    # (collection) and a front line, so the axes carries drawn artists.
    ax = fig.axes[0]
    assert ax.collections or ax.lines

    set_seed(6)
    cfg = GeneticAlgorithmOptimizerConfig(
        name="me",
        num_generations=8,
        population_size=30,
        n_jobs=2,
        joblib_prefer="threads",
        objective_mode="map-elites",
        descriptor_dim=2,
        archive_cells=24,
        stop_after_iterations=999,
    )
    opt = GeneticAlgorithmOptimizer(cfg, _sphere, _vars(), args={})
    opt.solve()
    fig2 = plot_map_elites(opt.soln_deck)
    assert fig2.axes and fig2.axes[0].collections
