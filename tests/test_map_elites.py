"""Phase-2 quality-diversity add-on: CVT MAP-Elites + Iso+LineDD (QD_PARETO_PLAN.md §4).

Covers the archive, the projection descriptor, the variation operator, and the
end-to-end map-elites mode on the solvers. The default scalar path must stay
untouched.
"""

import numpy as np

from optimizers.archive import (
    CVTArchive,
    RandomProjectionDescriptor,
    iso_line_dd,
)
from optimizers.solution_deck import SolutionDeck
from optimizers.continuous.ga import (
    GeneticAlgorithmOptimizer,
    GeneticAlgorithmOptimizerConfig,
)
from optimizers.continuous.aco import AntColonyOptimizer, AntColonyOptimizerConfig
from optimizers.continuous.variables import InputContinuousVariable
from optimizers.core.random import set_seed


def _vars(n=6):
    return [
        InputContinuousVariable(f"x{i}", lower_bound=-5.0, upper_bound=5.0)
        for i in range(n)
    ]


def _sphere(x):
    return float(np.sum((np.asarray(x) - 1.0) ** 2))


def _make_archive(num_vars=6, descriptor_dim=2, n_cells=16, seed=0):
    lower = np.full(num_vars, -5.0)
    upper = np.full(num_vars, 5.0)
    desc = RandomProjectionDescriptor(num_vars, descriptor_dim, lower, upper, seed=seed)
    return CVTArchive(
        num_vars=num_vars,
        lower=lower,
        upper=upper,
        descriptor_fn=desc,
        descriptor_dim=descriptor_dim,
        n_cells=n_cells,
        centroid_samples=500,
        seed=seed,
    )


def test_projection_descriptor_is_deterministic_and_shaped():
    lower, upper = np.full(4, -1.0), np.full(4, 1.0)
    d1 = RandomProjectionDescriptor(4, 3, lower, upper, seed=7)
    d2 = RandomProjectionDescriptor(4, 3, lower, upper, seed=7)
    x = np.random.RandomState(0).uniform(-1, 1, size=(5, 4))
    assert np.allclose(d1(x), d2(x))  # same seed → same projection
    assert d1(x).shape == (5, 3)


def test_iso_line_dd_shape_and_bounds():
    rng = np.random.default_rng(0)
    lower, upper = np.full(4, -2.0), np.full(4, 2.0)
    a = rng.uniform(-2, 2, size=(10, 4))
    b = rng.uniform(-2, 2, size=(10, 4))
    kids = iso_line_dd(a, b, 0.1, 0.3, lower, upper, rng)
    assert kids.shape == (10, 4)
    assert (kids >= lower).all() and (kids <= upper).all()


def test_cvt_archive_insertion_and_surface():
    arc = _make_archive()
    rng = np.random.default_rng(1)
    sols = rng.uniform(-5, 5, size=(200, 6))
    vals = np.sum((sols - 1.0) ** 2, axis=1)
    arc.add_generation(sols, vals)
    assert 0 < len(arc) <= arc.n_cells
    assert 0.0 < arc.coverage <= 1.0
    # scalar surface: values ascending (best-first), get_best is the minimum
    assert np.all(np.diff(arc.solution_value) >= 0)
    best_x, best_v, _ = arc.get_best()
    assert np.isclose(best_v, arc.solution_value.min())
    assert best_v <= vals.min() + 1e-9


def test_cvt_keeps_best_per_cell():
    arc = _make_archive(n_cells=8)
    # two solutions guaranteed same cell (identical descriptor): keep the better
    x = np.zeros((1, 6))
    arc.add_generation(np.repeat(x, 2, axis=0), np.array([5.0, 2.0]))
    assert len(arc) == 1
    assert np.isclose(arc.solution_value[0], 2.0)
    arc.add_generation(x, np.array([9.0]))  # worse → rejected
    assert np.isclose(arc.solution_value[0], 2.0)
    arc.add_generation(x, np.array([0.5]))  # better → replaces
    assert np.isclose(arc.solution_value[0], 0.5)


def test_cvt_parents_are_from_occupied_cells():
    arc = _make_archive()
    rng = np.random.default_rng(2)
    sols = rng.uniform(-5, 5, size=(100, 6))
    arc.add_generation(sols, np.sum(sols**2, axis=1))
    p = arc.parents(20, rng)
    assert p.shape == (20, 6)
    # every returned parent must be one of the stored cell elites
    occupied = arc.solution_archive
    for row in p:
        assert np.any(np.all(np.isclose(occupied, row), axis=1))


def test_cvt_checkpoint_roundtrip():
    arc = _make_archive()
    rng = np.random.default_rng(3)
    sols = rng.uniform(-5, 5, size=(120, 6))
    arc.add_generation(sols, np.sum((sols - 1.0) ** 2, axis=1))
    restored = CVTArchive.from_dict(arc.to_dict())
    assert len(restored) == len(arc)
    assert np.allclose(restored.get_best()[1], arc.get_best()[1])
    assert np.allclose(restored.centroids, arc.centroids)


def test_ga_map_elites_end_to_end_builds_cvt_and_covers():
    set_seed(5)
    cfg = GeneticAlgorithmOptimizerConfig(
        name="me",
        num_generations=15,
        population_size=40,
        n_jobs=2,
        joblib_prefer="threads",
        objective_mode="map-elites",
        descriptor_dim=2,
        archive_cells=32,
        stop_after_iterations=999,
    )
    opt = GeneticAlgorithmOptimizer(cfg, _sphere, _vars(), args={})
    res = opt.solve()
    assert isinstance(opt.soln_deck, CVTArchive)
    assert opt.soln_deck.coverage > 0.2
    assert np.isfinite(res.solution_score)
    # scalar consumers still see a best-first ranked deck
    assert np.all(np.diff(opt.soln_deck.solution_value) >= 0)


def test_ga_map_elites_iso_line_variation_runs():
    set_seed(5)
    cfg = GeneticAlgorithmOptimizerConfig(
        name="me",
        num_generations=15,
        population_size=40,
        n_jobs=2,
        joblib_prefer="threads",
        objective_mode="map-elites",
        descriptor_dim=2,
        archive_cells=32,
        qd_variation="iso_line",
        stop_after_iterations=999,
    )
    opt = GeneticAlgorithmOptimizer(cfg, _sphere, _vars(), args={})
    opt.solve()
    assert opt.soln_deck.coverage > 0.2


def test_aco_map_elites_end_to_end():
    set_seed(5)
    cfg = AntColonyOptimizerConfig(
        name="me",
        num_generations=10,
        population_size=40,
        n_jobs=2,
        joblib_prefer="threads",
        objective_mode="map-elites",
        descriptor_dim=2,
        archive_cells=32,
        stop_after_iterations=999,
    )
    opt = AntColonyOptimizer(cfg, _sphere, _vars(), args={})
    opt.solve()
    assert isinstance(opt.soln_deck, CVTArchive)
    assert opt.soln_deck.coverage > 0.0


def test_scalar_mode_uses_solution_deck_not_cvt():
    set_seed(5)
    cfg = GeneticAlgorithmOptimizerConfig(
        name="s",
        num_generations=5,
        population_size=20,
        n_jobs=1,
        joblib_prefer="threads",
    )
    opt = GeneticAlgorithmOptimizer(cfg, _sphere, _vars(), args={})
    opt.solve()
    assert isinstance(opt.soln_deck, SolutionDeck)
    assert not isinstance(opt.soln_deck, CVTArchive)


def test_outputs_descriptor_source_not_yet_supported():
    set_seed(5)
    cfg = GeneticAlgorithmOptimizerConfig(
        name="me",
        num_generations=3,
        population_size=20,
        n_jobs=1,
        joblib_prefer="threads",
        objective_mode="map-elites",
        descriptor_source="outputs",
        n_outputs=2,
    )
    try:
        GeneticAlgorithmOptimizer(cfg, _sphere, _vars(), args={})
        raise AssertionError("expected NotImplementedError for outputs source")
    except NotImplementedError:
        pass
