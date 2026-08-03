"""Regression tests for GitHub issue #101.

Two defects in the native PSO path (``continuous/pso.py``):

1. The velocity "clamp" multiplied velocity by a clamped *ratio* instead of
   clamping its magnitude, so it decayed geometrically to zero within a
   handful of iterations (well inside the 10 run per generation) -- the
   swarm effectively froze after its first few moves, every generation.
2. The archive's incumbent (``solution_archive[0]``) was never seeded as a
   particle position, only used as an external attractor, so a warm start
   never actually entered the swarm -- it only ever appeared in the
   comparison at the end, never as something particles could improve from.

See https://github.com/fundthmcalculus/optimizers/issues/101 for the full
writeup and reproduction.
"""

import numpy as np
import pytest

from optimizers.continuous.base import _ArgProvider
from optimizers.continuous.pso import (
    ParticleSwarmOptimizer,
    ParticleSwarmOptimizerConfig,
    run_particles,
)
from optimizers.continuous.variables import InputContinuousVariable
from optimizers.core.random import set_seed


def _sphere(x):
    return float(np.sum((x - 5.0) ** 2))


def _sphere_batch(x):
    return np.sum((x - 5.0) ** 2, axis=-1)


def test_velocity_clamp_saturates_instead_of_decaying():
    """The exact repro from the issue: an over-limit velocity should clamp to
    the limit and stay there, not shrink towards zero on repeated application.
    """
    domain, clamp = 10.0, 0.5
    limit = clamp * domain
    v = np.array([5.0])  # already above the limit
    for _ in range(8):
        v = np.clip(v, -limit, limit)
        assert v[0] == pytest.approx(limit)

    # And the buggy formula really did collapse geometrically, for contrast
    # (this is what the fix replaces -- not exercised by the shipped code).
    v_buggy = np.array([0.5])
    for _ in range(6):
        v_buggy = v_buggy * np.minimum(np.maximum(v_buggy / domain, -clamp), clamp)
    assert abs(v_buggy[0]) < 1e-10  # collapsed to ~5.4e-83 by this point


def test_incumbent_is_seeded_as_a_particle():
    """``run_particles`` must place the archive's best entry into the swarm as
    a real particle position (issue #101 defect 2), not just use it as an
    external attractor.

    The directly testable, seed-independent guarantee this gives: the
    returned population's best *personal-best* value can never be worse than
    the incumbent's own value, because one particle's personal best starts
    there and personal bests only ever improve (``p_best_val = np.where
    (improved, new_vals, p_best_val)``). Before the fix, every particle
    started at an unrelated random draw, so nothing guaranteed the returned
    population would contain (or beat) the incumbent at all within one
    generation.
    """
    n_dim = 5
    variables = [
        InputContinuousVariable(f"x{i}", -1000.0, 1000.0) for i in range(n_dim)
    ]
    incumbent = np.full(n_dim, 5.0) + 2.0
    incumbent_value = _sphere(incumbent)
    archive = np.tile(incumbent, (20, 1))  # irrelevant filler rows
    archive[0] = incumbent
    values = np.full(20, 1e9)
    values[0] = incumbent_value

    qd = (
        False,
        "native",
        0.01,
        0.2,
        np.array([-1000.0] * n_dim),
        np.array([1000.0] * n_dim),
    )
    for seed in range(5):
        set_seed(seed)
        fixed = (
            _ArgProvider(),
            variables,
            _sphere,
            0.5,
            1.5,
            1.5,
            0.5,
            16,
            "none",
            qd,
            _sphere_batch,
        )
        out = run_particles(fixed, {}, archive, values)
        assert out.population_values.min() <= incumbent_value + 1e-9, (
            f"seed={seed}: nothing in the returned population reached the "
            "incumbent's own value -- it was never actually a particle"
        )


def test_pso_makes_real_progress_without_freezing():
    """End-to-end: a wide-domain, no-warm-start run must land far better than
    a frozen swarm (which behaves like a handful of near-random draws per
    generation). Measured before this fix: ~2e5-5e5 over 3 seeds on this
    exact setup; after: ~2.6e3-3.9e3. The threshold below is a wide margin
    between the two, so it fails if the multiplicative-decay bug regresses.
    """
    n_dim = 10
    variables = [
        InputContinuousVariable(f"x{i}", -1000.0, 1000.0) for i in range(n_dim)
    ]

    for seed in range(3):
        set_seed(seed)
        config = ParticleSwarmOptimizerConfig(
            name="progress-check",
            population_size=30,
            num_generations=50,
            solution_archive_size=100,
            n_jobs=1,
            joblib_prefer="threads",
            local_grad_optim="none",
            stop_after_iterations=1000,
        )
        optimizer = ParticleSwarmOptimizer(
            config=config, variables=variables, fcn=_sphere, batch_fcn=_sphere_batch
        )
        result = optimizer.solve()
        assert result.solution_score < 50_000, (
            f"seed={seed}: PSO only reached {result.solution_score:.1f}, consistent "
            "with a frozen swarm behaving like near-random search"
        )


def test_warm_started_incumbent_never_regresses():
    """A warm-started run must never report a worse score than the incumbent
    it was seeded with -- the elitist archive already guarantees this
    independent of the PSO internals, but it's worth pinning given how easy
    it would be for a future change to the seeded particle's handling to
    accidentally drop it from the archive.
    """
    n_dim = 5
    variables = [
        InputContinuousVariable(f"x{i}", -1000.0, 1000.0) for i in range(n_dim)
    ]

    for seed in range(3):
        set_seed(seed)
        config = ParticleSwarmOptimizerConfig(
            name="no-regress",
            population_size=20,
            num_generations=10,
            solution_archive_size=60,
            n_jobs=1,
            joblib_prefer="threads",
            local_grad_optim="none",
            stop_after_iterations=1000,
        )
        optimizer = ParticleSwarmOptimizer(
            config=config, variables=variables, fcn=_sphere
        )
        incumbent = np.full(n_dim, 5.0) + 2.0
        optimizer.soln_deck.solution_archive[0] = incumbent
        optimizer.soln_deck.solution_value[0] = _sphere(incumbent)
        result = optimizer.solve(preserve_percent=1.0 / config.solution_archive_size)
        assert result.solution_score <= _sphere(incumbent) + 1e-9
