import copy

import numpy as np
import pytest

from optimizers.continuous.base import _ArgProvider
from optimizers.continuous.ga import (
    GeneticAlgorithmOptimizer,
    GeneticAlgorithmOptimizerConfig,
)
from optimizers.continuous.variables import InputContinuousVariable
from optimizers.core.base import ensure_literal_choice, literal_options, Phase
from optimizers.core.types import AF, F


@pytest.fixture()
def tiny_ga_cfg():
    # Small/fast config to exercise the pipeline
    return GeneticAlgorithmOptimizerConfig(
        name="GA-Metadata-Test",
        num_generations=3,
        population_size=8,
        solution_archive_size=16,
        n_jobs=1,  # keep single-process to simplify capturing
        stop_after_iterations=5,
        local_grad_optim="none",
    )


def test_phase_literal_definition():
    # The Phase Literal should expose only the allowed values
    opts = set(literal_options(Phase))
    assert opts == {"init", "evolve", "finalize"}
    # ensure_literal_choice should reject invalid values
    with pytest.raises(ValueError):
        ensure_literal_choice("bogus", Phase)


def test_metadata_injection_into_goal_and_constraints(tiny_ga_cfg):
    # Record snapshots of metadata seen inside the functions
    seen_goal = []

    user_args = {"note": "hello", "generation": -123}  # collision: metadata should win

    def obj(x: AF, args: dict) -> F:
        # Capture a compact snapshot for assertions
        snap = {
            "phase": args.get("phase"),
            "generation": args.get("generation"),
            "max_generation": args.get("max_generation"),
            "eval_count": args.get("eval_count"),
            "run_id": args.get("run_id"),
            "start_time": args.get("start_time"),
            "now": args.get("now"),
            "elapsed_time": args.get("elapsed_time"),
            "n_jobs": args.get("n_jobs"),
            "population_size": args.get("population_size"),
            "optimizer_name": args.get("optimizer_name"),
            "note": args.get("note"),
        }
        seen_goal.append(snap)
        # simple convex objective
        x = np.asarray(x, dtype=float)
        return float(np.sum(x**2))

    variables = [
        InputContinuousVariable("x0", -5.0, 5.0),
        InputContinuousVariable("x1", -5.0, 5.0),
    ]

    opt = GeneticAlgorithmOptimizer(
        tiny_ga_cfg,
        obj,
        variables,
        args=user_args,
    )

    result = opt.solve()

    # Basic sanity on result
    assert np.isfinite(result.solution_score)
    assert result.generations_completed >= 1

    # We should have seen multiple calls in init + evolve phases
    phases_seen = {snap["phase"] for snap in seen_goal}
    assert "init" in phases_seen  # during deck initialization
    assert "evolve" in phases_seen  # during GA loop

    # Metadata keys should be present and well-formed
    latest = seen_goal[-1]
    for key in [
        "generation",
        "max_generation",
        "phase",
        "eval_count",
        "now",
        "elapsed_time",
        "run_id",
        "start_time",
        "n_jobs",
        "population_size",
        "optimizer_name",
    ]:
        assert latest.get(key) is not None, f"Missing metadata key: {key}"

    # Types and relationships
    assert isinstance(latest["generation"], int)
    assert latest["max_generation"] == tiny_ga_cfg.num_generations
    assert latest["n_jobs"] == tiny_ga_cfg.n_jobs
    assert latest["population_size"] == tiny_ga_cfg.population_size
    assert latest["optimizer_name"] == tiny_ga_cfg.name
    assert latest["now"] >= latest["start_time"]
    assert latest["elapsed_time"] >= 0.0

    # User arg should be merged and preserved
    assert latest["note"] == "hello"
    # But colliding key 'generation' should be overwritten by metadata (never -123)
    assert all(s["generation"] != -123 for s in seen_goal)

    # Generations observed during evolve should be within range and non-decreasing
    evolve_generations = [s["generation"] for s in seen_goal if s["phase"] == "evolve"]
    assert all(0 <= g < tiny_ga_cfg.num_generations for g in evolve_generations)
    assert evolve_generations == sorted(evolve_generations)

    # eval_count should be monotonically increasing across all calls
    eval_counts = [s.get("eval_count", 0) for s in seen_goal]
    assert all(isinstance(e, int) and e >= 1 for e in eval_counts)
    assert eval_counts == sorted(eval_counts)

    # After solve completes, optimizer should report finalize phase in its metadata
    assert getattr(opt, "_arg_provider").meta.get("phase") == "finalize"


def test_arg_provider_eval_count_is_global_across_workers():
    """The eval_count protocol yields a true global running total.

    Simulates the parent/worker flow used under the ``processes`` backend, where
    each worker holds an independent copy of the arg provider: the parent ships a
    global base, each worker counts up from it, and the parent folds the
    per-worker deltas back into the authoritative total.
    """
    ap = _ArgProvider()

    # Parent initialization pass: 5 evaluations, counted globally from zero.
    for _ in range(5):
        ap.bump_eval()
    assert ap.eval_count == 5
    assert ap.current()["eval_count"] == 5
    # initialize() folds the init pass into the global base.
    ap.commit(ap.eval_delta)
    assert ap.eval_count == 5
    assert ap.eval_delta == 0

    # --- Generation 0: two independent process-worker copies ---
    base = ap.eval_count
    w1, w2 = copy.deepcopy(ap), copy.deepcopy(ap)
    w1.reset_local(base)
    w2.reset_local(base)
    for _ in range(3):
        w1.bump_eval()
    for _ in range(4):
        w2.bump_eval()
    # Each worker's goal fn sees a globally-offset count, not a from-zero tally.
    assert w1.current()["eval_count"] == base + 3
    assert w2.current()["eval_count"] == base + 4
    # The parent folds both per-worker deltas into the authoritative total.
    ap.commit(w1.eval_delta + w2.eval_delta)
    assert ap.eval_count == 5 + 3 + 4  # 12

    # --- Generation 1 ---
    base = ap.eval_count
    w3 = copy.deepcopy(ap)
    w3.reset_local(base)
    for _ in range(6):
        w3.bump_eval()
    assert w3.current()["eval_count"] == 12 + 6
    ap.commit(w3.eval_delta)
    assert ap.eval_count == 18  # exact global total, no double counting


def test_eval_count_global_under_processes_backend(monkeypatch):
    """End-to-end regression guard for issue #34's runtime ``eval_count``.

    Under the multiprocess backend each worker holds its own copy of the arg
    provider, so a naive counter tallies evaluations per-worker and the run-level
    count never reflects the global total. After the fix the parent's
    ``eval_count`` must grow across generations with every worker's evaluations
    folded in.
    """
    # Fast-mode caps force the backend to threads; disable so processes is used.
    monkeypatch.setenv("OPTIMIZERS_FAST", "")

    num_generations = 4
    population_size = 12
    cfg = GeneticAlgorithmOptimizerConfig(
        name="GA-GlobalEvalCount",
        num_generations=num_generations,
        population_size=population_size,
        solution_archive_size=24,
        n_jobs=2,
        joblib_prefer="processes",
        stop_after_iterations=999,  # never early-stop on no-improvement
        target_score=-1.0,  # never early-stop on target (sum-of-squares >= 0)
        local_grad_optim="none",
    )

    def obj(x: AF, args: dict) -> F:
        x = np.asarray(x, dtype=float)
        return float(np.sum(x**2))

    variables = [
        InputContinuousVariable("x0", -5.0, 5.0),
        InputContinuousVariable("x1", -5.0, 5.0),
    ]
    opt = GeneticAlgorithmOptimizer(cfg, obj, variables)
    # Guard: the process backend must actually be in effect for this to be a
    # meaningful test (fast mode would otherwise silently downgrade to threads).
    assert opt.config.joblib_prefer == "processes"

    opt.solve()

    total = opt._arg_provider.eval_count
    assert isinstance(total, int)
    # Each generation evaluates ~2*population candidates across all workers, on top
    # of the initialization pass. A per-worker counter would have stagnated near
    # the small init count; require at least one full population per generation.
    assert total >= num_generations * population_size
