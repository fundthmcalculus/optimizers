import time
import numpy as np
import pytest

from optimizers.continuous.ga import GeneticAlgorithmOptimizer, GeneticAlgorithmOptimizerConfig
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
        ensure_literal_choice("phase", "bogus", Phase)


def test_metadata_injection_into_goal_and_constraints(tiny_ga_cfg):
    # Record snapshots of metadata seen inside the functions
    seen_goal = []
    seen_cons = []

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
        return float(np.sum(x ** 2))

    def ineq_g(x: AF, args: dict) -> F:
        # Record only phase/generation for constraint path
        seen_cons.append((args.get("phase"), args.get("generation"), args.get("eval_count")))
        x = np.asarray(x, dtype=float)
        # encourage small x[0]
        return float(x[0] - 10.0)  # <= 0 when x0 <= 10

    variables = [
        InputContinuousVariable("x0", -5.0, 5.0),
        InputContinuousVariable("x1", -5.0, 5.0),
    ]

    opt = GeneticAlgorithmOptimizer(
        tiny_ga_cfg,
        obj,
        variables,
        args=user_args,
        inequality_constraints=[ineq_g],
    )

    result = opt.solve()

    # Basic sanity on result
    assert np.isfinite(result.solution_score)
    assert result.generations_completed >= 1

    # We should have seen multiple calls in init + evolve phases
    phases_seen = {snap["phase"] for snap in seen_goal}
    assert "init" in phases_seen  # during deck initialization
    assert "evolve" in phases_seen  # during GA loop

    # Constraint should also receive metadata and see similar phases
    cons_phases = {p for (p, g, c) in seen_cons}
    assert "init" in cons_phases
    assert "evolve" in cons_phases

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