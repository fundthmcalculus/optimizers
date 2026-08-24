from pathlib import Path

import numpy as np
import pytest

from optimizers import (
    GeneticAlgorithmOptimizer,
    GeneticAlgorithmOptimizerConfig,
    CheckpointConfig,
    save_checkpoint,
    load_checkpoint,
    run_multiple,
    plot_run_statistics,
)
from optimizers.continuous.optimizer_strategy import (
    GroupedVariableOptimizer,
    GroupedVariableOptimizerConfig,
    InputVariableGroup,
)
from optimizers.continuous.variables import InputContinuousVariable
from optimizers.solution_deck import SolutionDeck


@pytest.fixture()
def simple_ga_setup():
    # Simple convex sphere function in 2D
    def sphere(x):
        return float(np.sum(x**2))

    variables = [
        InputContinuousVariable("x1", -5.0, 5.0),
        InputContinuousVariable("x2", -5.0, 5.0),
    ]
    # Keep the numbers very small for quick tests
    cfg = GeneticAlgorithmOptimizerConfig(
        name="GA",
        num_generations=3,
        population_size=6,
        solution_archive_size=10,
        n_jobs=1,
        stop_after_iterations=2,
    )
    return sphere, variables, cfg


def test_save_and_load_checkpoint_single_run(tmp_path: Path, simple_ga_setup):
    fcn, variables, cfg = simple_ga_setup
    opt = GeneticAlgorithmOptimizer(config=cfg, fcn=fcn, variables=variables)
    res = opt.solve()

    # Save checkpoint
    cp_cfg = CheckpointConfig(enabled=True, folder=str(tmp_path), filename_prefix="ga")
    path = save_checkpoint(
        cp_cfg,
        optimizer_name="GA",
        config=cfg,
        solution_deck=opt.soln_deck,
        result=res,
        metadata={"note": "unit-test"},
    )

    assert path.exists(), "Checkpoint JSON should be created"

    # Load and validate
    loaded = load_checkpoint(path)
    assert "optimizer" in loaded and loaded["optimizer"] == "GA"
    assert isinstance(loaded.get("solution_deck"), SolutionDeck)

    deck: SolutionDeck = loaded["solution_deck"]
    # Ensure shapes and basic ordering (best first) are reasonable
    assert deck.solution_archive.shape[1] == len(variables)
    assert deck.solution_value.shape[0] == deck.solution_archive.shape[0]
    # Best value should be the first after sort
    deck.sort()
    assert np.isfinite(deck.solution_value[0])


def test_run_multiple_and_summary_and_plot(
    monkeypatch, tmp_path: Path, simple_ga_setup
):
    fcn, variables, base_cfg = simple_ga_setup

    def build_optimizer():
        # Fresh config/optimizer per run
        cfg = GeneticAlgorithmOptimizerConfig(**{**base_cfg.__dict__})
        opt = GeneticAlgorithmOptimizer(config=cfg, fcn=fcn, variables=variables)

        def runner():
            result = opt.solve()
            return opt.soln_deck, result

        return "GA", cfg, runner

    cp_cfg = CheckpointConfig(
        enabled=True, folder=str(tmp_path / "multi"), filename_prefix="ga_fold"
    )

    summary = run_multiple(
        n_runs=3,
        build_optimizer=build_optimizer,
        checkpoint_cfg=cp_cfg,
        summary_filename="summary.json",
    )

    # Basic structure checks
    assert summary["n_runs"] == 3
    assert isinstance(summary["scores"], list) and len(summary["scores"]) == 3
    assert isinstance(summary["runtimes"], list) and len(summary["runtimes"]) == 3
    assert isinstance(summary["runs"], list) and len(summary["runs"]) == 3

    # Summary file written
    summary_path = Path(cp_cfg.folder) / "summary.json"
    assert summary_path.exists(), "Summary JSON should be written"

    # Each run should have a checkpoint path
    for run_info in summary["runs"]:
        cp_path = run_info.get("checkpoint_path")
        assert cp_path is not None and Path(cp_path).exists()

    # Should not raise
    plot_run_statistics(summary, title_prefix="GA Multi-run Test")


def _grouped_config_kwargs():
    return dict(
        name="grouped-checkpoint-test",
        num_generations=2,
        population_size=4,
        solution_archive_size=6,
        n_jobs=1,
        stop_after_iterations=2,
        groups=[
            InputVariableGroup(name="x", variables=["x"], optimizer_type="ga"),
            InputVariableGroup(name="y", variables=["y"], optimizer_type="ga"),
        ],
    )


def test_grouped_variable_optimizer_checkpoints_each_round(tmp_path: Path):
    def sphere(x):
        return float(np.sum(x**2))

    variables = [
        InputContinuousVariable("x", -5.0, 5.0),
        InputContinuousVariable("y", -5.0, 5.0),
    ]
    cp_cfg = CheckpointConfig(
        enabled=True, folder=str(tmp_path), filename_prefix="grouped"
    )
    config = GroupedVariableOptimizerConfig(**_grouped_config_kwargs(), num_rounds=2)
    optimizer = GroupedVariableOptimizer(
        config=config, fcn=sphere, variables=variables, checkpoint_cfg=cp_cfg
    )
    result = optimizer.solve()

    checkpoints = sorted(tmp_path.glob("*.json"))
    assert len(checkpoints) == 2, "one checkpoint per completed round"
    rounds = sorted(load_checkpoint(p)["metadata"]["round"] for p in checkpoints)
    assert rounds == [0, 1]
    assert np.isfinite(result.solution_score)


def test_grouped_variable_optimizer_resume_skips_completed_rounds(tmp_path: Path):
    def sphere(x):
        return float(np.sum(x**2))

    variables = [
        InputContinuousVariable("x", -5.0, 5.0),
        InputContinuousVariable("y", -5.0, 5.0),
    ]
    cp_cfg = CheckpointConfig(
        enabled=True, folder=str(tmp_path), filename_prefix="grouped"
    )
    base_kwargs = _grouped_config_kwargs()

    # Run just round 0 and checkpoint it.
    first_config = GroupedVariableOptimizerConfig(**base_kwargs, num_rounds=1)
    first_optimizer = GroupedVariableOptimizer(
        config=first_config, fcn=sphere, variables=variables, checkpoint_cfg=cp_cfg
    )
    first_optimizer.solve()

    checkpoints = sorted(tmp_path.glob("*.json"))
    assert len(checkpoints) == 1
    round_zero_checkpoint = checkpoints[0]
    assert load_checkpoint(round_zero_checkpoint)["metadata"]["round"] == 0

    # A fresh optimizer, resumed from that checkpoint, over a longer run must
    # only execute (and checkpoint) the remaining round(s) -- not redo round 0.
    second_config = GroupedVariableOptimizerConfig(**base_kwargs, num_rounds=2)
    second_optimizer = GroupedVariableOptimizer(
        config=second_config, fcn=sphere, variables=variables, checkpoint_cfg=cp_cfg
    )
    result = second_optimizer.solve(resume_from=round_zero_checkpoint)

    checkpoints_after_resume = sorted(tmp_path.glob("*.json"))
    assert len(checkpoints_after_resume) == 2, "resume must add exactly one checkpoint"
    new_checkpoint = next(
        p for p in checkpoints_after_resume if p != round_zero_checkpoint
    )
    assert load_checkpoint(new_checkpoint)["metadata"]["round"] == 1
    assert np.isfinite(result.solution_score)


def test_grouped_variable_optimizer_resume_past_last_round_is_a_noop(tmp_path: Path):
    def sphere(x):
        return float(np.sum(x**2))

    variables = [
        InputContinuousVariable("x", -5.0, 5.0),
        InputContinuousVariable("y", -5.0, 5.0),
    ]
    cp_cfg = CheckpointConfig(
        enabled=True, folder=str(tmp_path), filename_prefix="grouped"
    )
    config = GroupedVariableOptimizerConfig(**_grouped_config_kwargs(), num_rounds=1)
    optimizer = GroupedVariableOptimizer(
        config=config, fcn=sphere, variables=variables, checkpoint_cfg=cp_cfg
    )
    optimizer.solve()
    (checkpoint_path,) = tmp_path.glob("*.json")

    # Resuming a run that already completed all its rounds must not crash and
    # must not write another checkpoint.
    result = optimizer.solve(resume_from=checkpoint_path)
    assert np.isfinite(result.solution_score)
    assert len(list(tmp_path.glob("*.json"))) == 1
