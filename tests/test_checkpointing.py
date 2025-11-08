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
    opt = GeneticAlgorithmOptimizer(cfg, fcn, variables)
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
        opt = GeneticAlgorithmOptimizer(cfg, fcn, variables)

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
