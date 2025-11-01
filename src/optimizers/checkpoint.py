from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Any

import numpy as np

from .core.base import IOptimizerConfig, OptimizerResult
from .solution_deck import SolutionDeck


@dataclass
class CheckpointConfig:
    """Configuration for saving and loading optimizer checkpoints.

    Attributes:
        enabled: Whether checkpointing is active.
        folder: Directory where checkpoints and summaries will be saved.
        filename_prefix: Prefix for run files; a UUID and timestamp will be appended.
        save_solution_deck: Save the full solution archive for later warm-starting.
        save_config_blob: Save the optimizer configuration blob.
        save_result_blob: Save the final OptimizerResult blob.
    """

    enabled: bool = True
    folder: str = "./checkpoints"
    filename_prefix: str = "run"
    save_solution_deck: bool = True
    save_config_blob: bool = True
    save_result_blob: bool = True


def _ensure_folder(folder: str | os.PathLike[str]) -> Path:
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _new_run_id() -> str:
    return uuid.uuid4().hex


def save_checkpoint(
    checkpoint_cfg: CheckpointConfig,
    optimizer_name: str,
    config: IOptimizerConfig,
    solution_deck: Optional[SolutionDeck] = None,
    result: Optional[OptimizerResult] = None,
    run_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    """Save a checkpoint JSON blob to disk.

    Returns path to the saved JSON file.
    """
    if not checkpoint_cfg.enabled:
        raise RuntimeError("Checkpointing is disabled in CheckpointConfig")

    folder = _ensure_folder(checkpoint_cfg.folder)
    rid = run_id or _new_run_id()
    timestamp = _now_iso()

    payload: dict[str, Any] = {
        "run_id": rid,
        "timestamp": timestamp,
        "optimizer": optimizer_name,
        "config": asdict(config) if checkpoint_cfg.save_config_blob else None,
        "solution_deck": solution_deck.to_dict() if (solution_deck and checkpoint_cfg.save_solution_deck) else None,
        "result": {
            "solution_score": float(result.solution_score),
            "solution_vector": np.asarray(result.solution_vector).tolist(),
            "solution_history": (
                None
                if result.solution_history is None
                else np.asarray(result.solution_history).tolist()
            ),
            "stop_reason": result.stop_reason,
            "generations_completed": int(result.generations_completed),
        }
        if (result is not None and checkpoint_cfg.save_result_blob)
        else None,
        "metadata": metadata or {},
    }

    fname = f"{checkpoint_cfg.filename_prefix}_{optimizer_name}_{rid}__{timestamp.replace(':', '-')}.json"
    fpath = folder / fname
    with fpath.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return fpath


def load_checkpoint(file_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a checkpoint file and reconstruct objects where possible.

    Returns dict with keys: run_id, timestamp, optimizer, config, solution_deck, result, metadata
    """
    p = Path(file_path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # Reconstruct SolutionDeck if present
    deck = None
    if raw.get("solution_deck"):
        deck = SolutionDeck.from_dict(raw["solution_deck"])

    # OptimizerResult is simple enough to leave as raw dict for now; caller may adapt as needed
    out = {
        **raw,
        "solution_deck": deck,
    }
    return out


def run_multiple(
    n_runs: int,
    build_optimizer: Callable[[], tuple[str, IOptimizerConfig, Callable[[], tuple[SolutionDeck | None, OptimizerResult]]]],
    checkpoint_cfg: Optional[CheckpointConfig] = None,
    summary_filename: str = "summary.json",
) -> dict[str, Any]:
    """Execute multiple (fold) runs and collect statistics.

    Parameters:
        n_runs: Number of runs to execute.
        build_optimizer: A factory that returns a tuple:
            (optimizer_name, config, runner)
            where runner() -> (solution_deck, result) executes a single run and returns
            the final SolutionDeck (or None) and OptimizerResult.
        checkpoint_cfg: Optional checkpoint configuration; if provided, each run will be saved.
        summary_filename: Name of the summary JSON written to the checkpoint folder.

    Returns:
        A dictionary summary with per-run stats and overall arrays for plotting.
    """
    stats: list[dict[str, Any]] = []
    scores: list[float] = []
    runtimes: list[float] = []

    for i in range(n_runs):
        opt_name, cfg, runner = build_optimizer()
        start = time.perf_counter()
        deck, res = runner()
        runtime_s = time.perf_counter() - start

        entry = {
            "run_index": i,
            "optimizer": opt_name,
            "solution_score": float(res.solution_score),
            "generations_completed": int(res.generations_completed),
            "stop_reason": res.stop_reason,
            "runtime_seconds": float(runtime_s),
        }
        scores.append(entry["solution_score"])
        runtimes.append(entry["runtime_seconds"])

        if checkpoint_cfg and checkpoint_cfg.enabled:
            cp_path = save_checkpoint(
                checkpoint_cfg,
                optimizer_name=opt_name,
                config=cfg,
                solution_deck=deck,
                result=res,
                metadata={"run_index": i, "runtime_seconds": runtime_s},
            )
            entry["checkpoint_path"] = str(cp_path)

        stats.append(entry)

    summary = {
        "n_runs": n_runs,
        "scores": scores,
        "runtimes": runtimes,
        "runs": stats,
        "timestamp": _now_iso(),
    }

    if checkpoint_cfg and checkpoint_cfg.enabled:
        folder = _ensure_folder(checkpoint_cfg.folder)
        with (folder / summary_filename).open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary
