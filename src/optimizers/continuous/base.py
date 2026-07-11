import abc
import tqdm
import joblib
import numpy as np
import time
import uuid
import inspect
from typing import Optional, Any, Callable

from ..core import InputVariables
from ..core.base import (
    IOptimizerConfig,
    OptimizerResult,
    OptimizerRun,
    StopReason,
    ensure_literal_choice,
    JoblibPrefer,
    setup_for_generations,
    Phase,
    InputArguments,
    GoalFcn,
)
from ..core.types import AF, F
from ..core.random import get_seed
from ..solution_deck import (
    SolutionDeck,
    WrappedGoalFcn,
)
from ..archive.cvt import CVTArchive
from ..archive.descriptor import RandomProjectionDescriptor
from ..archive.metrics import QDReport, qd_score, pareto_front, hypervolume


class _ArgProvider:
    """Holds and provides merged input arguments (user args + runtime metadata).

    This object is intentionally lightweight and picklable so wrapped callables
    can be serialized for parallel workers while still carrying the current metadata.

    Evaluation counting is a **true global count** across the whole run — the
    number the goal function reads as ``eval_count`` reflects every evaluation the
    run has done so far, not a per-worker tally. Two fields make this work across
    both parallel backends:

    * ``eval_base`` — evaluations completed *before* the current counting context
      (previous generations, plus the parent's initialization pass).
    * ``_local`` — evaluations since the last :meth:`reset_local`.

    The value exposed to the goal function is always ``eval_base + _local``. Under
    the ``threads`` backend workers share the parent's provider, so ``_local``
    simply accumulates globally. Under ``processes`` each worker holds an
    independent copy, so the parent ships the authoritative ``eval_base`` each
    generation (:meth:`reset_local`), the worker counts its own evaluations up
    from it, and the parent folds the per-worker deltas back in (:meth:`commit`) —
    see ``IOptimizer.live_meta`` / ``sync_worker_meta`` /
    ``IOptimizer._accumulate_eval_count``.
    """

    def __init__(self, base_args: Optional[InputArguments] = None):
        self.base_args: InputArguments = dict(base_args or {})
        self.meta: InputArguments = {}
        # Evaluations done before the current counting context, and within it.
        self.eval_base: int = 0
        self._local: int = 0

    def current(self) -> InputArguments:
        # merge without mutating user-provided dicts
        merged = {**self.base_args, **self.meta}
        return merged

    @property
    def eval_count(self) -> int:
        """Global running evaluation count (``eval_base + _local``)."""
        return self.eval_base + self._local

    @property
    def eval_delta(self) -> int:
        """Evaluations in the current context (this generation, per worker)."""
        return self._local

    def bump_eval(self) -> None:
        self._local += 1
        now = time.time()
        self.meta["eval_count"] = self.eval_base + self._local
        self.meta["now"] = now
        # Elapsed seconds since run start
        start = self.meta.get("start_time", now)
        try:
            self.meta["elapsed_time"] = float(now - start)
        except Exception:
            self.meta["elapsed_time"] = 0.0

    def reset_local(self, base: int) -> None:
        """Begin a fresh counting context from ``base`` (process-worker copies).

        Called on each worker at the start of a generation so its evaluations are
        offset by the authoritative global base shipped from the parent.
        """
        self.eval_base = int(base)
        self._local = 0
        self.meta["eval_count"] = self.eval_base

    def commit(self, delta: int) -> None:
        """Fold ``delta`` completed evaluations into the global base (parent side).

        Used to (a) close out the parent's initialization pass and (b) absorb the
        summed per-worker deltas after each generation under the processes backend.
        """
        self.eval_base += int(delta)
        self._local = 0
        self.meta["eval_count"] = self.eval_base


class IOptimizer(abc.ABC):
    """Base class for all optimizer implementations"""

    def __init__(
        self,
        config: IOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: Optional[InputArguments] = None,
        existing_soln_deck: Optional[SolutionDeck] = None,
    ):
        self.config: IOptimizerConfig = config
        self.variables: InputVariables = variables
        self.args: Optional[InputArguments] = args

        # Runtime metadata provider shared by wrapped callables
        self._arg_provider = _ArgProvider(args)
        self._run_id = str(uuid.uuid4())
        self._start_time = time.time()
        # Initialize baseline metadata
        self._arg_provider.meta.update(
            {
                "run_id": self._run_id,
                "start_time": self._start_time,
                "generation": 0,
                "max_generation": config.num_generations,
                "phase": "init",
                "n_jobs": config.n_jobs,
                "population_size": config.population_size,
                "optimizer_name": getattr(config, "name", type(config).__name__),
            }
        )

        # Detect whether functions accept (x, args) or just (x)
        def _accepts_args(func: Any) -> bool:
            try:
                sig = inspect.signature(func)
                return len(sig.parameters) >= 2
            except (ValueError, TypeError):
                return True  # assume safe to pass args

        # Quality-diversity add-on (QD_PARETO_PLAN.md §1). In a non-scalar
        # ``objective_mode`` the goal function returns ``(fitness, outputs)``. The
        # wrapped fitness callable still yields a scalar, so every existing solver
        # is untouched; the extra outputs are collected parent-side and tracked on
        # the deck. Default ``"scalar"`` mode is byte-identical to before.
        self._objective_mode = getattr(config, "objective_mode", "scalar")
        self._descriptor_source = getattr(config, "descriptor_source", "projection")
        self._n_outputs = int(getattr(config, "n_outputs", 0))
        # The goal fn returns (fitness, outputs) when the user asks to track
        # objectives for the Pareto report (n_outputs > 0). ``fitness`` still
        # drives the search (scalar); outputs are read at report time. The
        # default projection path tracks nothing and stays a plain scalar run.
        self._returns_outputs = self._objective_mode != "scalar" and self._n_outputs > 0

        def _wrap_goal(func: GoalFcn) -> WrappedGoalFcn:
            takes_args = _accepts_args(func)

            def __wrapped(
                x: AF,
                _f: Callable[..., F] = func,
                _ap: "_ArgProvider" = self._arg_provider,
            ) -> F:
                # Only pay the runtime-metadata bookkeeping (a time.time() call
                # plus dict writes, per evaluation) when the goal function
                # actually consumes args. For plain ``f(x)`` objectives nothing
                # reads the metadata, so skip it entirely. See report item #14.
                if takes_args:
                    _ap.bump_eval()
                    result = _f(x, _ap.current())
                else:
                    result = _f(x)
                # Multi-output goal fns return (fitness, outputs); solvers only
                # ever need the scalar fitness.
                return result[0] if self._returns_outputs else result

            return __wrapped

        # Wrap the goal function and constraint functions
        self.wrapped_fcn: WrappedGoalFcn = _wrap_goal(fcn)

        # Parent-side full evaluator, used only in multi-output mode to gather the
        # tracked outputs for archived solutions. It is a *recording* pass (no eval
        # bookkeeping), not a search evaluation. NOTE (Phase 1): this re-evaluates
        # the objective in the parent; Phase 2 moves output capture into the
        # workers so it costs nothing extra.
        _takes_args = _accepts_args(fcn)

        def _eval_full(
            x: AF,
            _f: Callable[..., F] = fcn,
            _ap: "_ArgProvider" = self._arg_provider,
            _ta: bool = _takes_args,
        ) -> F:
            return _f(x, _ap.current()) if _ta else _f(x)

        self._eval_full = _eval_full

        if existing_soln_deck is not None:
            self.soln_deck = existing_soln_deck
        elif self._objective_mode == "map-elites":
            self.soln_deck = self._build_mapelites_archive(config, variables)
        else:
            self.soln_deck = SolutionDeck(
                archive_size=config.solution_archive_size,
                num_vars=len(variables),
                n_outputs=self._n_outputs if self._returns_outputs else 0,
            )

    def _build_mapelites_archive(
        self, config: IOptimizerConfig, variables: InputVariables
    ) -> CVTArchive:
        """Construct the CVT MAP-Elites archive (Phase 2, QD_PARETO_PLAN.md §4.1)."""
        lower = np.array([v.lower_bound for v in variables], dtype=float)
        upper = np.array([v.upper_bound for v in variables], dtype=float)
        seed = get_seed()
        if self._descriptor_source == "outputs":
            raise NotImplementedError(
                "descriptor_source='outputs' lands in a later phase; use the "
                "default 'projection' descriptor for now."
            )
        ddim = int(config.descriptor_dim)
        descriptor_fn = RandomProjectionDescriptor(
            len(variables), ddim, lower, upper, seed=seed or 0
        )
        return CVTArchive(
            num_vars=len(variables),
            lower=lower,
            upper=upper,
            descriptor_fn=descriptor_fn,
            descriptor_dim=ddim,
            n_cells=int(config.archive_cells),
            descriptor_source=self._descriptor_source,
            seed=seed,
        )

    def live_meta(self) -> InputArguments:
        """The per-generation runtime metadata that changes during the run.

        The rest of the metadata (run_id, max_generation, population_size, ...)
        is constant and ships with the goal function once, so only these live
        fields need to be re-synced to worker processes each generation.
        """
        meta = self._arg_provider.meta
        live: InputArguments = {
            "generation": meta.get("generation"),
            "phase": meta.get("phase"),
        }
        if self.config.joblib_prefer == "processes":
            # Each process worker holds an independent copy of the arg provider,
            # so hand it the authoritative global evaluation base to count this
            # generation up from; the per-worker deltas are folded back in on the
            # parent (see ``_accumulate_eval_count``). Under threads the workers
            # share the parent's provider, whose counter is already global, so no
            # base is shipped and no reset happens.
            live["_eval_base"] = self._arg_provider.eval_count
        return live

    def _set_phase(self, phase: Phase) -> None:
        # Validate at runtime for extra safety in non-type-checked contexts
        try:
            ensure_literal_choice(phase, Phase)
        except Exception:
            # Keep fail-soft: still set, but this should not happen given our callers
            pass
        self._arg_provider.meta["phase"] = phase

    def _set_generation(self, generation: int) -> None:
        self._arg_provider.meta["generation"] = generation

    @abc.abstractmethod
    def solve(self, preserve_percent: float = 0.0) -> OptimizerResult:
        """
        Solve the given problem.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def initialize(
        self, preserve_percent: float
    ) -> tuple[list[F], tqdm.tqdm, int, int, int, joblib.Parallel, StopReason]:
        self.validate_config()
        self.soln_deck.initialize_solution_deck(
            self.variables, self.wrapped_fcn, preserve_percent
        )
        self.soln_deck.sort()
        # QD add-on: seed tracked outputs for the initial archive so the deck's
        # solution_outputs stays row-aligned from generation zero.
        if self.soln_deck.solution_outputs is not None:
            self.soln_deck.set_all_outputs(
                self._evaluate_outputs(self.soln_deck.solution_archive)
            )

        # The initialization pass above evaluated the goal function in the parent
        # process; fold those evaluations into the global base so the first
        # generation's ``eval_count`` continues from them (and process workers are
        # shipped the correct starting base). See ``_ArgProvider``.
        self._arg_provider.commit(self._arg_provider.eval_delta)

        # Add the progress bar
        generation_pbar, individuals_per_job, n_jobs, parallel = setup_for_generations(
            self.config
        )
        stopped_early: StopReason = "none"
        generations_completed = 0
        return (
            [],
            generation_pbar,
            generations_completed,
            individuals_per_job,
            n_jobs,
            parallel,
            stopped_early,
        )

    def _evaluate_outputs(self, solutions: AF) -> AF:
        """Collect the tracked-outputs vector for each solution (multi-output mode).

        A recording pass that keeps the deck's ``solution_outputs`` populated; it
        does not influence search (Phase 1). Expects the goal function to return
        ``(fitness, outputs)`` with ``outputs`` of length ``n_outputs``.
        """
        solutions = np.atleast_2d(solutions)
        outs = np.empty((solutions.shape[0], self._n_outputs), dtype=float)
        for i in range(solutions.shape[0]):
            _, outputs = self._eval_full(solutions[i])
            outs[i, :] = np.asarray(outputs, dtype=float).ravel()
        return outs

    def qd_report(self, reference: AF | None = None) -> QDReport:
        """Summarize the finished run (QD_PARETO_PLAN.md §4.5).

        Reports archive coverage / QD-score / best fitness, and — when the goal
        function tracks objectives (``n_outputs > 0``) — the **Pareto front** over
        those objectives across the final archive plus its hypervolume. Outputs
        are evaluated once here over the final elites (a cheap recording pass), so
        the per-generation search path pays nothing for the report.
        """
        archive = self.soln_deck
        archive.sort()
        values = np.asarray(archive.solution_value, dtype=float)
        num = int(len(archive))
        best = float(values[0]) if values.size else float("inf")
        coverage = getattr(archive, "coverage", None)
        report = QDReport(
            num_elites=num,
            best_fitness=best,
            coverage=coverage,
            qd_score=qd_score(values),
        )
        if self._returns_outputs and num > 0:
            objs = self._evaluate_outputs(archive.solution_archive)
            front = pareto_front(objs)
            report.all_objectives = objs
            report.pareto_solutions = archive.solution_archive[front]
            report.pareto_objectives = objs[front]
            ref = objs.max(axis=0) + 1e-9 if reference is None else reference
            report.hypervolume = hypervolume(objs, ref)
        return report

    def _accumulate_eval_count(self, job_output: list[OptimizerRun]) -> None:
        """Fold this generation's per-worker evaluation deltas into the global count.

        Only the ``processes`` backend needs this: each worker counted its own
        evaluations from the shipped global base, so the parent sums the deltas
        reported on each ``OptimizerRun`` and advances the authoritative base.
        Under ``threads`` the workers share the parent's provider, whose counter is
        already global, so there is nothing to fold (doing so would double-count).
        """
        if self.config.joblib_prefer != "processes":
            return
        delta = sum(int(getattr(o, "eval_count", 0) or 0) for o in job_output)
        self._arg_provider.commit(delta)

    def update_solution_deck(
        self, generation_pbar: tqdm.tqdm, job_output: list[OptimizerRun]
    ) -> None:
        if not job_output:
            return
        self._accumulate_eval_count(job_output)
        # Merge all worker outputs in a single batch, then deduplicate/sort and
        # truncate back to archive_size ONCE per generation. Previously this
        # looped per-output calling append + deduplicate (which sorts), so the
        # ever-growing archive was re-sorted n_jobs times every generation and
        # never bounded. See PERFORMANCE_REPORT.md items #12 (sort/dedup once)
        # and #3 (bound growth).
        all_solutions = np.vstack(
            [output.population_solutions for output in job_output]
        )
        all_values = np.concatenate(
            [np.atleast_1d(output.population_values) for output in job_output]
        )
        # QD add-on: record the tracked outputs (only when the descriptor / report
        # is built from goal-function outputs; the projection descriptor needs
        # none). The archive's ``add_generation`` owns the insertion rule — elitist
        # truncation for the scalar deck, cell replacement for MAP-Elites.
        all_outputs = None
        if self.soln_deck.solution_outputs is not None:
            all_outputs = self._evaluate_outputs(all_solutions)
        self.soln_deck.add_generation(
            all_solutions,
            all_values,
            outputs=all_outputs,
            local_optima=self.config.local_grad_optim != "none",
        )
        generation_pbar.set_postfix(best_value=self.soln_deck.solution_value[0])

    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        """
        # Validate joblib prefer value against allowed Literal options
        ensure_literal_choice(self.config.joblib_prefer, JoblibPrefer)
        # Set the default values for the config
        if self.config.solution_archive_size < 0:
            self.config.solution_archive_size = len(self.variables) * 2
        if self.config.population_size < 0:
            self.config.population_size = self.config.solution_archive_size // 3
        if self.config.n_jobs < 0:
            self.config.n_jobs = joblib.cpu_count() - 1

    def __str__(self) -> str:
        return f"Solver(name={self.config.name})"


def sync_worker_meta(
    arg_provider: "_ArgProvider | None", meta: Optional[InputArguments]
) -> None:
    """Apply the parent's live metadata snapshot to a worker's arg provider.

    In the ``processes`` backend each worker holds its own copy of the arg
    provider (shipped once with the goal function), so the optimizer passes the
    small live-metadata dict each generation and the worker merges it here before
    evaluating the goal function. That dict also carries ``_eval_base`` — the
    authoritative global evaluation count — so the worker restarts its local
    counter offset by it, keeping ``eval_count`` a true global running total (see
    ``_ArgProvider``). Threads share the parent's provider and ship no base.
    """
    if arg_provider is None or not meta:
        return
    base = meta.pop("_eval_base", None)
    arg_provider.meta.update(meta)
    if base is not None:
        arg_provider.reset_local(base)


def check_stop_early(
    config: IOptimizerConfig, best_soln_history: list[F], solution_values: AF
) -> StopReason:
    if solution_values[0] <= config.target_score:
        print("Target score reached, terminating early.")
        return "target_score"
    # Check if the solution hasn't improved
    if len(best_soln_history) < config.stop_after_iterations:
        return "none"
    recent_history = best_soln_history[-config.stop_after_iterations :]
    if np.allclose(recent_history[-1], recent_history[0], rtol=1e-2, atol=1e-2):
        print(
            f"No improvement in last {config.stop_after_iterations} iterations. Stopping early."
        )
        return "no_improvement"
    return "none"


def cdf(q: F, N: int) -> AF:
    """
    Parameters
    ----------
    q: float The weighting parameter for better ranked solutions.
    N: int The number of solutions in the solution archive.

    Returns
    -------
    af64 The cumulative density function.
    """
    j = np.r_[1 : N + 1]
    c1 = 1 - np.exp(-q * j / N)
    # Unity scaling, and since the CDF is positive-definite, we can use the last entry.
    return c1 / c1[-1]
