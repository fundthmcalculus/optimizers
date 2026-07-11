"""Parallel dispatch that ships run-constant data to workers only once.

The optimizers (ACO/PSO/GA) run many generations, each dispatching the same
worker function ``n_jobs`` times. The arguments split cleanly into:

* **fixed** — constant for the whole run (the input variables, the wrapped goal
  function, and scalar hyper-parameters). The goal function commonly closes over
  a large read-only dataset supplied via ``args``.
* **varying** — changes every generation (the current solution archive, the rank
  CDF, ...), and is small now that the archive is bounded.

Previously every ``joblib.delayed`` dispatch re-pickled *all* arguments to every
worker on *every* generation, so a large fixed dataset was replicated to each
worker process once per generation. ``GenerationRunner`` sends the fixed payload
to each worker exactly once (via a persistent pool's initializer) and then
dispatches only the small varying arguments per generation.

See PERFORMANCE_REPORT.md items #2 (copy fixed data once) and #11 (threads vs
processes).
"""

import uuid
from typing import Any, Callable, Sequence

import joblib
from joblib import cpu_count
from joblib.externals.loky import ProcessPoolExecutor

# Per-worker cache of run-constant payloads, keyed by an opaque token. Populated
# once per worker process by the pool initializer. This lives at module scope on
# purpose: the initializer and the task functions both resolve it by reference in
# the worker, so the initializer's write is visible to later tasks (a value that
# was cloudpickled into each task instead would not be shared).
_FIXED: dict = {}


def resolve_n_jobs(n_jobs: int) -> int:
    """Normalize a config ``n_jobs`` (``< 1`` means "all but one core")."""
    if n_jobs is None or n_jobs < 1:
        return max(1, cpu_count() - 1)
    return n_jobs


def _init_worker(token: str, payload: Any) -> None:
    _FIXED[token] = payload


def _call_with_fixed(
    worker_fn: Callable, token: str, args: Sequence[Any]
) -> Any:
    # Runs in the worker: resolve the once-shipped fixed payload and call through.
    return worker_fn(_FIXED[token], *args)


class GenerationRunner:
    """Dispatch repeated identical worker calls, shipping ``fixed`` only once.

    Parameters
    ----------
    n_jobs:
        Number of parallel workers (``< 1`` means all-but-one core).
    prefer:
        ``"threads"`` — shared memory, no pickling; ``fixed`` is passed directly.
        Best when the goal function releases the GIL (vectorized numpy) or on a
        free-threaded interpreter.
        ``"processes"`` — a dedicated ``loky`` process pool whose initializer
        stashes ``fixed`` once per worker (``cloudpickle`` handles closures such
        as the wrapped goal function). Best for CPU-bound pure-Python goal
        functions.
    fixed:
        The run-constant payload handed to every worker call as its first
        argument. Sent to each process worker exactly once.

    Notes
    -----
    The processes backend owns a *dedicated* ``loky.ProcessPoolExecutor`` for
    the run's lifetime rather than loky's process-global reusable executor.
    Sharing that global singleton would collide with ``joblib.Parallel(prefer=
    "processes")`` used elsewhere (e.g. gradient descent): joblib expects to
    create and decorate the reusable executor itself, and finding a foreign one
    raises ``AttributeError: ... has no attribute '_temp_folder_manager'``. A
    dedicated pool keeps our once-shipped ``fixed`` payload isolated and is torn
    down in :meth:`close`.
    """

    def __init__(self, n_jobs: int, prefer: str, fixed: Any):
        self.n_jobs = resolve_n_jobs(n_jobs)
        self.prefer = prefer
        self._fixed = fixed
        self._token: str | None = None
        self._executor = None
        self._parallel = None
        if prefer == "processes":
            self._token = uuid.uuid4().hex
            # A dedicated pool (not loky's global reusable executor) so its
            # initializer ships ``fixed`` once per worker for this run's token,
            # without clobbering the reusable executor joblib.Parallel relies on.
            self._executor = ProcessPoolExecutor(
                max_workers=self.n_jobs,
                initializer=_init_worker,
                initargs=(self._token, fixed),
            )
        else:
            self._parallel = joblib.Parallel(n_jobs=self.n_jobs, prefer="threads")

    def run(
        self, worker_fn: Callable, varying_args: Sequence[Any], count: int | None = None
    ) -> list:
        """Call ``worker_fn(fixed, *varying_args)`` ``count`` times in parallel.

        ``count`` defaults to ``n_jobs`` (one task per worker per generation).
        """
        n = self.n_jobs if count is None else count
        if self.prefer == "processes":
            futures = [
                self._executor.submit(
                    _call_with_fixed, worker_fn, self._token, varying_args
                )
                for _ in range(n)
            ]
            return [f.result() for f in futures]
        return self._parallel(
            joblib.delayed(worker_fn)(self._fixed, *varying_args) for _ in range(n)
        )

    def close(self) -> None:
        # Tear down our dedicated process pool so its workers (and the once-
        # shipped fixed payload they hold) are released promptly. The threads
        # backend holds no OS resources to free.
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._parallel = None
