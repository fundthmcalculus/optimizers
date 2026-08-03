import os
import random as pyrandom
import threading
from contextlib import contextmanager
from typing import Iterator

import numpy as np

# Global state for deterministic randomness across the project
_current_seed: int | None = None
_global_rng: np.random.Generator | None = None

# Root of the per-worker stream family. Kept separate from `_global_rng` so the
# main thread's number stream is byte-for-byte what it was before worker streams
# existed, and so the workers' streams are provably independent of it rather
# than children of the very sequence the main thread is drawing from.
_worker_sequence: np.random.SeedSequence | None = None

# Domain separator mixed into the worker root. Any fixed constant works; it only
# has to differ from the main stream's seeding so the two families cannot
# accidentally coincide.
_WORKER_DOMAIN = 0x5715_1CE5

# A parallel worker's own generator, set for the duration of one task. Thread
# state rather than global state because that is exactly the scope: concurrent
# tasks must not see each other's stream, and the main thread must keep its own.
_thread_local = threading.local()


def _new_entropy_seed() -> int:
    # Use OS entropy to create a 32-bit seed
    return int.from_bytes(os.urandom(8), byteorder="little") & 0x7FFFFFFF


def set_seed(seed: int | None) -> int:
    """
    Set the global random seed for NumPy and Python's random module.

    If seed is None, a fresh seed is generated from OS entropy.
    Returns the seed actually used.
    """
    global _current_seed, _global_rng, _worker_sequence
    if seed is None:
        seed = _new_entropy_seed()
    _current_seed = int(seed)
    # Seed Python's random
    pyrandom.seed(_current_seed)
    # Seed NumPy global RNG
    np.random.seed(_current_seed)
    # Create our shared Generator
    _global_rng = np.random.default_rng(_current_seed)
    # And the root the per-worker streams are spawned from.
    _worker_sequence = np.random.SeedSequence([_current_seed, _WORKER_DOMAIN])
    # A seed change invalidates any stream this thread was standing in.
    _thread_local.rng = None
    return _current_seed


def get_seed() -> int | None:
    """Return the current global seed if set; otherwise None."""
    return _current_seed


def rng() -> np.random.Generator:
    """
    Return the Generator the calling code should draw from.

    Inside a parallel worker task (see :func:`use_stream`) that is the task's own
    stream; everywhere else it is the shared global Generator. If no seed was set
    yet, create one using fresh entropy so default behavior remains
    non-deterministic unless the caller opted in via set_seed(...).
    """
    local: np.random.Generator | None = getattr(_thread_local, "rng", None)
    if local is not None:
        return local
    if _global_rng is None:
        set_seed(None)
    assert _global_rng is not None
    return _global_rng


def spawn_streams(n: int) -> list[np.random.Generator]:
    """``n`` independent Generators, derived deterministically from the seed.

    ``numpy.random.Generator`` is **not thread-safe**, so parallel workers sharing
    one Generator both race on its internal state and consume draws in whatever
    order the scheduler happens to pick -- which makes a seeded run irreproducible.
    Giving each task its own spawned stream fixes both: the streams are
    independent by construction (``SeedSequence.spawn``), and which numbers a task
    sees depends on its index, not on when it happens to run.

    Successive calls return successive families, so a per-generation call gives
    every generation fresh numbers while remaining a pure function of the seed and
    the call order. Call it from one thread -- the counter it advances is not
    itself synchronized.
    """
    if _worker_sequence is None:
        set_seed(None)
    assert _worker_sequence is not None
    return [np.random.default_rng(child) for child in _worker_sequence.spawn(n)]


@contextmanager
def use_stream(generator: np.random.Generator) -> Iterator[None]:
    """Make ``generator`` the result of :func:`rng` on this thread, for a task.

    Restores whatever was in place on exit, so nesting is safe and a worker
    thread reused by a later task never inherits the previous task's stream.
    """
    previous: np.random.Generator | None = getattr(_thread_local, "rng", None)
    _thread_local.rng = generator
    try:
        yield
    finally:
        _thread_local.rng = previous
