import os
import random as pyrandom
from typing import Optional

import numpy as np

# Global state for deterministic randomness across the project
_current_seed: Optional[int] = None
_global_rng: np.random.Generator | None = None


def _new_entropy_seed() -> int:
    # Use OS entropy to create a 32-bit seed
    return int.from_bytes(os.urandom(8), byteorder="little") & 0x7FFFFFFF


def set_seed(seed: Optional[int]) -> int:
    """
    Set the global random seed for NumPy and Python's random module.

    If seed is None, a fresh seed is generated from OS entropy.
    Returns the seed actually used.
    """
    global _current_seed, _global_rng
    if seed is None:
        seed = _new_entropy_seed()
    _current_seed = int(seed)
    # Seed Python's random
    pyrandom.seed(_current_seed)
    # Seed NumPy global RNG
    np.random.seed(_current_seed)
    # Create our shared Generator
    _global_rng = np.random.default_rng(_current_seed)
    return _current_seed


def get_seed() -> Optional[int]:
    """Return the current global seed if set; otherwise None."""
    return _current_seed


def rng() -> np.random.Generator:
    """
    Return the shared NumPy Generator. If no seed was set yet, create one using
    fresh entropy so default behavior remains non-deterministic unless the
    caller opted in via set_seed(...).
    """
    global _global_rng
    if _global_rng is None:
        set_seed(None)
    assert _global_rng is not None
    return _global_rng
