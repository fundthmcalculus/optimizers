"""Quasi-random (low-discrepancy) sampling for initial population initialization.

Uses scipy.stats.qmc to generate Sobol, Halton, or Latin Hypercube samples
that are more evenly distributed than uniform random draws.
"""

from typing import Literal
import warnings

import numpy as np
from scipy.stats import qmc

from ..core.random import get_seed

# Re-export QMC classes for convenience
Sobol = qmc.Sobol
Halton = qmc.Halton
LatinHypercube = qmc.LatinHypercube


def generate_qmc_samples(
    n: int,
    d: int,
    sampler: Literal["sobol", "halton", "lhs"],
    seed: int | None = None,
) -> np.ndarray:
    """Generate quasi-random samples in [0,1]^d.

    Args:
        n: Number of samples to generate
        d: Dimensionality of the hypercube
        sampler: Type of QMC sampler ("sobol", "halton", or "lhs")
        seed: Optional seed for reproducibility. If None, uses global seed.

    Returns:
        Array of shape (n, d) with samples in [0,1]^d.

    Raises:
        ValueError: If sampler is not one of the supported types.
    """
    if seed is None:
        seed = get_seed()
    if seed is None:
        seed = 0

    sampler = sampler.lower()
    if sampler == "sobol":
        # Sobol typically generates 2^k samples; for non-power-of-2 we'll use
        # scrambling and truncate to n samples. Suppress warnings about sequence length.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            engine = qmc.Sobol(d=d, scramble=True, seed=seed)
            samples = engine.random(n)
    elif sampler == "halton":
        engine = qmc.Halton(d=d, scramble=True, seed=seed)
        samples = engine.random(n)
    elif sampler == "lhs":
        # Latin Hypercube Sampling
        engine = qmc.LatinHypercube(d=d, seed=seed)
        samples = engine.random(n)
    else:
        raise ValueError(
            f"Unknown sampler {sampler!r}. Supported: 'sobol', 'halton', 'lhs'"
        )

    return np.asarray(samples, dtype=np.float64)
