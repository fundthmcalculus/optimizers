"""Samplers for generating initial populations in unit hypercube [0,1]^d.

Supports both pseudo-uniform (i.i.d.) and quasi-random (low-discrepancy)
designs for improved space-filling and reproducibility of initial archives.
"""

from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
from scipy.stats import qmc

from .random import rng as global_rng

SamplerType = Literal["uniform", "sobol", "halton", "lhs"]


class Sampler(ABC):
    """Base class for samplers generating points in [0,1]^d."""

    @abstractmethod
    def sample(self, n: int, d: int, seed: int | None = None) -> np.ndarray:
        """Generate n points in d-dimensional unit hypercube [0,1]^d.

        Args:
            n: Number of points
            d: Dimension (number of variables)
            seed: Random seed for reproducibility

        Returns:
            Array of shape (n, d) with values in [0, 1]
        """
        pass


class UniformSampler(Sampler):
    """I.i.d. uniform sampling (current default behavior)."""

    def sample(self, n: int, d: int, seed: int | None = None) -> np.ndarray:
        """Generate n random uniform points in [0,1]^d."""
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = global_rng()
        return rng.uniform(0, 1, size=(n, d))


class SobolSampler(Sampler):
    """Sobol sequence (space-filling, good for moderate dimensions)."""

    def sample(self, n: int, d: int, seed: int | None = None) -> np.ndarray:
        """Generate n Sobol points in [0,1]^d.

        Sobol sequences have excellent uniformity properties (low
        star-discrepancy) especially in lower dimensions. Works for any n
        (automatically scrambled if n is not a power of 2).
        """
        sampler = qmc.Sobol(d=d, seed=seed, scramble=True)
        return sampler.random(n)


class HaltonSampler(Sampler):
    """Halton sequence (low-discrepancy, works well in higher dimensions)."""

    def sample(self, n: int, d: int, seed: int | None = None) -> np.ndarray:
        """Generate n Halton points in [0,1]^d.

        Halton sequences use coprime bases to generate low-discrepancy points.
        They work well even in higher dimensions compared to Sobol.
        """
        sampler = qmc.Halton(d=d, seed=seed, scramble=True)
        return sampler.random(n)


class LatinHypercubeSampler(Sampler):
    """Latin Hypercube Sampling (stratified, good for optimization)."""

    def sample(self, n: int, d: int, seed: int | None = None) -> np.ndarray:
        """Generate n Latin hypercube samples in [0,1]^d.

        LHS divides each dimension into n equal intervals and samples one point
        from each interval per dimension, ensuring good marginal coverage.
        """
        sampler = qmc.LatinHypercube(d=d, seed=seed, scramble=True)
        return sampler.random(n)


def create_sampler(sampler_type: SamplerType) -> Sampler:
    """Factory function to create a sampler of the specified type."""
    samplers: dict[SamplerType, type[Sampler]] = {
        "uniform": UniformSampler,
        "sobol": SobolSampler,
        "halton": HaltonSampler,
        "lhs": LatinHypercubeSampler,
    }
    if sampler_type not in samplers:
        raise ValueError(
            f"Unknown sampler type '{sampler_type}'. "
            f"Must be one of {list(samplers.keys())}"
        )
    sampler_class = samplers[sampler_type]
    return sampler_class()
