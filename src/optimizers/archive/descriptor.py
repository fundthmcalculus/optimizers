"""Descriptor functions for MAP-Elites (QD_PARETO_PLAN.md §4.2).

A *descriptor* maps a solution to a low-dimensional coordinate; MAP-Elites uses
it to spread elites across a feature space. The default for high-dimensional
decision spaces (no natural behaviour descriptor) is a **fixed seeded random
projection** of the decision vector — cheap, dimension-agnostic, and
distance-preserving in expectation (Johnson-Lindenstrauss). Held constant across
a run so cells are stable; picklable so it survives checkpointing.
"""

from __future__ import annotations

import numpy as np

from ..core.types import AF


class RandomProjectionDescriptor:
    """``d(x) = normalize(x) @ R.T`` with a fixed seeded Gaussian ``R``.

    Decision vectors are first normalized to ``[0, 1]`` per dimension (so the
    projection is scale-invariant across heterogeneous variable domains), then
    projected to ``descriptor_dim`` coordinates. ``R`` is scaled by
    ``1/sqrt(num_vars)`` to keep the projected variance stable as dimension grows.
    """

    def __init__(
        self,
        num_vars: int,
        descriptor_dim: int,
        lower: AF,
        upper: AF,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.R = rng.standard_normal((descriptor_dim, num_vars)) / np.sqrt(num_vars)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.descriptor_dim = descriptor_dim

    def __call__(self, x: AF) -> AF:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        span = np.where(self.upper > self.lower, self.upper - self.lower, 1.0)
        x_norm = (x - self.lower) / span
        return x_norm @ self.R.T  # (n, descriptor_dim)


class OutputsDescriptor:
    """Use a subset of the goal function's tracked outputs as the descriptor.

    For ``descriptor_source="outputs"``: the archive passes the tracked outputs
    (shape ``(n, n_outputs)``) and this selects ``columns`` of them.
    """

    def __init__(self, columns: list[int]):
        self.columns = list(columns)
        self.descriptor_dim = len(columns)

    def __call__(self, outputs: AF) -> AF:
        outputs = np.atleast_2d(np.asarray(outputs, dtype=float))
        return outputs[:, self.columns]
