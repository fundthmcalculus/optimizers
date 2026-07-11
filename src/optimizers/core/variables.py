from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from .types import af64
from .random import rng as global_rng


class InputVariable(ABC):
    def __init__(self, name: str):
        self.name = name
        self.initial_value = 0.0

    @abstractmethod
    def random_value(
        self,
        current_value: float = np.nan,
        other_values: af64 | None = None,
        learning_rate: float = 0.7,
    ) -> float:
        pass

    def random_values(
        self,
        current_values: af64,
        other_values: af64 | None = None,
        learning_rate: float = 0.7,
        rng: Generator | None = None,
    ) -> af64:
        """Vectorized ``random_value`` — draw one sample per entry of
        ``current_values``. The base implementation loops (safe for any custom
        subclass); numeric subclasses override it with a vectorized version so a
        whole ant population can be sampled per variable in one call.
        """
        return np.array(
            [
                self.random_value(cv, other_values, learning_rate)
                for cv in np.asarray(current_values)
            ]
        )

    @abstractmethod
    def perturb_value(self, current_value: float, perturbation: float = 0.1) -> float:
        pass

    def perturb_values(
        self,
        current_values: af64,
        perturbation: float = 0.1,
        rng: Generator | None = None,
    ) -> af64:
        """Vectorized ``perturb_value`` — perturb each entry of
        ``current_values``. Base implementation loops; numeric subclasses
        override with a vectorized version.
        """
        return np.array(
            [self.perturb_value(cv, perturbation) for cv in np.asarray(current_values)]
        )

    @abstractmethod
    def initial_random_value(self, perturbation: float = 0.1) -> float:
        pass

    def initial_random_values(
        self, n: int, perturbation: float = 0.1, rng: Generator | None = None
    ) -> af64:
        """Vectorized ``initial_random_value`` — draw ``n`` fresh values. Base
        implementation loops; numeric subclasses override.
        """
        return np.array([self.initial_random_value(perturbation) for _ in range(n)])

    @abstractmethod
    def range_value(self, p: float) -> float:
        pass

    @property
    @abstractmethod
    def lower_bound(self) -> float:
        pass

    @property
    @abstractmethod
    def upper_bound(self) -> float:
        pass

    @property
    def domain(self) -> float:
        return abs(self.upper_bound - self.lower_bound)

    def initial_random_velocity(self) -> float:
        # NOTE - This doesn't mean as much for discrete variables, so we should probably ignore them?
        return global_rng().uniform(-self.domain, self.domain)

    def initial_random_velocities(self, n: int, rng: Generator | None = None) -> af64:
        """Vectorized ``initial_random_velocity`` — draw ``n`` velocities."""
        if rng is None:
            rng = global_rng()
        return rng.uniform(-self.domain, self.domain, size=n)


# Type hinting
InputVariables = list[InputVariable]
