import abc
import typing
from abc import abstractmethod

import numpy as np
from numpy.random import Generator
from scipy.stats import truncnorm

from opt_types import f64, af64, T


def get_truncated_normal(mean=0.0, stdev=1.0, low=0.0, high=10.0) -> float:
    if stdev == 0.0:
        stdev = 1.0
    return truncnorm(
        (low - mean) / stdev, (high - mean) / stdev, loc=mean, scale=stdev
    ).rvs()


class InputVariable(abc.ABC):
    def __init__(self, name: str):
        self.name = name
        self.initial_value = 0.0

    @abstractmethod
    def random_value(
        self,
        current_value: f64 = np.nan,
        other_values: af64 = None,
        learning_rate: float = 0.7,
    ) -> f64:
        pass

    @abstractmethod
    def perturb_value(self, current_value: f64) -> f64:
        pass

    @abstractmethod
    def initial_random_value(self, perturbation: f64 = 0.1) -> f64:
        pass

    @property
    @abstractmethod
    def lower_bound(self) -> f64:
        pass

    @property
    @abstractmethod
    def upper_bound(self) -> f64:
        pass


InputVariables = list[InputVariable]


class InputDiscreteVariable(InputVariable):
    def __init__(self, name: str, values: list[T], initial_value: T | None = None):
        super().__init__(name)
        self.values = values
        self.initial_value = initial_value or self.random_value()

    def __repr__(self):
        return f"DV:{self.name} in {self.values}"

    def perturb_value(self, current_value: f64) -> f64:
        # Just randomly tweak to another choice.
        return self.initial_random_value()

    def random_value(
        self,
        current_value: f64 = np.nan,
        other_values: np.array = None,
        learning_rate: float = 0.7,
    ):
        rng = np.random.default_rng()
        if other_values is not None:
            # Convert into a weighted count, but ensure every option has a non-zero probability
            all_values = np.concatenate((self.values, other_values))
            unique, counts = np.unique(all_values, return_counts=True)
            # Unity normalize - TODO - Utilize the learning rate to adjust the non-base weights
            p_count = counts / np.sum(counts)
            return rng.choice(self.values, p=p_count)
        return rng.choice(self.values)

    def initial_random_value(self, perturbation: float = 0.1) -> f64:
        rng = np.random.default_rng()
        return rng.choice(self.values)

    @property
    def lower_bound(self) -> f64:
        return min(self.values)

    @property
    def upper_bound(self) -> f64:
        return max(self.values)

    def get_nearest_value(self, x1):
        return self.values[np.argmin(np.abs(self.values - x1))]


class InputContinuousVariable(InputVariable):
    def __init__(
        self,
        name: str,
        lower_bound: float,
        upper_bound: float,
        initial_value: float = None,
        perturbation: float = 0.1,
    ):
        super().__init__(name)
        self.__lower_bound = lower_bound
        self.__upper_bound = upper_bound
        self.initial_value = self.initial_random_value()
        if initial_value is not None:
            # Use perturbation theory around the initial value
            self.initial_value = min(
                self.upper_bound,
                max(
                    self.lower_bound,
                    initial_value + perturbation * (upper_bound - lower_bound),
                ),
            )

    def __repr__(self):
        return f"CV:{self.name} in [{self.lower_bound}, {self.upper_bound}]"

    def perturb_value(self, current_value: f64) -> f64:
        # Move it in a gaussian spread around the current value.
        sigma = (self.upper_bound - self.lower_bound) / 10
        new_value = current_value + sigma * np.random.normal()
        return max(min(self.upper_bound, new_value), self.lower_bound)

    def random_value(
        self,
        current_value: f64 = np.nan,
        other_values: np.array = None,
        learning_rate: float = 0.7,
    ):
        rng = np.random.default_rng()
        if other_values is not None:
            # TODO - Other than Mahattan distance, what other distance metrics can be used?
            d2 = np.sum(np.abs(other_values - current_value)) / len(other_values)
            return get_truncated_normal(
                mean=current_value,
                stdev=learning_rate * d2,
                low=self.lower_bound,
                high=self.upper_bound,
            )
        return rng.uniform(self.lower_bound, self.upper_bound)

    def initial_random_value(
        self, rng: Generator | None = None, perturbation: float = 0.1
    ) -> f64:
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.lower_bound, self.upper_bound)

    @property
    def lower_bound(self) -> f64:
        return self.__lower_bound

    @property
    def upper_bound(self) -> f64:
        return self.__upper_bound


def print_optimal_solution(
    x: np.ndarray, variables: list[InputContinuousVariable]
) -> None:
    print("Optimal solution:")
    for ij, var in enumerate(variables):
        print(f"{var.name}: {x[ij]}")
