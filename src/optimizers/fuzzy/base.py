from abc import ABC, abstractmethod

import numpy as np

from optimizers.core.types import AF


epsilon_one: float = 0.999
epsilon_zero: float = 1.0 - epsilon_one
"""Epsilon value for handling roundoff errors in fuzzy logic operations."""


class TNormBase(ABC):
    @abstractmethod
    def norm(self, a: AF, b: AF) -> AF:
        raise NotImplementedError()

    @abstractmethod
    def conorm(self, a: AF, b: AF) -> AF:
        raise NotImplementedError()

    def negate(self, a: AF) -> AF:
        return 1.0 - a


class MinMaxNorm(TNormBase):
    def norm(self, a: AF, b: AF) -> AF:
        return np.min(a, b)

    def conorm(self, a: AF, b: AF) -> AF:
        return np.max(a, b)


class ProbabilityNorm(TNormBase):
    def norm(self, a: AF, b: AF) -> AF:
        return a * b

    def conorm(self, a: AF, b: AF) -> AF:
        return a + b - a * b


class LukasiewiczNorm(TNormBase):
    def norm(self, a: AF, b: AF) -> AF:
        return np.max(0, a + b - 1)

    def conorm(self, a: AF, b: AF) -> AF:
        return np.min(1, a + b)


class DrasticNorm(TNormBase):
    def norm(self, a: AF, b: AF) -> AF:
        p = np.zeros_like(a)
        p[a >= epsilon_one] = b[a >= epsilon_one]
        p[b >= epsilon_one] = a[b >= epsilon_one]
        return p

    def conorm(self, a: AF, b: AF) -> AF:
        p = np.ones_like(a)
        p[a <= epsilon_zero] = b[a <= epsilon_zero]
        p[b <= epsilon_zero] = a[b <= epsilon_zero]
        return p


class NilpotentNorm(TNormBase):
    def norm(self, a: AF, b: AF) -> AF:
        s = a + b
        p = np.zeros_like(s)
        p[s > epsilon_one] = np.min(a, b)
        return p

    def conorm(self, a: AF, b: AF) -> AF:
        s = a + b
        p = np.ones_like(s)
        p[s < epsilon_one] = np.max(a, b)
        return p


class HamacherNorm(TNormBase):
    def norm(self, a: AF, b: AF) -> AF:
        p = np.zeros_like(a)
        valid_mask = a >= epsilon_zero & b >= epsilon_zero
        p[valid_mask] = (
            a[valid_mask]
            * b[valid_mask]
            / (a[valid_mask] + b[valid_mask] - a[valid_mask] * b[valid_mask])
        )
        return p

    def conorm(self, a: AF, b: AF) -> AF:
        return (a + b) / (1 + a * b)


# TODO - This is a terrible inversion of control, but without resorting to Python metaclass magic, it's the easiest way to do this.
# TODO - Allow the user to define other t-norms and t-conorms!
default_norm: TNormBase = ProbabilityNorm()


def get_default_norm() -> TNormBase:
    return default_norm


def set_default_norm(norm: TNormBase) -> None:
    global default_norm
    default_norm = norm


def reset_default_norm() -> None:
    global default_norm
    default_norm = ProbabilityNorm()
