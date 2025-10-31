from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np

from .base import default_norm
from .membership import FuzzyVariable
from ..core.types import AF


FuzzyArgument = Union[AF, List[FuzzyVariable]]


class FuzzyOperator(ABC):
    def __call__(self, *varargs: FuzzyArgument) -> AF:
        return self.evaluate(*varargs)

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError("__str__() must be implemented in subclass")

    def __and__(self, other: "FuzzyOperator") -> "FuzzyOperator":
        return FuzzyAnd(self, other)

    def __or__(self, other: "FuzzyOperator") -> "FuzzyOperator":
        return FuzzyOr(self, other)

    def __neg__(self) -> "FuzzyOperator":
        return FuzzyNot(self)

    @abstractmethod
    def evaluate(self, *varargs: FuzzyArgument) -> AF:
        raise NotImplementedError("evaluate() must be implemented in subclass")


# These are the defaults, but there are alternative implementations of this!
class FuzzyAnd(FuzzyOperator):
    def __init__(self, a: FuzzyOperator, b: FuzzyOperator):
        self.a = a
        self.b = b

    def __str__(self) -> str:
        return f"({self.a} AND {self.b})"

    def evaluate(self, x: FuzzyArgument) -> AF:
        a1 = self.a(x)
        b1 = self.b(x)
        return default_norm.norm(a1, b1)


class FuzzyOr(FuzzyOperator):
    def __init__(self, a: FuzzyOperator, b: FuzzyOperator):
        self.a = a
        self.b = b

    def __str__(self) -> str:
        return f"({self.a} OR {self.b})"

    def evaluate(self, x: FuzzyArgument) -> AF:
        a1 = self.a(x)
        b1 = self.b(x)
        return default_norm.conorm(a1, b1)


class FuzzyNot(FuzzyOperator):
    def __init__(self, a: FuzzyOperator):
        self.a = a

    def __str__(self) -> str:
        return f"NOT {self.a}"

    def evaluate(self, x: FuzzyArgument) -> AF:
        a1 = self.a(x)
        return default_norm.negate(a1)
