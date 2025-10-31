from abc import ABC, abstractmethod
from typing import Union, List, Callable

import numpy as np

from .membership import MembershipFunction, FuzzyVariable
from .operator import FuzzyNot, FuzzyOperator, FuzzyArgument
from ..core.types import AF, F

# Some more type hints
FuzzyInput = Union[AF, FuzzyVariable, List[FuzzyVariable]]


class FuzzySet(ABC):
    def __init__(self, var_name: str, mf: list[MembershipFunction]):
        self.membership_functions = mf
        self.var_name = var_name

    def __call__(
        self,
        x: FuzzyInput,
        tgt_mf_name: str = "",
    ) -> list[FuzzyVariable]:
        # TODO - Better return type?
        return self.fuzzify(x, tgt_mf_name)

    def __str__(self) -> str:
        return f"FuzzySet:{self.var_name}: {self.membership_functions}"

    def __getitem__(self, item: str) -> MembershipFunction:
        for mf in self.membership_functions:
            if mf.name == item:
                return mf
        raise KeyError(f"Membership function {item} not found in {self}")

    def __contains__(self, item: AF | F) -> bool | AF:
        if isinstance(item, float):
            return self.domain[0] <= item <= self.domain[1]
        elif isinstance(item, AF):
            return np.logical_and(self.domain[0] <= item, item <= self.domain[1])
        else:
            raise TypeError("item must be float or AF")

    def __eq__(self, other: str | list[str]) -> FuzzyEquals:
        if isinstance(other, str):
            return FuzzyEquals(self, other)
        elif isinstance(other, list):
            y = FuzzyEquals(self, other[0])
            for i in range(1, len(other)):
                y = y | FuzzyEquals(self, other[i])
            return y
        else:
            raise TypeError("other must be str or list[str]")

    def __ne__(self, other: str | list[str]) -> FuzzyNot:
        return FuzzyNot(self == other)

    def __lt__(self, other: str | list[str]) -> FuzzyLessThan:
        return FuzzyLessThan(self, other[0])

    def __gt__(self, other: str | list[str]) -> FuzzyGreaterThan:
        return FuzzyGreaterThan(self, other[0])

    def __le__(self, other: str | list[str]) -> FuzzyLessThanEqual:
        return FuzzyLessThanEqual(self, other[0])

    def __ge__(self, other: str | list[str]) -> FuzzyGreaterThanEqual:
        return FuzzyGreaterThanEqual(self, other[0])

    @property
    def domain(self) -> AF:
        all_domains = [mf.domain() for mf in self.membership_functions]
        # TODO - Handle discontinuous domains!!
        return np.array(
            [min([d[0] for d in all_domains]), max([d[1] for d in all_domains])]
        )

    def fuzzify(
        self,
        x: AF | FuzzyVariable | list[FuzzyVariable],
        tgt_mf_name: str = "",
    ) -> list[FuzzyVariable]:
        if isinstance(x, list):
            # Ensure these are valid variables
            # TODO - Handle more than one!
            x = [v for v in x if v.var_name == self.var_name][0]
        if isinstance(x, FuzzyVariable):
            if tgt_mf_name:
                if tgt_mf_name not in [mf.name for mf in self.membership_functions]:
                    raise KeyError(
                        f"Membership function {tgt_mf_name} not found in {self}"
                    )
                # TODO - Should this be other than a list?
                return [
                    FuzzyVariable(mf.name, mf.mu(x.value))
                    for mf in self.membership_functions
                    if mf.name == tgt_mf_name
                ]
            return [
                FuzzyVariable(mf.name, mf.mu(x.value))
                for mf in self.membership_functions
            ]
        else:
            raise NotImplementedError("fuzzify() not implemented for NDArrays, yet!")

    def defuzzify(self, mu_x: AF) -> AF:
        raise NotImplementedError(
            "defuzzify() must be implemented in subclass by type of model!"
        )


class FuzzyInference(FuzzyVariable):
    def __init__(self, output_set: FuzzySet, var_name: str, mu_value: AF):
        super().__init__(var_name, mu_value)
        self.output_set: FuzzySet = output_set

    def __str__(self) -> str:
        return f"FuzzyInference:{self.var_name}:{self.value}"


class FuzzyEquals(FuzzyOperator):
    def __init__(self, a: FuzzySet, b: str):
        self.a = a
        self.b = b

    def __str__(self) -> str:
        return f"{self.a.var_name} == {self.b}"

    def evaluate(self, x: FuzzyArgument) -> AF:
        # Only look at the variable that matches this set
        if isinstance(x, list):
            req_var: FuzzyVariable = [v for v in x if v.var_name == self.a.var_name][0]
            y = self.a(req_var, self.b)
            # Convert to NDArray, since we know it is filtered to the right variable!
            return np.array([vy.value for vy in y])
        else:
            raise NotImplementedError("evaluate() not implemented for NDArrays, yet!")


class FuzzyLessThan(FuzzyOperator):
    def __init__(self, a: FuzzySet, b: str):
        self.a = a
        self.b = b

    def __str__(self) -> str:
        return f"{self.a.var_name} < {self.b}"

    def evaluate(self, x: FuzzyArgument) -> AF:
        raise NotImplementedError("evaluate() must be implemented in subclass")


class FuzzyLessThanEqual(FuzzyOperator):
    def __init__(self, a: FuzzySet, b: str):
        self.a = a
        self.b = b

    def __str__(self) -> str:
        return f"{self.a.var_name} <= {self.b}"

    def evaluate(self, x: FuzzyArgument) -> AF:
        raise NotImplementedError("evaluate() must be implemented in subclass")


class FuzzyGreaterThan(FuzzyOperator):
    def __init__(self, a: FuzzySet, b: str):
        self.a = a
        self.b = b

    def __str__(self) -> str:
        return f"{self.a.var_name} > {self.b}"

    def evaluate(self, x: FuzzyArgument) -> AF:
        raise NotImplementedError("evaluate() must be implemented in subclass")


class FuzzyGreaterThanEqual(FuzzyOperator):
    def __init__(self, a: FuzzySet, b: str):
        self.a = a
        self.b = b

    def __str__(self) -> str:
        return f"{self.a.var_name} >= {self.b}"

    def evaluate(self, x: FuzzyArgument) -> AF:
        raise NotImplementedError("evaluate() must be implemented in subclass")


class FuzzyRule(ABC):
    def __init__(
        self,
        rule_name: str,
        antecedent: FuzzyOperator,
        consequent: FuzzySet | None,
    ):
        self.rule_name = rule_name
        self.antecedent = antecedent
        self.consequent = consequent

    def __str__(self) -> str:
        return f"{self.rule_name}: IF {self.antecedent} THEN {self.consequent}"

    def __call__(self, x: FuzzyInput) -> FuzzyInference | AF:
        return self.evaluate(x)

    @abstractmethod
    def evaluate(self, x: FuzzyInput) -> FuzzyInference | AF:
        raise NotImplementedError("evaluate() must be implemented in subclass")


class MamdaniRule(FuzzyRule):
    def __init__(
        self,
        rule_name: str,
        antecedent: FuzzyOperator,
        consequent: FuzzySet,
        consequent_target: str,
    ):
        super().__init__(rule_name, antecedent, consequent)
        self.consequent_target = consequent_target

    def __str__(self) -> str:
        return f"{self.rule_name}: IF ({self.antecedent}) THEN {self.consequent.var_name} = {self.consequent_target}"

    def __call__(self, x: FuzzyInput) -> FuzzyInference:
        return self.evaluate(x)

    def evaluate(self, x: FuzzyInput) -> FuzzyInference:
        mu_x = self.antecedent(x)
        return FuzzyInference(self.consequent, self.consequent_target, mu_x)


class TSKRule(FuzzyRule):
    def __init__(
        self,
        rule_name: str,
        antecedent: FuzzyOperator,
        consequent_var_name: str,
        consequent_function: Callable[[AF], np.float64],
    ):
        super().__init__(rule_name, antecedent, None)
        self.consequent_var_name = consequent_var_name
        self.consequent_function = consequent_function

    def __str__(self) -> str:
        return f"{self.rule_name}: IF ({self.antecedent}) THEN {self.consequent_var_name} = {self.consequent}"

    def __call__(self, x: AF | list[FuzzyVariable]) -> AF:
        return self.evaluate(x)

    def evaluate(self, x: AF | list[FuzzyVariable]) -> AF:
        mu_x = self.antecedent(x)
        return mu_x * self.consequent_function(x)


class FuzzySystem(ABC):
    def __init__(self, fs: FuzzySet):
        self.fs = fs

    def __call__(self, x: AF) -> AF:
        return self.inference(x)

    @abstractmethod
    def inference(self, x: AF) -> AF:
        raise NotImplementedError("inference() must be implemented in subclass")
