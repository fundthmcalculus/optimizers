from abc import ABC
from typing import Union, List, Tuple, Dict

import numpy as np

from optimizers.core.types import AF

class FuzzyVariable:
    def __init__(self, var_name: str, value: AF):
        self.var_name = var_name
        self.value = value

    def __repr__(self) -> str:
        return f"FuzzyVariable:{self.var_name}: {self.value}"

    def __str__(self) -> str:
        return repr(self)

# Some more type hints
FuzzyInput = Union[Tuple[AF, Dict[str, int]], FuzzyVariable, List[FuzzyVariable]]


class MembershipFunction(ABC):
    def __init__(self, name: str):
        self.name = name
        pass

    def __call__(self, x: FuzzyInput) -> AF:
        return self.mu(x)

    def __str__(self):
        return f"{self.__class__.__name__}:{self.name}:{self.domain()}"

    def __repr__(self) -> str:
        return self.__str__()

    def mu(self, x: AF) -> AF:
        raise NotImplementedError("mu() must be implemented in subclass")

    def inverse_mu(self, y: AF) -> AF:
        raise NotImplementedError("inverse_mu() must be implemented in subclass")

    def domain(self) -> AF:
        raise NotImplementedError("domain() must be implemented in subclass")

    def in_domain(self, x: AF | float) -> bool:
        if isinstance(x, float):
            x = np.array([x])
        return np.all(np.logical_and(self.domain()[0] <= x, x <= self.domain()[1]))

    def centroid(self) -> float:
        raise NotImplementedError("centroid() must be implemented in subclass")

    def d_dx(self, x: AF) -> AF:
        raise NotImplementedError("derivative() must be implemented in subclass")

    def gradiant(self, x: AF) -> AF:
        raise NotImplementedError("gradient() must be implemented in subclass")

    def hessian(self, x: AF) -> AF:
        raise NotImplementedError("hessian() must be implemented in subclass")


class TriangleMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float, c: float):
        super().__init__(name)
        assert a <= b <= c
        self.a = a
        self.b = b
        self.c = c

    def mu(self, x: AF) -> AF:
        return np.maximum(
            np.minimum(
                (x - self.a) / (self.b - self.a), (self.c - x) / (self.c - self.b)
            ),
            0,
        )

    def domain(self) -> AF:
        return np.array([self.a, self.c])

    def centroid(self) -> float:
        C_ab = 2 * self.b / 3 + self.a / 3
        A_ab = 0.5 * (self.b - self.a)
        C_bc = 2 * self.b / 3 + self.c / 3
        A_bc = 0.5 * (self.c - self.b)
        return (C_ab * A_ab + C_bc * A_bc) / (A_ab + A_bc)


class TrapezoidMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float, c: float, d: float):
        super().__init__(name)
        assert a <= b <= c <= d
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def mu(self, x: AF) -> AF:
        return np.maximum(
            np.minimum(
                (x - self.a) / (self.b - self.a), 1, (self.d - x) / (self.d - self.c)
            ),
            0,
        )

    def domain(self) -> AF:
        return np.array([self.a, self.d])


class LeftShoulderMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float):
        super().__init__(name)
        assert a <= b
        self.a = a
        self.b = b

    def mu(self, x: AF) -> AF:
        return np.maximum(np.minimum((self.b - x) / (self.b - self.a), 1), 0)

    def domain(self) -> AF:
        # TODO - Handle technically infinite domain?
        return np.array([self.a, self.b])

    def centroid(self) -> float:
        return 2 * self.a / 3 + self.b / 3


class RightShoulderMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float):
        super().__init__(name)
        assert a <= b
        self.a = a
        self.b = b

    def mu(self, x: AF) -> AF:
        return np.maximum(np.minimum((x - self.a) / (self.b - self.a), 1), 0)

    def domain(self) -> AF:
        # TODO - Handle technically infinite domain?
        return np.array([self.a, self.b])

    def centroid(self) -> float:
        return 2 * self.b / 3 + self.a / 3


# Type Hint
TriangleMembershipSequence = Union[
    TriangleMF, TrapezoidMF, LeftShoulderMF, RightShoulderMF
]


# Factory methods
def create_uniform_triangle_memberships(
    name: str | list[str], x0: float, x1: float, n_fcns: int
) -> list[TriangleMembershipSequence]:
    n_fcns = int(n_fcns)
    if isinstance(name, str):
        name = [f"{name}-{i}" for i in range(n_fcns)]
    spacing = (x1 - x0) / (n_fcns - 1)
    all_mus: list[TriangleMembershipSequence] = [
        LeftShoulderMF(name[0], x0, x0 + spacing)
    ]
    for ij in range(1, n_fcns - 1):
        all_mus.append(
            TriangleMF(
                name[ij],
                x0 + (ij - 1) * spacing,
                x0 + ij * spacing,
                x0 + (ij + 1) * spacing,
            )
        )
    all_mus.append(RightShoulderMF(name[-1], x0 + (n_fcns - 2) * spacing, x1))
    return all_mus


def create_triangle_memberships(
    triangle_data: dict[str, float],
) -> list[TriangleMembershipSequence]:
    all_mus: list[TriangleMembershipSequence] = []
    items = list(triangle_data.items())
    for idx, (name, value) in enumerate(items):
        if idx == 0:
            all_mus.append(LeftShoulderMF(name, items[idx][1], items[idx + 1][1]))
        elif idx == len(items) - 1:
            all_mus.append(RightShoulderMF(name, items[idx - 1][1], items[idx][1]))
        else:
            a, b, c = items[idx - 1][1], items[idx][1], items[idx + 1][1]
            all_mus.append(TriangleMF(name, a, b, c))
    return all_mus


class SinusoidMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float, c: float):
        super().__init__(name)
        assert a <= b <= c
        self.a = a
        self.b = b
        self.c = c

    def mu(self, x: AF) -> AF:
        m1 = (1.0 - np.cos(np.pi * (x - self.a) / (self.b - self.a))) / 2.0
        m2 = (1.0 + np.cos(np.pi * (x - self.b) / (self.c - self.b))) / 2.0
        m = np.zeros_like(x)
        m[np.logical_and(self.a <= x, x <= self.b)] = m1[
            np.logical_and(self.a <= x, x <= self.b)
        ]
        m[np.logical_and(self.b <= x, x <= self.c)] = m2[
            np.logical_and(self.b <= x, x <= self.c)
        ]
        return m

    def domain(self) -> AF:
        return np.array([self.a, self.c])

    def centroid(self) -> float:
        # Left sinusoid
        l_A = (self.c - self.b) / 2.0
        # Right sinsuoid
        r_A = (self.b - self.a) / 2.0
        l_C = ((4 + np.pi**2) * self.b + (np.pi**2 - 4) * self.c) / (2 * np.pi**2)
        r_C = (np.pi**2 * (self.a + self.b) - 4 * self.a + 4 * self.b) / (2 * np.pi**2)

        return (l_A * l_C + r_A * r_C) / (l_A + r_A)


class LeftSinusoidMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float):
        super().__init__(name)
        assert a <= b
        self.a = a
        self.b = b

    def mu(self, x: AF) -> AF:
        m2 = (1.0 + np.cos(np.pi * (x - self.a) / (self.b - self.a))) / 2.0
        m = np.zeros_like(x)
        m[np.logical_and(self.a <= x, x <= self.b)] = m2[
            np.logical_and(self.a <= x, x <= self.b)
        ]
        return m

    def domain(self) -> AF:
        return np.array([self.a, self.b])

    def centroid(self) -> float:
        return ((4 + np.pi**2) * self.a + (np.pi**2 - 4) * self.b) / (2 * np.pi**2)


class RightSinusoidMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float):
        super().__init__(name)
        assert a <= b
        self.a = a
        self.b = b

    def mu(self, x: AF) -> AF:
        m1 = (1.0 - np.cos(np.pi * (x - self.a) / (self.b - self.a))) / 2.0
        m = np.zeros_like(x)
        m[np.logical_and(self.a <= x, x <= self.b)] = m1[
            np.logical_and(self.a <= x, x <= self.b)
        ]
        return m

    def domain(self) -> AF:
        return np.array([self.a, self.b])

    def centroid(self) -> float:
        return (np.pi**2 * (self.a + self.b) - 4 * self.a + 4 * self.b) / (2 * np.pi**2)


# Type Hint
SinusoidMembershipSequence = Union[LeftSinusoidMF, RightSinusoidMF, SinusoidMF]


def create_sinusoid_memberships(
    triangle_data: dict[str, float],
) -> list[SinusoidMembershipSequence]:
    all_mus: list[SinusoidMembershipSequence] = []
    items = list(triangle_data.items())
    for idx, (name, value) in enumerate(items):
        if idx == 0:
            all_mus.append(LeftSinusoidMF(name, items[idx][1], items[idx + 1][1]))
        elif idx == len(items) - 1:
            all_mus.append(RightSinusoidMF(name, items[idx - 1][1], items[idx][1]))
        else:
            a, b, c = items[idx - 1][1], items[idx][1], items[idx + 1][1]
            all_mus.append(SinusoidMF(name, a, b, c))
    return all_mus


# Other interesting membership functions


class CauchyMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float):
        super().__init__(name)
        self.a = a
        self.b = b

    def mu(self, x: AF) -> AF:
        return 1 / (1 + ((x - self.a) / self.b) ** 2.0)

    def domain(self) -> AF:
        # TODO - Handle the long-tail of the distribution! :)
        n_sigma = 4.0
        return np.array([self.a - n_sigma * self.b, self.a + n_sigma * self.b])

    def centroid(self) -> float:
        return self.a

    def inverse_mu(self, y: AF) -> AF:
        return self.a + self.b * np.sqrt(1 / y - 1)


class GuassianMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float):
        super().__init__(name)
        assert a <= b
        self.a = a
        self.b = b

    def mu(self, x: AF) -> AF:
        return np.exp(-(((x - self.a) / self.b) ** 2))

    def domain(self) -> AF:
        # TODO - Handle the long-tail of the distribution, since the domain is technically [-inf, inf]
        n_sigma = 4.0
        return np.array([self.a - n_sigma * self.b, self.a + n_sigma * self.b])

    def centroid(self) -> float:
        return self.a

    def inverse_mu(self, y: AF) -> AF:
        # Y = exp(- (x-a)^2 /b^2)
        # TODO - Handle the other side option.
        return np.sqrt(-self.b**2 * np.log(y)) + self.a

    def d_dx(self, x: AF) -> AF:
        return self.mu(x) * -2.0 * (x - self.a) / self.b

    # TODO - Gradient and Hessian!
