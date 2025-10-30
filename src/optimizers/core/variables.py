from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from typing import Optional, List
from .types import af64


class InputVariable(ABC):
    def __init__(self, name: str):
        self.name = name
        self.initial_value = 0.0

    @abstractmethod
    def random_value(
        self,
        current_value: float = np.nan,
        other_values: Optional[af64] = None,
        learning_rate: float = 0.7,
    ) -> float:
        pass

    @abstractmethod
    def perturb_value(self, current_value: float) -> float:
        pass

    @abstractmethod
    def initial_random_value(self, perturbation: float = 0.1) -> float:
        pass

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


# Type hinting
InputVariables = List[InputVariable]
