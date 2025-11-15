import numpy as np

from fuzzy.fuzzy_set import FuzzyRule
from fuzzy.membership import create_sinusoid_memberships


class FuzzyControl:
    def __init__(
        self,
        shape: tuple[int, int],
        input_range: tuple[float, float],
        output_range: tuple[float, float],
    ) -> None:
        self.shape = shape
        self.input_range = input_range
        self.output_range = output_range
        _centers1 = np.linspace(input_range[0], input_range[1], shape[0])
        _centers2 = np.linspace(input_range[0], input_range[1], shape[1])
        _mf1_names = [f"input1-{ij}" for ij in range(shape[0])]
        _mf2_names = [f"input2-{ij}" for ij in range(shape[1])]
        self._mfs1 = create_sinusoid_memberships(dict(zip(_mf1_names, _centers1)))
        self._mfs2 = create_sinusoid_memberships(dict(zip(_mf2_names, _centers2)))
        self._rules = FuzzyRule()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Each column is an input variable
        return None
