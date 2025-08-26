import typing

from numpy.typing import NDArray
import numpy as np

# Type shorthand:
i64 = np.int64
f64 = np.float64
af64 = NDArray[f64]
ai64 = NDArray[i64]
T = typing.TypeVar("T")
