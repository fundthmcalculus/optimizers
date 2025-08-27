import typing

from numpy.typing import NDArray
import numpy as np

# Type shorthand:
i64 = np.int64
f64 = np.float64
b8 = np.bool_
af64 = NDArray[f64]
ai64 = NDArray[i64]
ab8 = NDArray[b8]
T = typing.TypeVar("T")