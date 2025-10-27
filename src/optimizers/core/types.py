import typing

from numpy.typing import NDArray
import numpy as np

# Type shorthand:
i64 = np.int64
f64 = np.float64
i32 = np.int32
f32 = np.float32
i16 = np.int16
f16 = np.float16
b8 = np.bool_
af64 = NDArray[f64]
af32 = NDArray[f32]
af16 = NDArray[f16]
ai64 = NDArray[i64]
ai32 = NDArray[i32]
ai16 = NDArray[i16]
ab8 = NDArray[b8]
T = typing.TypeVar("T")

AF = typing.Union[af64, af32, af16]
AI = typing.Union[ai64, ai32, ai16]
F = typing.Union[f64, f32, f16]
I = typing.Union[i64, i32, i16]
B = typing.Union[b8]
