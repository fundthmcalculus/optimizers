import typing

from numpy.typing import NDArray
import numpy as np

# Concrete dtypes, for the places that genuinely pin one.
i64 = np.int64
f64 = np.float64
i32 = np.int32
f32 = np.float32
i16 = np.int16
f16 = np.float16
b8 = np.bool_
af64 = NDArray[f64]
ai64 = NDArray[i64]
ab8 = NDArray[b8]
T = typing.TypeVar("T")

# "Some float", "some integer" -- for the code that does not care about width.
#
# These were unions of the three concrete widths (float64|float32|float16 and
# int64|int32|int16). That spelling was both too narrow and too strict: it
# omitted longdouble, and a union of concrete instantiations demands an exact
# member match, so `np.float64` results routinely failed to satisfy `F` even
# though float64 is obviously a float. numpy already publishes the abstract
# hierarchy for exactly this, and `dtype` is covariant, so `af64` satisfies `AF`
# the way it always should have.
#
# `signedinteger`, not `integer`: the union was signed-only, and the compiled TSP
# kernels take `int64` buffers, so an unsigned array reaching them is a runtime
# failure the checker should still catch. Costs nothing -- both spellings measure
# the same error count.
AF = NDArray[np.floating[typing.Any]]
AI = NDArray[np.signedinteger[typing.Any]]
F = np.floating[typing.Any]
I = np.signedinteger[typing.Any]  # noqa: E741
B = b8
