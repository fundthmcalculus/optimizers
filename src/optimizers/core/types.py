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

# "An array of floats", "an array of integers" -- for the code that does not
# care about width.
#
# These were unions of the three concrete widths (float64|float32|float16 and
# int64|int32|int16). That spelling was both too narrow and too strict: it
# omitted longdouble, and a union of concrete instantiations demands an exact
# member match, so an `af64` routinely failed to satisfy `AF` even though a
# float64 array is obviously an array of floats. numpy already publishes the
# abstract hierarchy for exactly this, and `dtype` is covariant.
#
# `signedinteger`, not `integer`: the union was signed-only, and the compiled TSP
# kernels take `int64` buffers, so an unsigned array reaching them is a runtime
# failure the checker should still catch.
#
# There are no scalar counterparts. `F` and `I` used to exist and were used for
# scores, rates and tour lengths -- none of which is ever a numpy scalar in this
# codebase. They are `float` and `int`.
AF = NDArray[np.floating[typing.Any]]
AI = NDArray[np.signedinteger[typing.Any]]
B = b8
