"""Build the optional compiled TSP local-search extension.

Package metadata lives in ``pyproject.toml``; this file only declares the Cython
extension module (2-opt / 3-opt kernels — see CYTHON_ANALYSIS.md).

The extension is **optional**: if it can't be compiled (no C compiler, missing
Cython, unsupported toolchain) the build emits a warning and continues, and the
library falls back to the numba kernels at import time (see
``combinatorial/strategy.py``). So ``pip install .`` never hard-fails on a plain
source checkout.

OpenMP (``-fopenmp`` / ``/openmp``) powers the parallel ``*_batch`` kernels. It's
applied per platform: Apple's default clang ships no OpenMP, so on macOS the
extension is built serially (the ``prange`` loops degrade to serial) rather than
failing on an unknown ``-fopenmp`` flag.
"""

import platform

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError:  # pragma: no cover - PEP 517 build pulls Cython in via pyproject
    cythonize = None

_system = platform.system()
if _system == "Windows":
    # MSVC: /O2 is its optimize flag (-O3 is a GCC/Clang flag it ignores with a
    # warning); /openmp needs no separate link flag.
    _extra_compile = ["/O2", "/openmp"]
    _extra_link = []
elif _system == "Darwin":
    # Apple's stock clang has no OpenMP support; build serially rather than fail
    # on an unrecognized -fopenmp. (Users with libomp can inject flags via CFLAGS.)
    _extra_compile = ["-O3"]
    _extra_link = []
else:
    _extra_compile = ["-O3", "-fopenmp"]
    _extra_link = ["-fopenmp"]

_extension = Extension(
    "optimizers.combinatorial._tsp_cython",
    ["src/optimizers/combinatorial/_tsp_cython.pyx"],
    extra_compile_args=_extra_compile,
    extra_link_args=_extra_link,
    # A compile failure degrades to the numba fallback instead of failing install.
    optional=True,
)

if cythonize is not None:
    ext_modules = cythonize(
        [_extension],
        compiler_directives={"language_level": "3"},
    )
else:
    ext_modules = [_extension]

setup(ext_modules=ext_modules)
