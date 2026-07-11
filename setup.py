"""Build the compiled TSP local-search extension.

Package metadata lives in ``pyproject.toml``; this file only declares the Cython
extension module (2-opt / 3-opt kernels — see CYTHON_ANALYSIS.md). OpenMP
(``-fopenmp``) powers the parallel ``*_batch`` kernels; the flags are applied
per-platform so a missing OpenMP toolchain (e.g. bare macOS clang) degrades to a
serial build rather than failing.
"""

import platform

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError:  # pragma: no cover - build environments provide Cython
    cythonize = None

_openmp_compile = ["-fopenmp"]
_openmp_link = ["-fopenmp"]
if platform.system() == "Windows":
    _openmp_compile = ["/openmp"]
    _openmp_link = []

_extension = Extension(
    "optimizers.combinatorial._tsp_cython",
    ["src/optimizers/combinatorial/_tsp_cython.pyx"],
    extra_compile_args=["-O3", *_openmp_compile],
    extra_link_args=[*_openmp_link],
)

if cythonize is not None:
    ext_modules = cythonize(
        [_extension],
        compiler_directives={"language_level": "3"},
    )
else:
    ext_modules = [_extension]

setup(ext_modules=ext_modules)
