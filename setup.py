"""Build the optional compiled TSP local-search extension.

Package metadata lives in ``pyproject.toml``; this file only declares the Cython
extension modules (2-opt / 3-opt kernels — see CYTHON_ANALYSIS.md).

The extensions are **optional**: if one can't be compiled (no C compiler,
missing Cython, unsupported toolchain) the build emits a warning and continues,
and the library falls back to the numba kernels at import time (see
``combinatorial/strategy.py`` and ``benchmarks/cython_kernels.py``). So
``pip install .`` never hard-fails on a plain source checkout.

Two things below are less obvious than they look, and both were wrong before:

1. ``optional=True`` has to be re-stamped *after* ``cythonize()``, which does not
   return the Extension objects it was given.
2. Optimization flags are chosen once the compiler is actually known, not from
   ``platform.system()``. Those are two different questions and the old code
   answered both with the platform.
"""

import platform

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

try:
    from Cython.Build import cythonize
except ImportError:  # pragma: no cover - PEP 517 build pulls Cython in via pyproject
    cythonize = None


def _flags_for(compiler_type: str) -> tuple[list[str], list[str]]:
    """Return ``(extra_compile_args, extra_link_args)`` for the real compiler.

    There are two independent axes here, and branching on ``platform.system()``
    alone conflates them:

    * Flag **spelling** follows the *compiler*. MSVC wants ``/O2 /openmp``
      (``-O3`` is a GCC/Clang flag it ignores with a warning, and ``/openmp``
      needs no separate link flag). Everything else wants ``-O3 -fopenmp``.
      Choosing by platform meant a gcc toolchain on Windows — mingw-w64, or any
      ``[build_ext] compiler = mingw32`` — was handed ``/O2 /openmp``, which gcc
      reads as input filenames and fails on:
      ``gcc: error: /O2: linker input file not found``.

    * OpenMP **availability** follows the *platform*. Apple's stock clang ships
      no libomp, so on macOS the ``prange`` loops are built serially rather than
      failing on an unrecognized ``-fopenmp``. That is a platform question, and
      stays one.

    ``compiler_type`` is distutils' own name for the toolchain — ``msvc``,
    ``unix``, ``mingw32``, ``cygwin``, ``zos``. Anything that is not MSVC takes
    the GCC/Clang spelling, which is what the mingw32 and cygwin cases want.
    """
    if compiler_type == "msvc":
        return ["/O2", "/openmp"], []
    if platform.system() == "Darwin":
        # Users with libomp installed can still inject flags via CFLAGS/LDFLAGS;
        # build_extensions below appends to whatever is already on the Extension
        # rather than replacing it, so those survive.
        return ["-O3"], []
    return ["-O3", "-fopenmp"], ["-fopenmp"]


class build_ext_opts(build_ext):
    """Append optimization flags once ``self.compiler`` exists.

    ``build_ext.run()`` selects and configures the compiler before calling
    ``build_extensions()``, so this is the first point at which the toolchain is
    actually known. Flags are **appended**, not assigned, so anything set on the
    Extension itself or injected through ``CFLAGS``/``LDFLAGS`` is preserved.
    """

    def build_extensions(self) -> None:
        cargs, largs = _flags_for(self.compiler.compiler_type)
        for ext in self.extensions:
            if getattr(ext, "_opt_flags_applied", False):
                continue
            ext.extra_compile_args = list(ext.extra_compile_args or []) + cargs
            ext.extra_link_args = list(ext.extra_link_args or []) + largs
            ext._opt_flags_applied = True
        super().build_extensions()


_extensions = [
    Extension(
        "optimizers.combinatorial._tsp_cython",
        ["src/optimizers/combinatorial/_tsp_cython.pyx"],
        # A compile failure degrades to the numba fallback instead of failing
        # install. Re-stamped after cythonize() below, which drops it.
        optional=True,
    ),
    Extension(
        "optimizers.benchmarks._bench_cython",
        ["src/optimizers/benchmarks/_bench_cython.pyx"],
        # A compile failure degrades to the pure-NumPy batch functions instead
        # of failing install (see benchmarks/cython_kernels.py).
        optional=True,
    ),
]

if cythonize is not None:
    ext_modules = cythonize(
        _extensions,
        compiler_directives={"language_level": "3"},
    )
    # cythonize() does not hand back the Extension objects it was given. It
    # rebuilds each one from Cython.Build.Dependencies.distutils_settings, a
    # fixed allowlist of distutils keys that has no `optional` entry -- only
    # `py_limited_api` is forwarded specially. So the flags set above are
    # silently dropped: measured True in, False out, and a different object id.
    #
    # Because Cython is in build-system.requires it is never None, which makes
    # the `else` branch unreachable in any PEP 517 build -- so until now the
    # graceful degradation this module's docstring promises had never once
    # happened. Re-stamp it.
    for _ext in ext_modules:
        _ext.optional = True
else:  # pragma: no cover - only reachable outside a PEP 517 build
    ext_modules = _extensions

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext_opts})
