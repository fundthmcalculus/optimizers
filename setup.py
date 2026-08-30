"""Build the optional compiled TSP local-search extension.

Package metadata lives in ``pyproject.toml``; this file only declares the Cython
extension modules (2-opt / 3-opt kernels — see CYTHON_ANALYSIS.md).

**This file is now on the packaging path, indirectly.** The build backend is
``hatchling``, which never executes ``setup.py`` on its own. It is reached
because ``hatch_build.py`` -- the wheel build hook added for #132 -- shells out
to::

    python setup.py build_ext --build-lib <tmpdir>

and force-includes whatever lands there. So the flag and compiler logic below
is what every wheel is compiled with, not just a developer's ``--inplace`` run.

That force-include is load-bearing: hatchling's file selection honours
``.gitignore``, which excludes ``*.so`` and ``*.py[codz]`` (that last pattern
matches ``.pyd``), so compiled artifacts are stripped from a wheel even when
they exist in ``src/``. Before the hook, ``python -m build`` emitted
``py3-none-any`` even immediately after a successful ``build_ext --inplace``,
and ``HAS_CYTHON`` was ``False`` in every installed copy.

The direct developer invocation still works and is what the `test` CI job
uses::

    python setup.py build_ext --inplace

The extensions are **optional**: if one can't be compiled the build emits a
warning and continues, and the library falls back to the numba kernels at import
time (see ``combinatorial/strategy.py`` and ``benchmarks/cython_kernels.py``).
That covers an unsupported toolchain, an ordinary compile error, and a missing
Cython — without Cython the ``.pyx`` reaches the compiler untranslated and
``object_filenames()`` raises ``UnknownFileType``, a ``CCompilerError`` subclass,
which ``optional=True`` swallows. It does **not** cover the total absence of a
compiler on the mingw32 path, where ``build_ext`` dies in the compiler
constructor's version probe before ``optional`` is ever consulted.

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
except ImportError:
    # Reachable: this file is only ever run by hand or by CI, so whether Cython
    # is importable depends on the caller's environment, not on
    # build-system.requires (which provisions hatchling's isolated env, not this
    # one). CI installs Cython explicitly before invoking build_ext.
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
    the GCC/Clang spelling, which is what the mingw32 and cygwin cases want. That
    is a spelling choice only; no claim is made that ``zos``/xlc actually builds.

    The macOS case is **unchanged in behaviour** by the move to compiler_type,
    and is worth naming as a known limitation rather than a fix: a Homebrew gcc
    or clang-with-libomp on macOS reports ``compiler_type == 'unix'``, but the
    platform test below still strips OpenMP, so those users silently lose
    parallel ``prange`` exactly as they did before. Deciding it properly means
    test-compiling a trivial ``#include <omp.h>`` translation unit, which is what
    numpy and scikit-learn do and what this should eventually become.
    """
    if compiler_type == "msvc":
        return ["/O2", "/openmp"], []
    if platform.system() == "Darwin":
        return ["-O3"], []
    return ["-O3", "-fopenmp"], ["-fopenmp"]


class build_ext_opts(build_ext):
    """Append optimization flags once ``self.compiler`` exists.

    ``build_ext.run()`` selects and configures the compiler before calling
    ``build_extensions()``, so this is the first point at which the toolchain is
    actually known. Flags are **appended**, not assigned, so anything set on the
    Extension itself is preserved. (``CFLAGS`` was already honoured before this
    change, by distutils' own ``customize_compiler()``; what is new here is
    preserving Extension-level args.)

    ``self.compiler`` is a *string* — the ``--compiler=`` option — or ``None``
    until ``run()`` replaces it with a compiler object, so the lookup below is
    defensive against anything that calls ``build_extensions()`` on its own.
    """

    def build_extensions(self) -> None:
        cargs, largs = _flags_for(getattr(self.compiler, "compiler_type", ""))
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
    # Keep this ahead-of-time cythonize(). setuptools' build_ext still inherits
    # from Cython's when Cython is importable, and Cython's build_extension()
    # re-cythonizes per extension using its OWN cython_directives (empty by
    # default), not the language_level below. Because the sources are already
    # .c by then, that re-run is a no-op -- which is exactly why this call has
    # to stay.
    ext_modules = cythonize(
        _extensions,
        compiler_directives={"language_level": "3"},
    )
    # cythonize() does not hand back the Extension objects it was given. It
    # rebuilds each one from Cython.Build.Dependencies.distutils_settings, a
    # fixed allowlist of distutils keys that has no `optional` entry -- only
    # `py_limited_api` is forwarded specially. So the flag set above is silently
    # dropped: measured True in, False out, and a different object id, while
    # extra_compile_args on the same objects survives untouched. Re-stamp it, or
    # every compile failure aborts the build the docstring says it should
    # survive.
    for _ext in ext_modules:
        _ext.optional = True
else:
    ext_modules = _extensions

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext_opts},
    # Stated rather than inferred. Without it the src/ layout is recovered only
    # by setuptools' auto-discovery, which runs because `packages`/`py_modules`
    # are absent -- and if that heuristic ever stops firing, `--inplace` writes
    # the .pyd to the repo root instead of src/optimizers/..., where nothing on
    # PYTHONPATH=./src can find it. The resulting ImportError reads exactly like
    # a compile failure, which is a bad half-hour to hand someone.
    package_dir={"": "src"},
)
