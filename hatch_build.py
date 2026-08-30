"""Hatchling build hook that compiles the Cython kernels into the wheel.

Without this hook the build backend never executes ``setup.py``, so ``pip
install .``, ``uv sync`` and ``python -m build`` all produced a pure-Python
``py3-none-any`` wheel that shipped ``_tsp_cython.pyx`` as *package data* and no
compiled module. Both ``combinatorial.strategy.HAS_CYTHON`` and
``benchmarks.cython_kernels.HAS_CYTHON`` were therefore ``False`` in every
installed copy, and the kernels existed only for someone working from a source
checkout who had separately run ``setup.py build_ext --inplace``. See issue #132.

Nor could a local build leak into a wheel: hatchling's file selection honours
``.gitignore``, which carries ``*.so`` and ``*.py[codz]`` (that second pattern
matches ``.pyd``), so a compiled artifact sitting in ``src/`` was actively
excluded. The ``force_include`` below is what bypasses that, and it is the
reason a real build hook was the only workable fix.

How it works
------------
``initialize()`` shells out to ``setup.py build_ext``, pointing both
``--build-lib`` and ``--build-temp`` at a temp tree, then force-includes whatever
landed there; ``finalize()`` removes the tree. Reusing ``setup.py`` rather than
reimplementing the compile is deliberate: that file already carries the
compiler-detection and flag-selection logic fixed in #130, and duplicating it
here would give the project two build paths to keep in agreement.

Partial builds are all-or-nothing. If one extension compiles and the other does
not, the default path discards both and emits a pure-Python wheel, because a
platform-tagged wheel carrying an arbitrary subset of the kernels is a confusing
artifact to receive a bug report about. Under ``OPTIMIZERS_REQUIRE_CYTHON`` a
partial build is fatal, so no release can take that path.

Degradation policy
------------------
The compile is **best-effort by default and mandatory on request**, which are
genuinely different needs:

* A source install on a machine with no C compiler must keep working. It
  degrades to the numba / pure-NumPy fallbacks exactly as before, and the wheel
  stays honestly tagged ``py3-none-any``.
* A *release* wheel must never silently ship without the kernels — that is the
  bug this hook exists to fix, and it would be invisible again. Setting
  ``OPTIMIZERS_REQUIRE_CYTHON=1`` turns any failure into a hard error. CI and
  the cibuildwheel release job set it.

``setup.py`` marks both extensions ``optional=True``, so ``build_ext`` exits 0
even when a compile fails. The success test here is therefore **the presence of
the artifacts**, never the exit status.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# Import path -> the package directory the artifact must land in inside the
# wheel. Kept in step with the Extension list in setup.py.
_EXPECTED_MODULES = {
    "_tsp_cython": "optimizers/combinatorial",
    "_bench_cython": "optimizers/benchmarks",
}

_REQUIRE_ENV = "OPTIMIZERS_REQUIRE_CYTHON"
_SKIP_ENV = "OPTIMIZERS_SKIP_CYTHON"


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() not in ("", "0", "false", "no")


class CustomBuildHook(BuildHookInterface):
    """Compile the extensions and force them into the wheel."""

    PLUGIN_NAME = "custom"

    # Set once a compile is attempted; removed in finalize().
    _build_root: Path | None = None

    def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:
        """Remove the temp build tree once the wheel has been written.

        It cannot be cleaned in ``initialize()``: hatchling copies the
        ``force_include`` sources while assembling the archive, which happens
        after ``initialize()`` returns, so deleting earlier would drop the very
        artifacts this hook exists to add.
        """
        if self._build_root is not None:
            shutil.rmtree(self._build_root, ignore_errors=True)
            self._build_root = None

    def initialize(self, version: str, build_data: dict) -> None:
        # sdists carry the .pyx sources and are built on the consumer's machine;
        # compiling here would be both wrong and impossible to tag.
        if self.target_name != "wheel":
            return

        require = _is_truthy(os.environ.get(_REQUIRE_ENV))

        if _is_truthy(os.environ.get(_SKIP_ENV)):
            if require:
                raise RuntimeError(
                    f"{_SKIP_ENV} and {_REQUIRE_ENV} are both set; they contradict."
                )
            self.app.display_warning(
                f"{_SKIP_ENV} is set - building a pure-Python wheel with no compiled kernels."
            )
            return

        try:
            built = self._build_extensions()
        except Exception as exc:  # noqa: BLE001 - re-raised below when required
            if require:
                raise
            self.app.display_warning(
                f"Cython kernels could not be built ({exc}); falling back to a "
                f"pure-Python wheel. Set {_REQUIRE_ENV}=1 to make this fatal."
            )
            return

        missing = sorted(_EXPECTED_MODULES.keys() - built.keys())
        if missing:
            message = (
                f"Cython kernels did not build: {', '.join(missing)}. "
                "The extensions are declared optional=True, so build_ext exits 0 "
                "on a compile failure - re-run with -v to see the compiler error."
            )
            if require:
                raise RuntimeError(message)
            self.app.display_warning(
                f"{message} Falling back to a pure-Python wheel; set "
                f"{_REQUIRE_ENV}=1 to make this fatal."
            )
            return

        # force_include bypasses the .gitignore-driven file selection that
        # otherwise strips every .so/.pyd from the wheel.
        for source, relative in built.values():
            build_data["force_include"][str(source)] = relative

        # The wheel now contains platform-specific binaries, so it must stop
        # claiming Root-Is-Purelib and pick up a real ABI/platform tag
        # (cp313-cp313-manylinux_x86_64 and friends) instead of py3-none-any.
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

    def _build_extensions(self) -> dict[str, tuple[Path, str]]:
        """Run ``setup.py build_ext`` and collect what it produced.

        Returns a mapping of module stem -> (absolute built path, wheel-relative
        destination). A missing entry means that extension failed to compile.
        """
        root = Path(self.root)
        # Both output directories go to a temp tree, so a wheel build leaves
        # nothing behind in the working copy.
        #
        # --build-lib alone is NOT enough: --build-temp defaults to
        # `build/temp.*` *relative to cwd*, which is the repo root, and `build/`
        # is not in .gitignore. Redirecting only the first would leave every
        # `pip install .` with a dirty `git status` -- exactly the breakage #131
        # was filed to remove, arriving through a different door.
        self._build_root = Path(tempfile.mkdtemp(prefix="optimizers-build-ext-"))
        build_lib = self._build_root / "lib"
        build_temp = self._build_root / "temp"

        command = [
            sys.executable,
            str(root / "setup.py"),
            "build_ext",
            "--build-lib",
            str(build_lib),
            "--build-temp",
            str(build_temp),
        ]
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "setup.py build_ext failed with exit code "
                f"{result.returncode}:\n{result.stdout}\n{result.stderr}"
            )

        found: dict[str, tuple[Path, str]] = {}
        for stem, package_dir in _EXPECTED_MODULES.items():
            # The filename carries the interpreter/ABI/platform triple
            # (_tsp_cython.cpython-313-x86_64-linux-gnu.so), so glob rather than
            # reconstruct it.
            matches = [
                path
                for path in build_lib.rglob(f"{stem}*")
                if path.suffix in (".so", ".pyd")
            ]
            if matches:
                artifact = matches[0]
                found[stem] = (artifact, f"{package_dir}/{artifact.name}")
        return found
