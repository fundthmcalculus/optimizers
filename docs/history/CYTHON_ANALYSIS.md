# Analysis: Compiled `.pyx` (Cython) Modules for the Optimizers Library

**Question:** should this library adopt Cython (`.pyx`) compiled extensions for performance?

**Short answer:** For most hot loops, **numba is the better tool** and it is *already a
dependency* — it delivers the same order-of-magnitude speedup with zero build/packaging
cost. Cython earns its keep in a **narrow but real** set of cases: (1) `nogil` kernels that
enable *true* multi-threaded parallelism (directly serving your multi-threading goal),
(2) shipping a compiled artifact so end users don't pay JIT warm-up, and (3) kernels that
mix Python objects with tight numeric loops where numba's nopython mode is awkward. This
document maps where each applies and what the cost is.

---

## 1. Where the time actually goes (recap)

The profiled hot paths are all **tight scalar loops over numpy arrays**:

- Continuous ACO `run_ants` — per-ant, per-variable sampling loop (`continuous/aco.py`).
- Continuous PSO `run_particles` — per-dimension velocity/position update (`continuous/pso.py`).
- Combinatorial tour construction, `check_path_distance`, `delta_tau`, 2-opt/3-opt
  (`combinatorial/base.py`, `strategy.py`, `aco.py`).
- The truncated-normal sampler (addressed in PR2 without any compilation — pure scipy call fix).

None of these need Python object semantics inside the loop; they operate on `float64`/`int64`
arrays. That is precisely the profile where **both** numba and Cython excel, and where the
choice between them is about *tooling*, not capability.

---

## 2. Cython vs. numba for this codebase

| Dimension | numba (`@njit`) | Cython (`.pyx`) |
|-----------|-----------------|-----------------|
| Already a dependency? | **Yes** (`pyproject.toml`) | No — adds a build-time dep |
| Build step / C toolchain | None (JIT at runtime) | **Required** — `cythonize`, C compiler, per-platform wheels |
| Distribution | Pure-Python wheel | Needs compiled wheels per OS/arch (cibuildwheel) or sdist + compiler on user machines |
| First-call cost | JIT warm-up (100 ms–2 s), amortized by `cache=True` | **None** — compiled ahead of time |
| Speedup on scalar array loops | ~10–100× (matches C) | ~10–100× (is C) |
| True multi-thread (`nogil`) | `nopython` + `parallel=True`/`prange` releases GIL | `nogil` blocks + `prange` (OpenMP) — **most explicit control** |
| Debugging | Harder (JIT); good error messages now | Standard C tooling; `cython -a` HTML annotation is excellent |
| Mixed Python-object + numeric | Poor (nopython is strict) | **Good** — can drop to `object` locally |
| Maintenance surface | Low (decorate a function) | Higher (`.pyx` + build config + wheels) |

**Consequence:** numba covers items #5, #6, #7, #10, #13 in the report at essentially no
project cost, and PR4/PR6 will use it. Cython is only worth introducing where it does
something numba can't do as cleanly.

---

## 3. Where Cython specifically wins here

### 3a. `nogil` kernels for real thread-level parallelism (aligns with your multi-threading goal)

Your project defaults to `joblib_prefer="threads"`, but the current worker bodies are pure
Python, so the **GIL serializes them** — threads add overhead without parallelism (report
#11). A Cython kernel marked `nogil` (or a numba `@njit(nogil=True)` kernel) lets several
threads run the *same* compiled inner loop truly concurrently, **without pickling any data**
— which is the cheapest possible answer to "don't replicate the large fixed dataset to each
worker" (report #2). No process pool, no serialization, shared memory by construction.

Illustrative `.pyx` for the ACO per-generation ant batch (schematic):

```cython
# ants.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange

def run_ants_batch(double[:, ::1] archive,       # fixed-ish per generation, shared
                   double[::1] cp_j,             # precomputed CDF
                   double[:, ::1] out,           # (n_ants, n_vars) preallocated
                   int n_ants, int n_vars) nogil:  # <-- no GIL held
    cdef int a, i
    for a in prange(n_ants, schedule='dynamic'):   # OpenMP threads, real parallel
        for i in range(n_vars):
            out[a, i] = _sample_variable(archive, cp_j, i)
```

This is the strongest Cython argument for *this* library: it turns the existing (but
currently ineffective) thread backend into genuine multicore execution with **zero data
copying**, which processes-based parallelism can never fully avoid. numba's
`@njit(parallel=True, nogil=True)` achieves the same and stays dependency-free — so this is
a genuine numba-vs-Cython toss-up, decided by whether you also want 3b/3c.

**Caveat:** the *user's goal function* is still Python and holds the GIL. `nogil`
parallelism only helps the parts of the inner loop that are pure numeric (sampling,
distance, pheromone update, 2-opt deltas). It shines most for combinatorial local search
(2-opt/3-opt, NN) where the whole kernel is numeric and the objective is an array reduction.

### 3b. Ahead-of-time compilation — no JIT warm-up

numba pays a one-time compile per kernel per process. With `processes` backends, **each
worker re-JITs** (unless the `NUMBA_CACHE_DIR` on-disk cache is warm and shared). For
short-lived solves or many small parallel processes, that warm-up can rival the useful work.
Cython artifacts are compiled once at build time and impose **no** per-process cost — a real
edge when you fan out many worker processes for short jobs.

### 3c. Kernels that straddle Python objects and numeric code

`InputVariable` is a Python class hierarchy; the sampling loop calls virtual methods
(`variable.random_value(...)`). numba can't see through that. Two paths:
- **Data-oriented refactor** (preferred, and numba-friendly): flatten variable metadata into
  arrays (`lower[]`, `upper[]`, `is_discrete[]`, category tables) and pass those to a
  compiled kernel. Works for *both* numba and Cython.
- If you want to keep the object model but still compile the hot arithmetic, Cython's ability
  to drop to `object` for the dispatch while keeping `cdef double` locals is more ergonomic
  than fighting numba's nopython constraints.

---

## 4. Concrete `.pyx` candidates, ranked

| Candidate | File today | Why Cython/compiled | numba enough? |
|-----------|-----------|---------------------|:-------------:|
| 2-opt / 3-opt local search | `combinatorial/strategy.py`, `ga.py` | Fully numeric O(n²)/O(n³) scalar loops; ideal `nogil`/`prange` | **numba fine** |
| `check_path_distance`, `delta_tau` | `combinatorial/base.py`, `aco.py` | Tiny numeric kernels called millions of times | **numba fine** |
| ACO ant-batch sampler | `continuous/aco.py` + `variables.py` | Benefits from `nogil` threading + no-copy shared archive | numba fine *after* data-oriented refactor |
| Truncated-normal sampler | `continuous/variables.py` | Hot, but PR2 fixes it with a plain scipy `ndtri` call | **neither needed** |
| PSO velocity/position update | `continuous/pso.py` | Already vectorizable in pure numpy | **plain numpy** |

**Reading:** the only place Cython offers something numba doesn't is the `nogil`/no-copy
threading story (3a) and warm-up elimination (3b) — and those benefit the **combinatorial
local-search kernels** most, because they are 100% numeric.

---

## 5. Build & packaging cost (the real downside)

Adopting Cython means:

- `pyproject.toml` build-system gains `Cython` and `numpy` build requirements; a `setup.py`
  with `cythonize([...])` and `numpy.get_include()`.
- CI must build **per-platform wheels** (`cibuildwheel` across linux/macos/windows × CPython
  versions) or ship an sdist that requires a C compiler on the user's machine — a support
  burden for a library "used heavily for various applications."
- OpenMP (`prange`) adds a `-fopenmp` link flag and a libgomp runtime dependency, which is
  fiddly on macOS (needs `libomp` via brew).
- Source no longer "just runs" from a checkout without a build step; `pip install -e .`
  triggers compilation.

numba imposes **none** of these. That asymmetry is the crux of the recommendation.

---

## 6. Recommendation

1. **Do the numba/numpy work first** (PR4, PR6). It captures the large majority of the
   available speedup with no build/packaging cost, and it's already a dependency.
2. **Hold Cython in reserve for one targeted use case:** if, after PR5, thread-based
   parallelism with a large shared read-only dataset is the dominant deployment pattern,
   introduce a single small compiled extension for the **combinatorial local-search kernels**
   (2-opt/3-opt) as `nogil`/`prange` `.pyx`. Prototype the *same* kernels with
   `@njit(parallel=True, nogil=True, cache=True)` first — if numba hits the target, **skip
   Cython** and avoid the packaging tax entirely.
3. **Enabler regardless of tool:** refactor `InputVariable` metadata into flat arrays
   (`lower`, `upper`, discrete category tables) so the sampling kernel is compilable by
   *either* backend. This is the highest-leverage structural change and is tool-agnostic.
4. **Free-threaded CPython (3.13t):** as no-GIL builds mature, `nogil` compiled kernels
   (Cython *or* numba) become the cleanest path to multicore scaling with zero pickling —
   worth revisiting then. `sample.py` already probes `sys._is_gil_enabled()`, so the codebase
   is forward-looking here.

**Bottom line:** Cython is a *conditional yes* — valuable only for `nogil` combinatorial
kernels under a shared-memory threading deployment, and only if numba's parallel `nogil`
mode proves insufficient in benchmarking. For everything else in this library, numba +
vectorized numpy is the correct, lower-cost choice. The stacked PRs therefore use
numba/numpy; a Cython extension is scoped as an optional follow-up gated on a benchmark.
