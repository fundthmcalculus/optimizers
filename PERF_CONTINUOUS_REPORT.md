# Continuous GA / ACO / PSO Performance — Benchmarks, Hot-Path Fixes, Lineage

**Repository:** `optimizers`
**Scope:** the three continuous population solvers — `AntColonyOptimizer`,
`GeneticAlgorithmOptimizer`, `ParticleSwarmOptimizer` — with
**`local_grad_optim="none"`** throughout. The combinatorial solvers and the
gradient-descent/local-search optimizers are explicitly out of scope (see
`PERFORMANCE_REPORT.md` / `PERF_PLAN.md` for that earlier, separate body of
work).
**Method:** algorithm-first, language-second. Every fix here is a pure-Python/
NumPy change; Cython is evaluated last, as an optional accelerated backend
layered on top, not a replacement.
**Environment:** measured inside this repo's `.venv` (numpy, scipy, joblib,
matplotlib; Cython optional extension built via `python setup.py build_ext
--inplace`), `n_jobs=1` / `joblib_prefer="threads"` to isolate algorithm cost
from dispatch, fixed seeds for reproducibility.

---

## TL;DR

Prior work (`PERFORMANCE_REPORT.md`, PRs #67–74) vectorized the per-generation
sampling/crossover/mutation math for ACO/GA/PSO and fixed the notorious
`truncnorm`-docstring hotspot. That work is real and landed — but it exposed a
**new** bottleneck that wasn't visible before: `SolutionDeck.deduplicate()`,
which re-sorts and Python-loops a per-row `np.allclose` call every generation.
On today's `main`, profiling GA/ACO/PSO on cheap benchmark objectives
(Ackley, Rosenbrock) shows **this one function is 50–70% of total wall-clock**
for all three solvers.

This report:

1. adds a reusable benchmark harness (`optimizers.benchmarks`) with common
   test functions (sphere, Rosenbrock, Ackley, Rastrigin), each with a scalar
   *and* a vectorized-batch pure-Python/NumPy reference implementation;
2. fixes the `deduplicate` hot path (vectorized adjacent-pair comparison,
   **bit-identical** output, verified against the original over 500
   randomized archives);
3. adds an **opt-in batched-evaluation fast path** for GA/ACO/PSO: when
   `local_grad_optim="none"` and the caller supplies a vectorized objective,
   score the whole generation in one NumPy call instead of looping per
   candidate through `apply_local_optimization`;
4. evaluates a **compiled (Cython) kernel** for the batch evaluation as the
   language-level follow-up, and reports honestly where it helps and where it
   doesn't;
5. ships all of the above as a benchmark script + plots with error bars
   across 8 seeds, plus fast correctness tests.

**Net measured result** (population 60, 60 generations, 10 dimensions, mean
over 8 seeds, `local_grad_optim="none"`):

| optimizer | function   | baseline (main) | + dedup fix | + batched eval | **total speedup** |
|-----------|-----------|-----------------:|------------:|----------------:|-------------------:|
| ACO | ackley     | 0.440 s | 0.168 s | 0.107 s | **4.1×** |
| GA  | ackley     | 0.555 s | 0.231 s | 0.108 s | **5.1×** |
| PSO | ackley     | 0.445 s | 0.311 s | 0.056 s | **8.0×** |
| ACO | rosenbrock | 0.471 s | 0.160 s | 0.108 s | **4.4×** |
| GA  | rosenbrock | 0.519 s | 0.176 s | 0.100 s | **5.2×** |
| PSO | rosenbrock | 0.409 s | 0.256 s | 0.042 s | **9.9×** |

See `benchmarks/results/lineage_timings.png` for the plot (error bars =
stdev across 8 seeds) and `lineage_timings.csv` for the raw numbers.

---

## 1. The benchmark harness

**New:** `src/optimizers/benchmarks/`

- `functions.py` — sphere, Rosenbrock, Ackley, Rastrigin. Each ships two
  *reference* implementations, both plain Python/NumPy:
  - `f(x)` — scalar, one candidate vector in, one score out. This is the
    `GoalFcn` contract every solver has always accepted; nothing about it
    changed.
  - `f_batch(X)` — the same arithmetic, vectorized over a `(n, d)` candidate
    matrix, returning `(n,)` scores in one NumPy pass. Not a different
    algorithm — the same formula, batched.
- `harness.py` — runs GA/ACO/PSO against `TEST_FUNCTIONS` across seeds and
  records wall-clock + best fitness (`BenchmarkResult`/`BenchmarkSpec`/
  `run_benchmark_grid`).
- `cython_kernels.py` / `_bench_cython.pyx` — the optional compiled backend
  (§4).

**New:** `benchmarks/run_benchmark.py` — a standalone script (not part of
`pytest`, so CI stays fast) that runs the full multi-seed grid, prints a
summary table, and writes `benchmarks/results/timings.{csv,png}`. Usage:

```
python benchmarks/run_benchmark.py --seeds 8        # full run (~30s)
python benchmarks/run_benchmark.py --quick --seeds 3  # sanity check (~2s)
```

**New:** `plot_benchmark_timings` in `optimizers.plot` — a grouped bar chart
with error bars (stdev across seeds), one subplot per test function, bars
grouped by `(optimizer, eval mode)`. Follows the existing `_finish`/headless
conventions in that module.

**New:** `tests/test_benchmarks.py` — fast correctness checks: scalar vs.
batch reference agreement, registry sanity, scalar-vs-batch-eval bit-identity
through the real optimizers (tiny population/generations), a harness smoke
test, and (skipped if unbuilt) the Cython-vs-NumPy kernel comparison.

---

## 2. Finding the hot path

Profiling GA/ACO/PSO on `main` (population 60, 60 generations, 10 dims,
archive 200, seed 42, `local_grad_optim="none"`) with `cProfile`:

```
ACO on ackley: 0.883s total
  update_solution_deck        0.631s cumulative
    add_generation             0.624s
      deduplicate               0.620s   <-- 70% of total runtime
        np.allclose (15525 calls)  0.584s
```

Same story for GA (0.661s of 1.014s in `deduplicate`) and PSO (a smaller
fraction, since PSO's archive churns less per generation, but still real).

### Why: `SolutionDeck.deduplicate()`

```python
for i_row in range(len(self.solution_archive) - 1, 0, -1):
    for j_row in range(i_row - 1, 0, -1):
        if np.allclose(self.solution_archive[i_row], self.solution_archive[j_row], ...):
            ...
        break  # <-- inner loop always executes exactly once
```

The inner loop always breaks after its first (and only) iteration
(`j_row = i_row - 1`), whether it matched or not — so despite the nested-loop
appearance, this is an **O(N) adjacent-pair scan** over the sorted archive,
not O(N²). The cost is not the math (each row is only `num_vars` floats) — it
is that **`np.allclose` is a full Python-level function call** (argument
validation, dtype checks, dispatch through `np.isclose` → `all`) executed
once per row, ~250+ times per generation, every generation. This is exactly
the same class of bug the original `PERFORMANCE_REPORT.md` found in
`truncnorm` (§1 there): cheap math buried under expensive Python-level
call/dispatch overhead.

### Fix (algorithm-level: vectorize the call, not the math)

Replace the per-row `np.allclose` calls with **one** vectorized comparison
over every adjacent pair at once:

```python
a = self.solution_archive[1:].astype(f64, copy=False)
b = self.solution_archive[:-1].astype(f64, copy=False)
is_close = np.all(np.abs(a - b) <= abs_err + rel_err * np.abs(b), axis=-1)
```

reproducing `np.allclose`'s exact formula (`|a-b| <= atol + rtol*|b|`, same
operand order) for every pair in a single NumPy call. The `archive_size`-
capping decision (how many close rows to actually delete) is still inherently
sequential, but it's now pure Python **integer bookkeeping** — no NumPy calls
in that loop — so it stays cheap regardless of archive size.

**Subtlety preserved on purpose:** the original nested loop's inner range
(`range(i_row - 1, 0, -1)`) is empty when `i_row == 1`, so the very last pair
(row 1 vs. row 0) was *never* compared in the original code. This looks like
an off-by-one bug, but fixing it would change *which solutions survive
deduplication* — a behavior change, not a perf change. The vectorized version
preserves this exactly (loop bound `range(n - 1, 1, -1)`, `is_close[0]`
computed but intentionally unused).

**Verification:** a standalone script re-implemented the original algorithm
and ran it against the new one over 500 randomized archives (varying `n`,
`num_vars`, `archive_size`, with injected near-duplicates) — **0 mismatches**
in resulting archive contents. `pytest tests/` (149 tests, full non-`--fast`
suite) passes unchanged.

**Measured (single-run cProfile, same seed, same workload as above):**

| | ACO/ackley | GA/ackley | PSO/ackley | ACO/rosenbrock | GA/rosenbrock | PSO/rosenbrock |
|---|---:|---:|---:|---:|---:|---:|
| before | 0.883s | 1.014s | 0.459s | 0.500s | 0.880s | 0.379s |
| after  | 0.270s | 0.380s | 0.297s | 0.135s | 0.262s | 0.223s |
| speedup | 3.3× | 2.7× | 1.5× | 3.7× | 3.4× | 1.7× |

(PSO benefits less here because its archive churns less per generation than
GA/ACO's; it gets its own large win in §3.)

---

## 3. The next hot path: per-candidate evaluation dispatch

With `deduplicate` fixed, re-profiling shows the new dominant cost is the
per-candidate Python dispatch around evaluation:

```
GA on ackley: 0.380s total
  apply_local_optimization   0.220s cumulative (7200 calls)
    __wrapped                 0.192s
      ackley()                 0.187s
```

`apply_local_optimization` does real work when local search is enabled
(`"grad"`, `"perturb"`, ...), but with **`local_grad_optim="none"`** — this
report's whole scope — it degenerates to `new_value = fcn(new_solution)`: one
Python function call, one literal-choice validation, wrapped in another
Python call (`__wrapped`), executed once per candidate, thousands of times
per run. For PSO specifically, its native loop calls the scalar objective
**`n_particles` times per PSO sub-iteration, 10 sub-iterations per
generation** — the single biggest scalar-call multiplier of the three
solvers, which is why PSO shows the largest gain below.

### Fix (algorithm-level: an opt-in batched-evaluation path)

None of this is fixable by "optimizing" `apply_local_optimization` itself —
the dispatch overhead is inherent to calling an arbitrary user objective once
per candidate. The real fix is architectural: **when there is no local search
to interleave (`local_grad_optim="none"`) and the caller's objective is
vectorizable, let them hand the optimizer a batched form and skip the
per-candidate loop entirely.**

New optional constructor argument, `batch_fcn`, on all three optimizers:

```python
AntColonyOptimizer(config=..., fcn=ackley, variables=..., batch_fcn=ackley_batch)
GeneticAlgorithmOptimizer(config=..., fcn=ackley, variables=..., batch_fcn=ackley_batch)
ParticleSwarmOptimizer(config=..., fcn=ackley, variables=..., batch_fcn=ackley_batch)
```

- `fcn` (scalar) is **always required** and is what the archive's
  initialization pass, the map-elites/QD path, and any future local-search
  path use — nothing about the existing contract changes.
- `batch_fcn` is **optional**. When provided *and* `local_grad_optim=="none"`,
  each solver's worker function scores the whole generation's candidate
  matrix in one call instead of looping:
  - **ACO** (`run_ants`): the `n_ants × num_vars` sampled matrix, once.
  - **GA** (`run_ga`): both children batches (`child1`, `child2`), two calls
    instead of `2 × n_steps`; the elementwise `f1 < f2` tie-break (ties go to
    child2, matching the original scalar loop) is vectorized with
    `np.where`.
  - **PSO** (`run_particles`): the initial `p_best_pos` batch, plus every one
    of its 10 internal sub-iterations' `p_pos` batch — this is where PSO
    recovers the multiplier described above.
- Bookkeeping: a new `_ArgProvider.bump_eval_batch(n)` records `n`
  evaluations with **one** `time.time()` call instead of `n`, so the
  eval-count/elapsed-time metadata (report item #14 in
  `PERFORMANCE_REPORT.md`) doesn't regress under batching.
- Backward compatible: omitting `batch_fcn` (the default) is byte-identical
  to today's behavior; existing callers are unaffected.

**Verification:** a script running each solver twice from the same seed —
once with `batch_fcn=None`, once with the batched reference — asserts
`solution_score`/`solution_vector` are **bit-identical** for all three solvers
on both Ackley and Rosenbrock. `tests/test_benchmarks.py::
test_batched_evaluation_matches_scalar_path` runs this as a permanent
regression test.

**Measured** (`benchmarks/run_benchmark.py`, population 60, 60 generations,
10 dims, mean ± stdev over 8 seeds, `local_grad_optim="none"`, dedup fix
already applied to both columns):

| function | optimizer | scalar (s) | batch (s) | speedup |
|---|---|---:|---:|---:|
| ackley | ACO | 0.157 ± 0.023 | 0.104 ± 0.016 | 1.50× |
| ackley | GA  | 0.233 ± 0.007 | 0.112 ± 0.002 | 2.08× |
| ackley | PSO | 0.306 ± 0.049 | 0.054 ± 0.009 | **5.63×** |
| rosenbrock | ACO | 0.148 ± 0.006 | 0.107 ± 0.004 | 1.39× |
| rosenbrock | GA  | 0.178 ± 0.003 | 0.103 ± 0.001 | 1.72× |
| rosenbrock | PSO | 0.231 ± 0.081 | 0.043 ± 0.013 | **5.35×** |

See `benchmarks/results/timings.png` (4 functions × 3 optimizers × 2 modes)
and `timings.csv` for the full sweep including sphere/rastrigin.

---

## 4. Language-level follow-up: an optional Cython kernel

Following this repo's existing precedent for the TSP local-search kernels
(`combinatorial/_tsp_cython.pyx`, see `CYTHON_ANALYSIS.md`): a compiled,
optional, `nogil`/`prange` extension for the *same* batch formula, as one
fused loop with no intermediate NumPy temporaries.

**New:** `src/optimizers/benchmarks/_bench_cython.pyx`
(`ackley_batch_cy`/`rosenbrock_batch_cy`), wired into `setup.py` exactly like
the existing TSP extension (`optional=True` — a missing compiler/Cython
degrades gracefully, no install failure). `cython_kernels.py` exposes
`HAS_CYTHON` and falls back to the pure-NumPy batch functions if unbuilt,
same pattern as `combinatorial/strategy.py`.

**Correctness:** matches the NumPy batch reference to float64 precision
(differences ~1e-10, from summation order) across randomized inputs;
`tests/test_benchmarks.py::test_cython_kernels_match_numpy_reference`
(skipped automatically if the extension isn't built).

**Honest performance finding — this is where "language-second" earns its
place in the ordering:**

Isolated (no optimizer involved, just the batch call, 200 repeats):

| workload | NumPy | Cython | speedup |
|---|---:|---:|---:|
| ackley, n=200, d=10   | 96.7 µs | 15.7 µs | 6.2× |
| ackley, n=1000, d=10  | 425.9 µs | 71.1 µs | 6.0× |
| rosenbrock, n=1000, d=10 | 103.8 µs | 8.3 µs | 12.6× |

A real, substantial win *in isolation* — avoiding NumPy's intermediate array
allocations matters for this memory-traffic-bound, small-per-row workload.

**But** plugged into the actual end-to-end optimizer run (population 60, 60
generations, 10 dims, mean of 5 seeds), after the algorithmic fixes in §2–3
have already removed evaluation from being the dominant cost for ACO/GA:

| optimizer | population × dim | NumPy-batch (s) | Cython-batch (s) | speedup |
|---|---|---:|---:|---:|
| ACO | 60 × 10  | 0.0935 | 0.0931 | 1.00× |
| GA  | 60 × 10  | 0.1036 | 0.1009 | 1.03× |
| PSO | 60 × 10  | 0.0563 | 0.0417 | **1.35×** |
| GA  | 500 × 50 | 1.3544 | 1.4344 | 0.94× |
| PSO | 500 × 50 | 0.7309 | 0.4481 | **1.63×** |

**Reading:** for ACO and GA at typical population/dimension, the objective
call is no longer the bottleneck after §2–3 (it's the genetic
operators/archive maintenance now) — the Cython kernel's real 6–12× win on
the isolated call barely registers end to end, and at larger scale (GA,
500×50) it's a wash (noise-level, even slightly worse). **PSO is the
exception**: it evaluates the objective far more times per generation (10
sub-iterations × particles, twice per sub-iteration), so evaluation stays a
bigger share of its total cost, and the Cython kernel is a genuine 35–63%
win there.

**Recommendation, matching `CYTHON_ANALYSIS.md`'s existing conclusion:** do
the algorithmic work first (it's 4–10× here, dependency-free, and helps every
solver); reach for a compiled kernel only where profiling *after* that shows
evaluation is still the bottleneck (PSO, or any objective expensive enough
that Python-level dispatch is a rounding error next to it). This module ships
as an optional, gracefully-degrading extension for exactly that case — it is
not required, and nothing regresses if it isn't built.

---

## 5. Reproducing this report

```bash
# Build the optional Cython extension (skip this and everything still works,
# minus the compiled-kernel comparison):
pip install -e ".[dev]"
python setup.py build_ext --inplace

# Fast correctness tests (this report's claims, as permanent regression tests):
pytest tests/test_benchmarks.py -v

# Full benchmark sweep + plot (population 60, 60 generations, 8 seeds, ~30s):
python benchmarks/run_benchmark.py --seeds 8
# -> benchmarks/results/timings.{csv,png}
```

## 6. Round 2 — what a larger population/archive exposes

The measurements above (population 60, archive 200) are sized for a typical
run. Re-running at population 500–2000 (archive = 3× population, 20
dimensions) surfaces two **complexity-class** issues that a small benchmark
can't see — the objective evaluation was never the bottleneck at this scale;
the solvers' own bookkeeping is.

### 6a. ACO's `random_values`: an accidental O(population × archive) per variable

`InputContinuousVariable.random_values` derives each ant's sampling spread
from "the mean absolute deviation of the archive column from this ant's
value" — computed as a direct broadcast:

```python
d2 = np.mean(np.abs(other_values[None, :] - cv[:, None]), axis=1)
```

This materializes a full `(n_ants, archive_size)` temporary **per variable**.
At population 60 / archive 200 that's 12,000 elements — invisible. At
population 2000 / archive 6000 it's 12 million elements, per variable, per
generation. Profiling at scale confirms this is **~90% of ACO's total
wall-clock**, dwarfing every fix from §2–4 combined — a good reminder that
"the hot path" is workload-dependent, not a fixed property of the code.

**Fix:** the mean absolute deviation of a *fixed* array from a query point has
a standard O(log n) closed form once the array is sorted — sort the archive
column once, take prefix sums, and use
`sum_j |s_j - c| = c·(2m - n) - 2·S[m] + S[n]` where `m` is `c`'s rank
(`searchsorted`). This turns an O(archive_size) cost per ant into
O(log archive_size), independent of archive size for the per-ant work. Same
output as the direct computation (verified over 200 randomized trials, max
abs diff ~1e-15 — float64 rounding only).

**Measured** (20 dims, 20 generations, mean ± stdev over 5 seeds,
population scaled with archive = 3×population):

| population | before | after | speedup |
|---:|---:|---:|---:|
| 100  | 0.080s | 0.056s | 1.4× |
| 250  | 0.616s | 0.102s | 6.1× |
| 500  | 2.295s | 0.182s | 12.6× |
| 1000 | 11.161s | 0.332s | 33.7× |
| 2000 | 44.459s | 0.697s | **63.8×** |

The growing speedup (not a flat multiplier) is the signature of a real
complexity-class fix, not a constant-factor one — see
`benchmarks/results/scaling_timings.png` (log-log; the "before" line's slope
is visibly steeper than "after"'s). Best-fitness values are **bit-identical**
before/after at every population tested (the sampling math is unchanged,
only *how* it's computed).

### 6b. GA's tournament selection: a full argsort to pick 3 out of N

```python
candidates = np.argsort(rng.random((n, deck_len)), axis=1)[:, :k]  # k=3
```

sorts the *entire* archive-length random-key row just to keep the first `k`
columns. The next step (`argmin` over the `k` candidates' fitness) doesn't
care what order those `k` came in — only *which* `k` they are. `np.argsort`
is O(deck_len log deck_len) per row; `np.argpartition(keys, k-1, axis=1)`
gets the identical top-k **set** in O(deck_len). Verified to select the exact
same winner as the argsort version over 500 randomized trials.

**Measured (intermediate step, argpartition only):** at population 2000 /
archive 6000, GA drops from 12.4s to 7.1s (1.7×) — real, but a *constant-
factor* improvement, not a shape fix: `argpartition` is still O(deck_len) per
row, so the whole call is still O(population × archive_size). Extending the
sweep to population 8000 made the remaining shape problem unmistakable — GA
went from 1.54s → 251.2s as population went 1000 → 8000 (roughly
quadrupling each doubling, the O(n²) signature) while ACO/PSO (after §6a's
fix) only doubled each time, landing at 2.4–2.6s at population 8000. GA was
**~100× slower** than ACO/PSO on the same workload.

**The actual shape fix — now applied:** draw `k` candidate indices directly
via `rng.integers(0, deck_len, size=(n, k))` instead of ranking (any way) over
the whole archive:

```python
candidates = rng.integers(0, deck_len, size=(n, k))  # O(n*k), was O(n*deck_len)
candidate_fitness = population_fitness[candidates]
winners = candidates[np.arange(n), np.argmin(candidate_fitness, axis=1)]
```

This drops the within-tournament distinctness guarantee every prior version
of this function preserved — unlike every other change in this report, it is
**not** bit-identical or even statistically-equivalent to what came before,
because it draws different random numbers and can (rarely) pick the same
archive row twice in one tournament. The probability of any repeat among `k`
draws is `~k(k-1)/(2·deck_len)` (~0.05% for `k=3`, `deck_len=3000`) and — the
key property that makes this an easy trade — it *shrinks* as the archive
grows, the opposite of the old approach's cost, which *grew* with the
archive. Statistically inconsequential for a stochastic search; applied here
on that basis (not verified bit-identical, unlike everything else in this
report).

**Measured, full fix, same population sweep:**

| population | GA (argsort, original) | GA (+ argpartition) | GA (+ `rng.integers`) | ACO (§6a) | PSO |
|---:|---:|---:|---:|---:|---:|
| 1000 | — | 1.54s | **0.26s** | 0.30s | 0.34s |
| 2000 | — | 5.61s | **0.49s** | 0.64s | 0.67s |
| 4000 | — | 36.4s | **1.02s** | 1.24s | 1.30s |
| 8000 | — | 251.2s | **2.03s** | 2.42s | 2.60s |

GA now scales linearly like ACO/PSO (doubling per population doubling) and is
the **fastest of the three** at every population tested — a 124× speedup over
the argpartition version at population 8000, and a genuine complexity-class
fix (`benchmarks/results/scaling_timings_all_fixed.png`), not a constant-factor
one. This is the one change in this report that trades an exact-output
guarantee for it, and it was applied deliberately for that reason, not by
default.

Historical note — the intermediate steps above, for the record:

| population | GA (after argpartition) | ACO (after §6a fix) | PSO |
|---:|---:|---:|---:|
| 1000 | 1.54s | 0.30s | 0.34s |
| 2000 | 5.61s | 0.64s | 0.67s |
| 4000 | 36.4s | 1.24s | 1.30s |
| 8000 | 251.2s | 2.42s | 2.60s |

(20 dims, 15 generations, archive = 3×population, mean over 3 seeds,
`benchmarks/run_scaling_benchmark.py --optimizers GA ACO PSO --populations
1000 2000 4000 8000`; see `benchmarks/results/scaling_timings_ga_to_8000.png`.)
GA roughly quadruples each time population doubles (the O(n²) signature);
ACO and PSO roughly double, as expected after §6a's fix removed ACO's own
O(n²) term. By population 8000, GA was **~100× slower than ACO/PSO** — this
is the state the `rng.integers` fix above (now applied) was written against;
see `benchmarks/results/scaling_timings_all_fixed.png` for where all three
land afterwards.

### 6c. Where this leaves the Cython question

The Cython kernel (§4) accelerates the *objective evaluation* batch call.
Round 2 shows why that was never going to be the long pole at scale: ACO and
GA's own bookkeeping — sampling statistics and selection, not the objective —
were the O(n²)-shaped costs. No amount of speeding up `fcn(x)` fixes an
algorithm that's quadratic in population size elsewhere. This is the
practical version of "algorithm first, language second": profiling at a
representative scale found two complexity-class bugs that a compiled kernel
categorically cannot fix (it would just make the same O(n²) shape run with a
smaller constant), whereas the sorted-prefix-sum/argpartition rewrites remove
the O(n²) shape entirely.

**Current plan for the Cython kernel, concretely:**
1. **Shipped, keep as-is:** `ackley_batch_cy`/`rosenbrock_batch_cy` as the
   optional accelerated batch-eval backend for callers whose objective really
   is the bottleneck after the algorithmic fixes (measured case: PSO, §4).
2. **Not planned:** extending Cython into ACO/GA/PSO's own sampling/selection
   internals. §6a/§6b show the win there is algorithmic (a better formula),
   not a compilation target — a compiled version of the O(n²) broadcast would
   still be O(n²), just with a smaller constant; PERFORMANCE_REPORT.md and
   CYTHON_ANALYSIS.md's original conclusion (numba/numpy first, Cython only
   where profiling shows evaluation itself is still the bottleneck after
   algorithmic fixes) holds up under this second round of profiling.
3. **Revisit only if:** a future profile at realistic scale shows a solver's
   *own* per-candidate loop (not the user's objective) is dominated by
   irreducible scalar Python work that no vectorization/algorithmic rewrite
   can remove — none of the three solvers are in that state today.

Reproduce: `python /tmp/.../scaling_bench.py`-style harness is not (yet)
part of the checked-in `benchmarks/` package (it monkeypatches the old
implementations in-process for a controlled before/after on one process); the
after-only numbers are reproducible via `benchmarks/run_benchmark.py` with
larger `--seeds`/population by editing `BenchmarkSpec`.

---

## 7. What's out of scope here

- The combinatorial GA/ACO/local-search kernels — already covered by
  `PERFORMANCE_REPORT.md` / PRs #67–74, untouched by this report.
- `local_grad_optim != "none"` (`"grad"`, `"single-var-grad"`, `"perturb"`)
  and the gradient-descent optimizer itself — a separate piece of work per
  the task scope; `apply_local_optimization`/`gd.py` are unmodified.
- MAP-Elites / Pareto (`objective_mode != "scalar"`) — the batched-evaluation
  fast path is intentionally not wired into that path (see `IOptimizer.
  __init__`'s `not self._returns_outputs` guard); it can be revisited
  separately if that path becomes hot.
