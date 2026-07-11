# Operation "Improve Performance A Lot" — Findings & Ranked Action Plan

**Repository:** `optimizers`  **Date:** 2026-07-10
**Target hardware:** Intel Core i7-1185G7 (4C/8T), 16 GB RAM — all benchmarks sized to fit.
**Environment:** measured inside the project `.venv` (numpy 2.4.4, scipy 1.17.1, numba 0.65.1, joblib 1.5.3, Python 3.13).

---

## TL;DR — the headline

Profiling the continuous **ACO** solver (9 vars, pop 50, 25 gens, single worker) shows it spends **7.5 s of 7.9 s** inside `run_ants`, and **~4.8 s of that (≈60% of total runtime) is scipy building *documentation strings* for a `truncnorm` object that is re-created on every single variable sample.** This is pure waste — no math, just `_construct_doc` / `docformat` / `splitlines`.

Replacing the frozen-distribution call with a direct inverse-CDF sample is a **measured 177× speedup on the sampling primitive** and should cut end-to-end ACO time by **~5–8×** with zero behavioral change. This one change (Rank #1) is the biggest single win in the codebase and is a ~15-line edit.

The rest of the report ranks 14 further changes, including the multiprocessing "copy fixed data once" architecture you flagged.

---

## How to read the ranking

Each item is scored on **Impact** (how much wall-clock it saves), **Effort** (dev cost/risk), and **Scope** (which algorithms benefit). Do them roughly top-to-bottom; items 1–4 are high-impact *and* low-effort and should land first.

| # | Change | Impact | Effort | Scope |
|---|--------|:------:|:------:|-------|
| 1 | Kill per-sample `truncnorm` object creation | ★★★★★ | XS | ACO, GA, all continuous |
| 2 | Send fixed data to workers **once** (persistent pool / memmap) | ★★★★☆ | M | ACO/PSO/GA (processes) |
| 3 | Bound the solution archive to `archive_size` | ★★★★☆ | S | all continuous |
| 4 | Reuse the global RNG instead of `np.random.default_rng()` per call | ★★★☆☆ | XS | all continuous |
| 5 | Vectorize ACO `run_ants` across ants & precompute the CDF | ★★★★☆ | M | continuous ACO |
| 6 | Precompute `eta**beta` / `tau**alpha` in combinatorial ACO | ★★★★★ | S | combinatorial ACO/MST |
| 7 | Stop rebuilding the GA archive with `vstack` inside the loop | ★★★★☆ | S | combinatorial GA |
| 8 | Replace FCM's BFGS with closed-form alternating updates | ★★★★☆ | M | clustering (FCM) |
| 9 | `@njit` the iVAT path-max double loop | ★★★★☆ | S | VAT/iVAT |
| 10 | Vectorize `check_path_distance` & `delta_tau` (numba) | ★★★☆☆ | S | combinatorial |
| 11 | Choose threads vs processes correctly + document it | ★★★☆☆ | S | all |
| 12 | Remove redundant `sort()`/`deduplicate()` passes per generation | ★★★☆☆ | S | all continuous |
| 13 | `@njit` the local-search kernels (2-opt / 3-opt / NN) | ★★★☆☆ | M | combinatorial |
| 14 | Trim per-eval overhead in `bump_eval` (`time.time()` every call) | ★★☆☆☆ | XS | all |
| 15 | Vectorize discrete `random_value` (`np.unique` per sample) | ★★★☆☆ | S | discrete vars |

---

## The changes in detail

### 1. Kill per-sample `truncnorm` object creation — the big one
**Files:** `src/optimizers/continuous/variables.py:106-129` (`__get_truncated_normal`, `random_value`)

`InputContinuousVariable.random_value` calls `truncnorm(a, b, loc, scale).rvs()` **once per variable, per ant, per generation**. Each call constructs a *fresh frozen distribution object*, and scipy eagerly builds the object's docstring (`_construct_doc` → `docformat` → millions of `str.splitlines`). The actual random draw is a rounding error next to the string formatting.

**Evidence (measured on your hardware):**
```
truncnorm frozen per-call : 8.697s  (434.9 us/call)
ndtri inverse-CDF per-call: 0.049s  (  2.5 us/call)
SPEEDUP on sampling       : 177x
```
And in the ACO cProfile, `scipy/_distn_infrastructure.py:892(freeze)` + `_construct_doc` + `doccer.docformat` account for ~4.8 s of a 7.9 s run.

**Fix:** sample the truncated normal directly with the inverse CDF, allocating nothing:
```python
from scipy.special import ndtr, ndtri  # module-level import

def _truncated_normal(mean, std, low, high, rng):
    if std <= 0.0:
        std = 1.0
    a = ndtr((low - mean) / std)
    b = ndtr((high - mean) / std)
    u = rng.uniform(a, b)          # draw directly in the truncated CDF range
    return float(mean + std * ndtri(u))
```
Behavior is statistically identical. This also removes the scipy `truncnorm` import from the hot path entirely.

---

### 2. Copy fixed data to workers **once**, not every generation *(your flagged item)*
**Files:** `src/optimizers/continuous/aco.py:114-125`, `pso.py:140-153`, `ga.py:154-166`; `src/optimizers/core/base.py:68-77` (`setup_for_generations`)

Today every generation calls `parallel(joblib.delayed(run_ants)(..., self.variables, self.wrapped_fcn, self.soln_deck.solution_archive))`. With `joblib_prefer="processes"`, joblib **pickles and ships every argument to every worker on every generation** — including the `variables` list, the wrapped goal function (which closes over `args`, i.e. your large fixed dataset), and the archive.

**Evidence:** re-pickling a 16 MB fixed array for a 25-gen × 4-job run costs **0.57 s just in serialization** (5.7 ms per dispatch), before any transport or fitness work. For the large datasets you describe, this scales linearly and dominates.

**Fix — send the immutable data once.** Two complementary approaches, both general:

- **Persistent worker pool + initializer.** Replace the per-generation `joblib.delayed` fan-out with a `loky` reusable executor (or `concurrent.futures.ProcessPoolExecutor`) created once at `solve()` start, with an `initializer` that stashes the fixed payload (`variables`, `args`, goal fn) in a module global on each worker. Per generation you then dispatch only the *changing* data (the current archive, or just its indices). joblib's `loky` backend already keeps workers warm across `parallel()` calls when you reuse a single `Parallel(backend="loky")` instance — the missing piece is not re-sending the constant args.
- **Memory-map large read-only arrays.** For big numpy payloads, hand workers a `np.memmap` (or rely on joblib's `max_nbytes` auto-memmap, which kicks in >1 MB) so the OS shares one physical copy across processes instead of pickling N copies. This is the lowest-risk way to stop replicating "a large section of memory."

Design note: the current code is *structurally* ready for this — the fixed args (`variables`, `wrapped_fcn`) never change during a solve, and only `solution_archive` mutates per generation. Factor the parallel dispatch into a small reusable helper so ACO/PSO/GA share one implementation.

---

### 3. Bound the solution archive to `archive_size`
**Files:** `src/optimizers/solution_deck.py:31-52` (`append`), `140-144` (`sort`); `src/optimizers/continuous/base.py:165-175` (`update_solution_deck`)

`append` does `np.vstack`/`np.hstack` (full reallocation + copy) for every job output, and **the archive is never truncated back down to `archive_size`.** `deduplicate` only removes *near-duplicates*, so with a well-behaved objective the archive grows by ~`population_size` every generation forever.

**Evidence:** an archive configured at **200 grew to 950** after a single early-stopped ACO run (would reach ~1450 over a full 25-gen run). Every `sort()` — called multiple times per generation via `deduplicate()` and `get_best()` — and every CDF / `searchsorted` / `other_values` slice then runs over this ever-growing array, so per-generation cost *increases* as the run proceeds.

**Fix:**
- After merging a generation, `sort()` once and slice `[:archive_size]`. Elitism is preserved (best solutions are kept) and memory/CPU become constant per generation.
- Preallocate the archive as a fixed `(archive_size + max_batch, num_vars)` buffer and write into it, instead of `vstack`/`hstack` reallocating each append.

---

### 4. Reuse the global RNG instead of constructing one per call
**Files:** `src/optimizers/continuous/variables.py:119` and `:135`

`InputContinuousVariable.random_value` and `initial_random_value` call `np.random.default_rng()` **on every invocation**, constructing a fresh bit-generator each time (**~11.6 µs/call measured**) and *ignoring* the seeded global RNG in `core/random.py` — so results aren't reproducible even after `set_seed()`.

**Fix:** use the shared `global_rng()` (already imported elsewhere in the file). Correctness bonus: reproducibility is restored. In worker processes, seed each worker's generator once from a base seed + worker id (via the initializer from #2).

---

### 5. Vectorize ACO `run_ants` across the population
**File:** `src/optimizers/continuous/aco.py:35-70`

`run_ants` loops in pure Python over ants × variables, calling `np.searchsorted` and `variable.random_value` scalar-by-scalar. `cdf(q_weight, len(archive))` is also recomputed on every worker call though it depends only on `q` and archive length.

**Fix:** hoist `cdf` out (compute once per generation, pass in). Draw all base-solution indices for the batch at once (`np.searchsorted(cp_j, rng.uniform(size=n_ants))`). After #1, the per-variable sampling can be vectorized across ants because the truncated-normal draw becomes a pure array op (`ndtri` is vectorized). This turns the inner double loop into a handful of array operations.

---

### 6. Precompute `eta**beta` and `tau**alpha` in combinatorial ACO — ✅ IMPLEMENTED
**Files:** `src/optimizers/combinatorial/aco.py` (`p_xy`, `solve`), `aco_mst.py` (`p_xy`, `solve`)

`p_xy` evaluated `np.power(tau_xy[x,:], alpha) * np.power(eta_xy[x,:], beta)` on **every step of every ant of every generation.** `eta` (desirability, `1/distance`) *never changes*, yet `eta**beta` was recomputed billions of times; `tau` changes only once per generation.

**Done:** `eta_beta = eta**beta` is computed **once** in `solve()`; `tau_alpha = tau**alpha` **once per generation** before the parallel dispatch. `p_xy` is now a single elementwise product. Both `aco.py` and `aco_mst.py` updated — collapses the power cost from O(pop·steps·gens) to O(gens)/O(1). Combined with #10, measured ACO-TSP wall-clock 3.56 s → 2.86 s at N=200.

---

### 7. Stop rebuilding the GA archive with `vstack` inside the results loop — ✅ IMPLEMENTED
**File:** `src/optimizers/combinatorial/ga.py`

For every individual returned every generation, the entire `genome` (archive_size × N) and `genome_value` were reallocated via `np.vstack`/`np.hstack` — O(pop²·N) copying per generation.

**Done:** the generation's offspring are collected into Python lists, then the genome grows with a **single** `np.vstack`/`np.concatenate` before one argsort/truncate. Measured GA-TSP wall-clock 0.50 s → 0.32 s at N=200.

---

### 8. Replace FCM's BFGS with closed-form alternating updates
**File:** `src/cluster/fcm.py:8-40`

`minimize(optim_j_w_c, c.flatten(), method="BFGS")` uses finite-difference gradients, so the objective is called O(n·d) extra times per iteration, and each call rebuilds the full `N×C×D` broadcast tensor. Worse, `_j_w_c` computes the pairwise-difference tensor **twice** (once in `_get_weights` via `np.linalg.norm`, once inline as `np.sum((...)**2)`), taking a sqrt only to square it again; `_get_weights` also materializes an `N×C×C` tensor per call.

**Fix:** fuzzy c-means has the standard closed-form alternating optimization — recompute memberships, then centers `c_j = Σ wᵐ x / Σ wᵐ` — which converges in a few cheap vectorized iterations with no finite-difference explosion. Compute the squared-distance matrix `N×C` **once** per iteration and reuse it for both memberships and objective. This is both faster and more numerically standard.

---

### 9. `@njit` the iVAT path-max double loop
**File:** `src/cluster/mergevat.py:19-26`

The iVAT reordering recursion (`for r in range(1,N): for c in range(r): ...`) is inherently O(N²) but runs as interpreted Python with scalar writes and a per-row `np.argmin` slice. The rest of the module already uses `@njit(cache=True)` well — this loop is the odd one out.

**Fix:** move it into an `@njit(cache=True)` kernel (it's trivially numba-compatible: scalar indexing, `max`, `argmin`). Expect ~50–100× on this stage. **Related cleanups in the same file:** `progress_bar.update(1)` is called inside the inner O(N²) loop (`:65-66`) — update once per outer iteration; `key[vertices]`/`adj[u, vertices]` with `vertices=arange(N)` force full-length copies every Prim step (`:167-169`) — index directly; and `_get_dist` is computed twice in `vat_prim_mst_seq` (`:238` and `:240`). Also verify `heapq` inside the `@njit` `vat_prim_mst` (`:107`) actually compiles in nopython mode — if it silently falls back to object mode the `@njit` is doing nothing; an array-based Prim is faster for dense matrices anyway.

---

### 10. Vectorize `check_path_distance` and `delta_tau` — ✅ IMPLEMENTED
**Files:** `src/optimizers/combinatorial/base.py` (`check_path_distance`), `aco.py` (deposit + sampling)

`check_path_distance` was a scalar Python loop over the tour (called 2× per GA individual and per 2-opt result). Pheromone deposit `delta_tau` was a Python per-edge loop; `run_ant` sampled with `np.random.choice(..., p=p)`.

**Done:**
- Distance: `distances[order[:-1], order[1:]].sum()` + the return-to-start edge, in one gather.
- Deposit: `np.add.at(delta_tau, (order[:-1], order[1:]), q/tour_length)` + the closing edge — the whole tour at once. (`aco_mst`'s deposit is a 2-D `(from,to)` form with distinct semantics and was left as-is.)
- Sampling: `np.searchsorted(np.cumsum(p), np.random.random())` replaces `np.random.choice(..., p=p)`, avoiding its per-call re-validation/re-cumsum and global lock. Same distribution (searchsorted `side='left'` skips zero-probability cities).

---

### 11. Choose threads vs processes correctly — and document it
**Files:** `src/optimizers/core/base.py:106` (`joblib_prefer="threads"` default), all `solve()` methods

The default is `"threads"`. The inner worker bodies (`run_ants`, `run_ga`, ACO tour construction) are **largely pure Python**, so under the GIL threads give little speedup and add dispatch overhead. Threads only win when the fitness function is numpy-vectorized (releases the GIL) or you're on a free-threaded build (`sample.py` already probes `sys._is_gil_enabled()`).

**Guidance to bake in:**
- **CPU-bound pure-Python fitness →** `processes` (with #2 so fixed data isn't re-shipped).
- **numpy-vectorized fitness / free-threaded 3.13t →** `threads` (no pickling, shared memory).
- Document this tradeoff in the config docstring and pick a smarter default (e.g. `processes` when a large `args` payload is detected, else `threads`). Keep it a user-visible knob — it's workload-dependent.

---

### 12. Remove redundant `sort()` / `deduplicate()` passes per generation
**Files:** `src/optimizers/continuous/base.py:165-175` (`update_solution_deck`), `solution_deck.py:97-144`

`update_solution_deck` loops over each job output calling `append` then `deduplicate()` — and `deduplicate()` calls `sort()` internally, so with `n_jobs` outputs you re-sort the (growing, see #3) archive `n_jobs` times per generation, then `get_best()` sorts again.

**Fix:** append **all** job outputs first, then `deduplicate()`/`sort()`/truncate **once** per generation.

---

### 13. `@njit` the local-search kernels — ✅ IMPLEMENTED (2-opt, 3-opt, NN)
**Files:** `src/optimizers/combinatorial/strategy.py` — `_two_opt_kernel`, `_three_opt_kernel` (new `@njit(cache=True)` kernels), `TwoOptTSP.solve`, `ThreeOptTSP.solve`, `NearestNeighborTSP.solve`

These were triple/double-nested Python loops with scalar matrix indexing and, in 3-opt, an `np.zeros(8)` allocated in the innermost O(n³) loop. `NearestNeighborTSP` used a Python `set` membership scan.

**Done:**
- 2-opt / 3-opt: moved into `@njit(cache=True)` kernels (the per-iteration `np.zeros(8)` is gone — the 8 lengths are scalars, and argmin is an unrolled comparison chain). Logic verified **bit-identical** to the original on N=30/80/150 (routes and values match exactly for the well-defined `back_to_start` case, including 3-opt with real `num_iterations`).
- NN: boolean `visited` mask + `np.argmin(np.where(visited, inf, distances[current]))` — `argmin`'s first-min tie-break matches the old strict-`<` first-found, so routes are identical.
- **Measured (numba warm):** 2-opt full-scan N=400 **479 ms → 1.3 ms (~370×)**; full 3-opt (3 iters) at N=500 went from **>120 s (timed out) → 0.63 s**.
- **Bug found & fixed:** the original 3-opt indexed `route[kl+1]` up to `N`, in-bounds only because `back_to_start` appends a depot node; with no appended node it raised `IndexError`. The kernel caps the inner bound by route length (a no-op for the appended case) so njit — which skips bounds checks — cannot read out of bounds.

**Not done:** `ConvexHullTSP` (geometric windmill scan, not a hot path in practice) and `ga.py:_2opt_refine` (a single bounded pass per child — left as-is to keep the GA worker cloudpickle-simple; it benefits indirectly from the vectorized `check_path_distance`).

---

### 14. Trim per-evaluation overhead in `bump_eval`
**File:** `src/optimizers/continuous/base.py:47-58`

Every wrapped fitness call runs `bump_eval`: a `time.time()` syscall plus several dict writes, on *every* objective evaluation. For millions of evals this is measurable pure overhead, and under `processes` the counter is per-worker anyway (so its "global" count is misleading).

**Fix:** make the metadata bookkeeping opt-in (only when something actually consumes `eval_count`/`elapsed_time`), or update it once per generation rather than per evaluation.

---

### 15. Vectorize discrete `random_value`
**File:** `src/optimizers/continuous/variables.py:33-47`

`InputDiscreteVariable.random_value` does `np.concatenate((values, other_values))` then `np.unique(..., return_counts=True)` **per sample** to build a weighting — an O(M log M) sort on every draw.

**Fix:** compute the weighting once per generation (it depends on the archive column, not the individual draw) and reuse it across all ants sampling that variable; or use `np.bincount` on integer-encoded categories.

---

## Suggested rollout

1. **Quick wins first (a day):** #1, #4, #3, #12, #14 — all low-risk, mostly local edits, and together they should cut continuous-optimizer wall-clock several-fold. Run `pytest tests/` after each.
2. **Architecture (the multiprocessing item):** #2 + #11 — factor a shared parallel-dispatch helper with a persistent pool/initializer and memmap for large read-only data. This is the change that pays off most on *your* large-dataset applications.
3. **Vectorization/numba sweep:** #5, #6, #7, #10, #13 — biggest combinatorial gains.
4. **Clustering:** #8, #9 — independent of the optimizer work; do whenever FCM/VAT matters.

## Verification method

Every change should be validated with (a) `pytest tests/` for correctness, and (b) a before/after timing on a fixed seed (`set_seed(42)`) using the existing test objectives (`optim_ackley`, `optim_rosenbrock`) at pop 50 / 25 gens — sized to run comfortably in 16 GB. The profiling harness used for this report samples ACO/PSO/GA under `n_jobs=1` (to isolate algorithm cost from dispatch) and separately under `processes` (to measure dispatch + serialization).

---

## Appendix — measured evidence

**ACO single-worker timing (9 vars, pop 50, 25 gens):** ACO 6.30 s vs PSO 0.71 s vs GA 0.44 s — ACO is ~9–14× slower purely due to #1.

**cProfile top of ACO run (7.96 s total):**
- `run_ants` — 7.55 s cumulative
- `variables.random_value` — 7.43 s → `__get_truncated_normal` 7.05 s
- `scipy freeze/_construct_doc/docformat/splitlines` — **~4.8 s of pure docstring formatting**

**Micro-benchmarks (Core i7-1185G7, inside `.venv`):**
```
truncnorm frozen per-call : 434.9 us/call
ndtri inverse-CDF per-call:   2.5 us/call     -> 177x
np.random.default_rng()   :  11.6 us/call     (constructed per sample today)
re-pickle 16MB fixed data : 5.7 ms per generation-dispatch (100 dispatches = 0.57s)
```

**Archive growth:** configured `archive_size=200` → **950 rows** after one early-stopped run (unbounded growth, see #3).
