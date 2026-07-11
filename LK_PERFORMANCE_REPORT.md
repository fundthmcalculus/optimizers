# Lin-Kernighan — Performance Comparison Report

**Date:** 2026-07-11  **Hardware:** Intel Core i7-1185G7 (4C/8T), 16 GB.
**Scope:** the Lin-Kernighan local search (`LinKernighanTSP`), its two compiled
backends (numba vs the new Cython `.pyx` kernel), and how it compares to the
other TSP local-search heuristics.

All numbers are steady-state (numba warmed) unless noted; distances are Euclidean
on uniform-random points; local searches start from the same nearest-neighbour
tour. Reproduce with `tests/test_tsp_compare.py` and `tests/test_tsp_cython.py`
(`pytest -s`).

---

## 1. What LK adds

`LinKernighanTSP` is a candidate-list variable-neighbourhood local search:
**2-opt reversals + Or-opt relocations** (segment lengths 1–3, forward/reversed),
restricted to each city's *k* nearest neighbours. The extra Or-opt moves let it
escape the reversal-only local optima that trap `TwoOptTSP`/`ThreeOptTSP`.

It has two interchangeable, **bit-identical** compiled backends, selected with
`LinKernighanTSPConfig(local_search_backend="numba" | "cython")`:

- **numba** (default) — JIT kernel `strategy._lk_kernel`.
- **cython** — ahead-of-time `_tsp_cython.lin_kernighan` (`nogil`), plus
  `lin_kernighan_batch` for optimizing many tours over OpenMP threads.

Both produce the *same tour move-for-move* (asserted in the tests), so the choice
is purely about speed/packaging, never quality.

---

## 2. Heuristic quality + speed (NN vs 2-opt vs 3-opt vs LK)

From a common NN start; `gap %` is relative to the best tour found; times exclude
JIT warm-up.

**N = 200**
```
heuristic               length    gap %   time (ms)
Lin-Kernighan          1164.25     0.00        0.63
2-opt                  1238.28     6.36        0.29
3-opt                  1267.03     8.83        0.55
Nearest Neighbor       1369.31    17.61        1.20
```

**N = 500**
```
heuristic               length    gap %   time (ms)
Lin-Kernighan          1759.25     0.00        6.22
2-opt                  1885.40     7.17        4.40
3-opt                  2044.24    16.20        3.82
Nearest Neighbor       2145.89    21.98        6.23
```

**N = 1000**
```
heuristic               length    gap %   time (ms)
Lin-Kernighan          2428.88     0.00       23.80
2-opt                  2577.35     6.11       33.82
3-opt                  2795.91    15.11       15.40
Nearest Neighbor       2989.89    23.10       18.48
```

**Reading:** LK is the shortest tour at every size — **~6–7 % under 2-opt** and
**~9–16 % under 3-opt** — at a comparable (sometimes lower, e.g. N=1000)
runtime, because its candidate lists keep each move search O(*k*) rather than
O(*N*). This is the whole reason to add it for comparison work.

> Caveat: LK here is candidate-restricted (`candidate_k=8`), so it wins *on
> average*, not on literally every instance — full 2-opt can occasionally find a
> far reversal LK's candidates miss.

---

## 3. LK backend: numba vs Cython (single tour, warm)

Representative run (single warmed tour; sub-millisecond timings are noisy, so
treat these as a *range*, not exact figures):

```
   N      numba      cython    speedup
 200   0.4–0.6 ms  0.3–0.5 ms   ~1.1–1.4x
 500   1.0–2.8 ms  0.9–2.2 ms   ~1.1–1.3x
1000   4.8–9.0 ms  4.2–6.5 ms   ~1.1–1.4x
```

**Unlike 2-opt (where Cython was ~2.7–3× faster than warm numba), LK's Cython
edge on a single warm tour is modest — roughly 1.1–1.4× and run-dependent.** LK
is a branchier, heavier kernel — the Or-opt relocation rebuilds an O(*N*) buffer
per accepted move and the candidate scans dominate — so numba's LLVM output and
the Cython C are close at steady state, and the small absolute times make the
ratio noisy. The Cython backend's real, dependable advantages for LK are the next
two sections.

---

## 4. Cython-only win #1 — no JIT warm-up

numba compiles each kernel on its first call in a process (~1–2 s), amortized
only if its on-disk cache is warm and shared. Under a **processes** parallel
backend, or for many short-lived solves, that warm-up is paid repeatedly. The
Cython kernels are compiled at build time and pay **zero** per-process warm-up —
a real edge when fanning out many worker processes for short jobs.

---

## 5. Cython-only win #2 — parallel batch (`nogil` + OpenMP)

`lin_kernighan_batch` optimizes many independent tours across OpenMP threads with
no data copying — impossible for the GIL-bound Python/numba object path.

```
LK batch, 64 tours, N=300 (8 cores):
  sequential singles   115.8 ms
  parallel batch        79.7 ms      speedup 1.45x
```

The ~1.45× (vs ~3× for the 2-opt batch) reflects LK being heavier and more
memory-bound per tour, so it scales less cleanly — but it is still a free,
correctness-preserving speedup for population-style workloads (e.g. refining a
GA/QD archive of tours).

---

## 6. Recommendation

- **Quality:** prefer **LK** over 2-opt/3-opt for tour quality (~6–16 % shorter)
  at similar cost; keep 2-opt/3-opt for the comparison baseline.
- **Backend:** the default **numba** LK is fine for one-off, long-lived solves.
  Choose **`local_search_backend="cython"`** when (a) you spawn many short-lived
  processes (avoids repeated JIT warm-up), or (b) you optimize many tours at once
  (`lin_kernighan_batch` parallelism). For a single warm solve the two are close
  (~1.1–1.4×).
- 2-opt/3-opt see a much larger Cython speedup (~2.7–3×, see `CYTHON_ANALYSIS.md`);
  LK's gain is smaller because its kernel is not a tight reversal-only loop.

*Numbers regenerated from `compare_tsp_heuristics` and the benchmark tests in
`tests/test_tsp_cython.py` / `tests/test_tsp_compare.py`.*
