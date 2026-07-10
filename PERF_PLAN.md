# Performance Work — Stacked PR Plan

This branch is the base of a **stacked PR chain**. Each branch merges into the one below
it, so review and merge bottom-up. Findings and evidence live in
[`PERFORMANCE_REPORT.md`](./PERFORMANCE_REPORT.md); the Cython/PYX evaluation lives in
[`CYTHON_ANALYSIS.md`](./CYTHON_ANALYSIS.md).

**Scope note:** the `src/cluster/` package (FCM, VAT/iVAT) is explicitly **out of scope** —
it is being split into a separate package. Report items #8 (FCM) and #9 (iVAT) are deferred.

## The stack

| PR | Branch | Merges into | Report items | Risk |
|----|--------|-------------|--------------|:----:|
| 1 | `perf/1-plan-and-cython-analysis` | `main` | Plan + Cython analysis (docs only) | none |
| 2 | `perf/2-continuous-quickwins` | PR1 | #1 truncnorm sampling, #4 RNG reuse, #14 `bump_eval` overhead | low |
| 3 | `perf/3-solution-deck` | PR2 | #3 bound archive to `archive_size`, #12 sort/dedup once per gen | low |
| 4 | `perf/4-aco-vectorize` | PR3 | #5 vectorize `run_ants` + hoist CDF, #15 discrete `random_value` | med |
| 5 | `perf/5-parallel-arch` | PR4 | #2 send fixed data to workers once, #11 threads/processes guidance | med |
| 6 | `perf/6-combinatorial` | PR5 | #6 precompute `eta**beta`/`tau**alpha`, #7 GA `vstack`, #10 vectorize distance/deposit, #13 njit local search | med |

Each PR:
- keeps changes **general and flexible** (no problem-specific assumptions);
- preserves behavior (same optimizer semantics; seeded runs stay reproducible);
- must keep `pytest tests/` green (excluding the out-of-scope `tests/test_cluster.py`);
- includes a before/after timing on a fixed seed where it claims a speedup.

## Baseline (Core i7-1185G7, inside `.venv`)

- Full non-cluster suite: **18 passed in ~128 s** — the runtime is dominated by the
  ACO/`truncnorm` hotspot that PR2 removes.
- Headline hotspot: ~60% of ACO wall-clock is scipy building docstrings for a
  per-sample `truncnorm` object (see report). Measured primitive speedup from the
  PR2 fix: **177×**.

## Ordering rationale

1. **Quick wins first** (PR2) — biggest win for least risk; also makes the whole test
   suite fast, which speeds up every subsequent PR's verification.
2. **Solution deck** (PR3) — removes unbounded growth so later benchmarks are stable.
3. **ACO vectorization** (PR4) — builds on the faster sampling primitive from PR2.
4. **Parallel architecture** (PR5) — the "copy fixed data once" refactor; benefits from a
   clean, bounded deck and vectorized workers underneath it.
5. **Combinatorial** (PR6) — independent of the continuous stack; lands last so its
   larger surface area doesn't block the high-value continuous fixes.
