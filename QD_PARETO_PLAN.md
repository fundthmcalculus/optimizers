# Multi-Output Optimization: Pareto + MAP-Elites Add-On — Design & Plan

**Status:** design proposal (no code yet). **Date:** 2026-07-11.
**Goal:** let the optimizers track *multiple outputs per solution* and maintain a
*diverse archive* of good solutions, then feed that diversity back as **better
crossover parents**. This is a meta-heuristic layer that sits on top of the
existing GA / PSO / ACO / GD solvers rather than a new solver.

**Primary objective (confirmed):** *better exploration of the search space and
better convergence* — a Quality-Diversity / **MAP-Elites** emphasis. Keeping a
diverse archive of stepping-stone solutions and recombining across it is the
mechanism for both: wider exploration up front, and a stronger single best answer
at the end because good regions are reached via intermediate elites a pure fitness
gradient would discard.

**Secondary (reporting only):** *Pareto trade-off analysis.* We still track the
multiple outputs per solution and, at report time, compute and plot the
non-dominated (Pareto) front across those objectives. Pareto here is an **output
report / analysis layer**, not the driver of selection — the archive that steers
search is the MAP-Elites grid.

> Design consequence: because the search fitness stays **scalar** (the quantity
> you actually minimize) and diversity comes from a separate **descriptor**, the
> legacy scalar solvers keep working essentially unchanged — MAP-Elites is
> naturally scalar-fitness compatible (best-overall elite = global best). The
> messy "scalarize a vector of objectives for ACO/PSO" problem is thus mostly
> avoided; it only appears if you later want *Pareto-as-driver* (kept as an
> optional mode, §4.6).

---

## 1. What we're actually building (and why)

Today every objective returns a single scalar and the `SolutionDeck` keeps the
top-`archive_size` solutions ranked by that scalar. Every solver reads its
parents from that one deck:

- **GA** — tournament selection over `solution_value` (`continuous/ga.py`).
- **ACO** — a rank CDF over the sorted archive (`continuous/aco.py`, `cdf`).
- **PSO** — the single global best `solution_archive[0]` (`continuous/pso.py`).
- **stop criteria / `get_best`** — `solution_value[0]`.

The problem this add-on solves is twofold:

1. **Multiple outputs.** Real objectives are often vector-valued (cost *and*
   robustness *and* time). We want to track all of them and surface the trade-off
   front instead of collapsing everything into one weighted number up front.
2. **Better crossover through diversity.** A deck ranked by one scalar collapses
   onto a few near-duplicate elites, so crossover mixes parents that are already
   alike — premature convergence. If instead we keep an archive that is diverse
   *by construction* (a Pareto front, or a grid of "elites" across a feature
   space), crossover draws structurally different parents and explores far more of
   the space. This "diverse parents → better recombination" effect is the core of
   Quality-Diversity (QD) research and is the meta-heuristic payoff here.

The insight that ties both together: **the archive is the leverage point.** All
solvers already delegate parent memory to one object. Swap that object for a
multi-objective / quality-diversity archive that exposes the *same interface*,
and every solver gains multi-output tracking and diversity-driven crossover with
minimal churn in the solver code.

---

## 2. Literature review

### 2.1 Pareto multi-objective evolutionary algorithms

- **Dominance & non-dominated sorting.** Solution *a* Pareto-dominates *b* if it
  is no worse on every objective and strictly better on at least one. The
  non-dominated set (Pareto front) is the set of solutions nobody dominates.
- **NSGA-II** (Deb, Pratap, Agarwal, Meyarivan, 2002) — the workhorse. Two ideas
  we will reuse directly:
  - *Fast non-dominated sort*: partition the population into dominance "fronts"
    (rank 0 = Pareto front, rank 1 = what's left after removing rank 0, …), in
    O(M·N²) for M objectives, N solutions.
  - *Crowding distance*: within a front, prefer solutions in sparsely populated
    regions (sum of normalized per-objective neighbour gaps). This is how NSGA-II
    keeps the front spread out and how it truncates an over-full archive.
- **SPEA2** (Zitzler, Laumanns, Thiele, 2001) — strength/density-based fitness +
  a fixed external archive with a density-preserving truncation. Good reference
  for a *bounded* Pareto archive.
- **MOEA/D** (Zhang & Li, 2007) — decomposes one M-objective problem into many
  scalarized single-objective subproblems (weighted-sum / Tchebycheff) solved
  cooperatively. Relevant because our legacy solvers are scalar; MOEA/D shows the
  principled way to *scalarize on the fly* and is a natural bridge (see §4.4).
- **NSGA-III** (Deb & Jain, 2014) — reference-point based, for many-objective
  (≥4) problems where crowding distance degrades.
- **Quality indicators.** The **hypervolume** (Zitzler & Thiele, 1999) — the
  volume of objective space dominated by the front relative to a reference point
  — is the standard single-number measure of front quality. Exact hypervolume is
  expensive in high M (exponential); fine for M≤3, which covers most use.

### 2.2 MAP-Elites & Quality-Diversity

- **MAP-Elites** (Mouret & Clune, 2015, "Illuminating search spaces by mapping
  elites"). Define a low-dimensional **feature/behavior descriptor** for each
  solution (e.g. two extra measured outputs). Discretize descriptor space into a
  grid; each cell keeps the single best-fitness solution ("elite") seen with that
  descriptor. Loop: pick a random elite, mutate/recombine it, evaluate, and if the
  offspring's cell is empty or it beats the incumbent, it takes the cell. The
  output is not one optimum but an *illuminated map* of the best solution at every
  point of the feature space — quality **and** diversity.
- **QD as a framework** (Pugh, Soros, Stanley 2016, "Quality Diversity: A New
  Frontier"; Cully & Demiris 2018, "A Unifying Modular Framework"). Formalizes
  container (archive) + selection + variation as swappable modules — the exact
  separation we want for a clean add-on.
- **CVT-MAP-Elites** (Vassiliades, Chatzilygeroudis, Mouret, 2018). A regular grid
  explodes combinatorially as descriptor dimension grows. Instead, precompute *k*
  centroids by Centroidal Voronoi Tessellation and use "nearest centroid" as the
  cell. Keeps a fixed archive size (k cells) for any descriptor dimension — the
  scalable choice.
- **CMA-ME** (Fontaine, Togelius, Nikolaidis, Hoover, 2020). Marries CMA-ES-style
  adaptive sampling ("emitters") with the MAP-Elites archive; big gains on
  continuous problems. A later phase could add an emitter abstraction; not needed
  for v1.
- **Recent (2020-2025):** PGA-MAP-Elites (gradient-assisted variation),
  Differentiable QD / CMA-MEGA (Fontaine & Nikolaidis 2021) when gradients are
  available, deep-learning descriptors. Out of scope for v1 but the archive
  interface should not preclude them.

### 2.3 The crossover angle (the meta-heuristic payoff)

- **Iso+LineDD** (Vassiliades & Mouret, 2018, "Discovering the Elite Hypervolume
  by Leveraging Interspecies Correlation", GECCO). The key operator for us. Given
  two elites xᵢ, xⱼ from the archive, produce a child:

  ```
  child = xᵢ + σ₁ · N(0, I) + σ₂ · (xⱼ − xᵢ) · N(0, 1)
  ```

  The first term is ordinary isotropic mutation; the second is a *directional*
  step along the line connecting two elites, scaled by a single Gaussian. Because
  good solutions in a problem tend to be correlated ("elite hypervolume"), moving
  along inter-elite directions is dramatically more productive than isotropic
  noise. This is literally "better crossover from a diverse parent pool," and it
  drops straight into our continuous variables.
- **Novelty search / stepping stones** (Lehman & Stanley, 2011, "Abandoning
  Objectives: Evolution through the Search for Novelty Alone"; and the popular
  treatment "Why Greatness Cannot Be Planned"). The theoretical backing for *why*
  diversity helps: many good solutions are only reachable through intermediate
  "stepping stones" that a pure fitness gradient discards. A diverse archive
  preserves those stepping stones as crossover material.
- **Simulated Binary Crossover (SBX)** (Deb & Agrawal, 1995) and polynomial
  mutation — the standard NSGA-II variation operators for real vectors; worth
  offering alongside Iso+LineDD for the Pareto path.

### 2.4 Reference implementations to borrow abstractions from

- **pyribs** (Tjanaka et al., 2023, "pyribs: A Bare-Bones Python Library for
  Quality Diversity"). Cleanest abstraction in the field, and the one we should
  mirror: three decoupled pieces —
  - **Archive** — stores solutions by measures/descriptor (`GridArchive`,
    `CVTArchive`, `SlidingBoundariesArchive`); knows how to `add()` and to report
    coverage/QD-score.
  - **Emitter** — decides *which* solutions to generate next (Gaussian, Iso+Line,
    CMA-ES/CMA-ME, …); holds the variation strategy.
  - **Scheduler** — the loop glue: `ask()` candidates from emitters → user
    evaluates → `tell()` results back to archive+emitters.

  MIT-licensed. We will adopt the **Archive / (Emitter) / loop** split but keep
  our existing solvers as the "emitter+loop," so the add-on is mostly the Archive.
- **pymoo** (Blank & Deb, 2020) — comprehensive multi-objective library (NSGA-II/
  III, MOEA/D, reference directions, performance indicators). Good source of
  vetted operator/indicator implementations to port or validate against.
  Apache-2.0.
- **qdpy**, **DEAP** — additional QD/EA references; DEAP's `tools` has tested
  non-dominated sort and hypervolume we can cross-check.

**Takeaway for our design:** one `Archive` interface, two-to-three concrete
archives (scalar / Pareto / MAP-Elites grid), variation operators as small pure
functions, and a scalarization bridge so the *existing* scalar solvers keep
working. That is the minimal clean surface.

---

## 3. How this maps onto the current codebase

The whole add-on hangs off one existing seam.

**The seam:** `SolutionDeck` (`src/optimizers/solution_deck.py`) is the single
shared parent memory. Its de-facto contract, relied on by every solver:

| Member | Consumers | Multi-output implication |
|---|---|---|
| `solution_archive` (N×vars) | GA/ACO/PSO parent pool | unchanged |
| `solution_value` (N,), ascending=better | GA tournament, ACO rank CDF, PSO best, stop, `get_best` | must become a **scalar surrogate** (rank/scalarization) when objectives are vectors |
| `sort()` | keeps best-first | becomes non-dominated sort / grid ordering |
| `get_best()` | final result, PSO g-best | returns a knee/weighted pick or best-fitness elite |
| `append/deduplicate/truncate` | `update_solution_deck` | become dominance/crowding or cell insertion |
| `to_dict/from_dict` | checkpointing | must serialize objectives + descriptors |

**The other seam:** the objective contract. `GoalFcn`/`WrappedGoalFcn`
(`core/base.py`) is `Callable[[x], float]`, wrapped in `IOptimizer.__init__`
(`continuous/base.py::_wrap_goal`). Multi-output means the raw objective may
return a *vector* of objectives and optionally a *descriptor* vector.

**The scalar assumption is pervasive but shallow.** These spots assume a scalar
and need the surrogate/scalarizer, but none need structural change:
`check_stop_early`, `update_solution_deck` postfix, GA `f1<f2` child pick,
`apply_local_optimization` (gradient/perturb local search compares scalar
fitness). All of them read through the archive's scalar surrogate, so a single
scalarization policy covers them.

---

## 4. Proposed architecture

New subpackage `src/optimizers/archive/` (name TBD — `quality_diversity/` also
fine), leaving `solution_deck.py` in place and making it one implementation of a
shared interface.

```
src/optimizers/archive/
  base.py        # Archive ABC/Protocol + shared helpers (dominance, scalarize)
  scalar.py      # ScalarArchive: today's SolutionDeck behaviour (default)
  pareto.py      # ParetoArchive: NSGA-II non-dominated sort + crowding
  grid.py        # GridArchive (MAP-Elites) + CVTArchive (high-dim descriptors)
  variation.py   # iso_line_dd(), sbx_crossover(), polynomial_mutation()
  metrics.py     # hypervolume(), coverage(), qd_score()
```

### 4.1 The `Archive` interface

Preserve the current `SolutionDeck` surface so existing solvers are unchanged,
and add the multi-output methods:

```python
class Archive(Protocol):
    # --- legacy scalar surface (kept working via scalarization) ---
    solution_archive: AF          # (N, num_vars)
    solution_value: AF            # (N,) scalar surrogate, ascending = better
    def sort(self) -> None: ...
    def get_best(self) -> tuple[AF, F, b8]: ...
    def truncate(self, size: int = -1) -> None: ...

    # --- multi-output surface (new) ---
    def add(self, solutions: AF, objectives: AF,          # objectives (N, m)
            descriptors: AF | None = None) -> AddInfo: ...  # (N, d) or None
    def parents(self, n: int, rng: Generator) -> AF:      # diversity-aware draw
    @property
    def objectives(self) -> AF: ...                       # (N, m)
    def front(self) -> tuple[AF, AF]: ...                 # non-dominated (x, f)
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "Archive": ...
```

- `ScalarArchive` is `m == 1`; `objectives` is `solution_value[:, None]`,
  `parents()` = current rank/tournament behaviour. Zero behaviour change → this is
  the default and the back-compat guarantee.
- **MAP-Elites archive — the flagship for this project.** Bins solutions by a
  low-dim **descriptor**; each cell holds the best-**fitness** elite (fitness stays
  the single scalar you minimize). `solution_value` surrogate is the per-cell
  fitness, so scalar solvers see a normal ranked deck; `get_best()` = best elite
  overall. `parents()` samples *uniformly across occupied cells* — structurally
  diverse parents, which is exactly what drives exploration and better crossover.
  Two backends behind one class: **`CVTArchive`** (centroid cells; the default for
  auto/projection descriptors with unknown bounds — see §4.2) and **`GridArchive`**
  (fixed-bounds regular grid; for explicit descriptors with known ranges).
- `ParetoArchive` (secondary) keeps a bounded non-dominated set; surrogate =
  `nondomination_rank + (1 − normalized_crowding)`. Used mainly to *report* the
  trade-off front (§4.5); available as an optional selection driver (§4.6).

### 4.2 Objective contract (multi-output)

Extend the wrap logic so the user objective may return, all back-compatible:

- `float` → single objective (today).
- **`(fitness: float, outputs: np.ndarray)` → the primary QD shape:** one scalar
  *fitness* to minimize, plus a vector of *tracked outputs*. The outputs feed two
  things — (i) the MAP-Elites **descriptor** (by default a chosen 1–3 of them),
  and (ii) the **Pareto report** over all tracked outputs (§4.5).
- `np.ndarray (m,)` → m objectives with no separate fitness (pure Pareto driver,
  the optional §4.6 mode).

Descriptor sourcing — **confirmed default: auto random projection** (the user
works in high-dimensional decision spaces and typically has no natural low-dim
behavior descriptor):

- **auto (default)** — a fixed **random linear projection** of the decision vector
  down to `descriptor_dim` (≈2–4) coordinates: `d(x) = R · x`, with `R` drawn once
  at solve start from a seeded Gaussian and *held constant* across generations (so
  cells are stable). By the Johnson–Lindenstrauss lemma a random projection
  approximately preserves pairwise distances, so nearby projections ≈ nearby
  solutions — a cheap, dimension-agnostic diversity proxy that needs zero domain
  input and works for arbitrarily high decision dimension.
- **explicit** — user names which tracked outputs form the descriptor, or supplies
  a `descriptor(x, outputs) -> np.ndarray` callable (domain behavior). Better
  illumination when a meaningful descriptor exists; optional.

> **High-dim consequence for the archive choice.** Projection coordinates have
> *unknown, unbounded ranges* — you can't set fixed grid bounds a priori. So the
> **default archive for auto-descriptors is `CVTArchive`** (centroids sampled to
> cover the observed descriptor cloud; fixed cell count for any `descriptor_dim`)
> or a bounds-adaptive *sliding-boundaries* grid, **not** a fixed-bounds
> `GridArchive`. A fixed `GridArchive` is only appropriate when the user gives an
> explicit descriptor with known bounds. This promotes CVT from "phase 4 scaling"
> to the phase-2 default (see §5).

Detection is by return shape, mirroring the existing `_accepts_args` arity sniff.
Config makes intent explicit and validates shapes:

```python
objective_mode: Literal["scalar", "map-elites", "pareto"] = "scalar"
n_outputs: int = 1                               # tracked outputs per solution
descriptor: DescriptorSpec | None = None         # explicit: outputs / callable + bounds
descriptor_auto: Literal["projection", "outputs"] = "projection"  # default when descriptor is None
descriptor_dim: int = 2                          # projection target dim (auto mode)
archive_kind: Literal["cvt", "grid", "sliding"] = "cvt"  # cvt is the high-dim default
n_cells: int = 512                               # cvt/grid capacity (bounded archive)
```

### 4.3 Variation operators (`variation.py`) — the crossover payoff

Pure functions over `(parents, variables, rng)` returning children, so any solver
can call them and workers stay cloudpickle-simple:

- `iso_line_dd(xi, xj, sigma_iso, sigma_line, rng)` — the Iso+LineDD operator
  (§2.3). Default `σ_iso≈0.01·domain`, `σ_line≈0.2`.
- `sbx_crossover(p1, p2, eta, rng)` + `polynomial_mutation` — NSGA-II operators.
- Batched, vectorized versions mirroring the existing `_crossover_batch` /
  `_mutate_batch` so we keep the perf-work vectorization.

The meta-heuristic wiring: GA's `_tournament_selection_batch` gets a sibling
`archive.parents(n)` that draws **diverse** parents (across cells / fronts), and
GA's crossover can be swapped to `iso_line_dd`. That is the whole "better
crossover" mechanism — diverse parents + a directional operator.

### 4.4 Scalarization bridge (only for the optional Pareto-driver mode)

In the **primary MAP-Elites mode this is not needed** — fitness is already scalar.
It is only required if you run the optional *Pareto-as-driver* mode (§4.6), where
ACO/PSO/GD need a scalar surrogate for a vector objective. In that case a small
policy object provides it:

- `"rank"` (default) — non-domination rank + crowding.
- `"weighted"` — user weights `w·f`.
- `"tchebycheff"` — `max_i w_i·|f_i − z_i*|` vs. an ideal point (MOEA/D-style;
  better on non-convex fronts).

`get_best()` in that mode returns a **knee point** (max front bend) or the
user-weighted pick.

### 4.5 Metrics, results, plots

- `metrics.py`: `hypervolume` (M≤3 exact; WFG/Monte-Carlo for higher),
  `coverage` (fraction of cells filled), `qd_score` (sum of cell fitnesses).
- `OptimizerResult` gains optional `pareto_front: (X, F)` and
  `archive_snapshot`, plus per-generation `hypervolume_history` /
  `coverage_history`. Existing scalar fields stay populated via the scalarizer.
- Convergence: replace scalar "no-improvement" with "hypervolume/coverage
  plateaued for k generations."
- `plot/`: a 2-D/3-D Pareto-front scatter and a MAP-Elites heatmap
  (`plotly`, matching the existing plot module).

### 4.6 Optional: Pareto-as-driver mode (not the default)

For completeness and future use, `objective_mode="pareto"` lets the *search* be
driven by a `ParetoArchive` (NSGA-II selection over a vector objective, no scalar
fitness). This is the classic multi-objective EA. It is **not** the primary path
here — it needs the scalarization bridge (§4.4) to keep ACO/PSO/GD working and it
does not give the MAP-Elites exploration benefit. It is kept as a mode because the
`Archive` interface already supports it and it costs little once Phases 1–3 exist.
Our default remains MAP-Elites search + Pareto *reporting*.

---

## 5. Phased implementation plan (ranked)

Each phase is independently shippable, keeps `pytest` green, and preserves scalar
behaviour by default.

| Phase | Deliverable | Risk | Payoff |
|---|---|---|---|
| **1. Archive interface + multi-output plumbing** | Extract the `Archive` interface; make `ScalarArchive` the current behaviour (pure refactor, characterization-tested). Add the `(fitness, outputs)` objective contract in `_wrap_goal` + config, and record tracked outputs on every archived solution. No new search behaviour yet. | low | foundation; multi-output *tracking* |
| **2. MAP-Elites (`CVTArchive`) + auto random-projection descriptor + Iso+LineDD + diverse-parent selection** | The flagship, sized for high-dim decision spaces. Fixed seeded random projection → `descriptor_dim` coordinates; **`CVTArchive`** (centroid cells, handles unknown/unbounded projection ranges); per-cell elites; `archive.parents()` uniform-over-cells sampling; the `iso_line_dd` operator; GA wired to diverse-parent + directional variation. Scalar fitness ⇒ ACO/PSO/`get_best` work unchanged. **This is the exploration + convergence engine.** | med | wider exploration, stronger convergence, better crossover |
| **3. Reporting: Pareto front + QD metrics + plots** | Non-dominated filtering over the tracked outputs → **Pareto-front report + plot** (the "a" deliverable); coverage / QD-score / best-fitness histories; CVT-cell / descriptor scatter (2-D) instead of a grid heatmap; QD-aware stop criteria (coverage/best plateau). | low | the trade-off report + observability |
| **4. Scale & advanced** | fixed-bounds `GridArchive` + sliding-boundaries variant for the explicit-descriptor case; SBX/polynomial operators; optional **Pareto-as-driver** mode (§4.6) with the scalarization bridge; optional pyribs-style **emitter** abstraction (opens the door to CMA-ME). | med | more archive/operator options |

**Sequencing rationale:** the primary goal (exploration + convergence in
high-dimensional spaces) is delivered by **Phase 2**, which is why `CVTArchive` +
the random-projection descriptor are pulled *into* Phase 2 rather than left as
"scaling" — they are the correct default for this user's problems, not an add-on.
Phase 1 stays deliberately thin (interface refactor + output plumbing, scalar path
proven unchanged). Phase 3 turns the tracked outputs into the Pareto report and the
QD metrics that let us *show* the exploration gain. Phase 4 is alternative
archives/operators and the optional Pareto-driven mode.

---

## 6. Key design decisions & risks

- **Back-compatibility is non-negotiable.** `objective_mode="scalar"` is the
  default and must be byte-identical to today (Phase-1 refactor is covered by the
  existing test suite + a "scalar archive == old SolutionDeck" characterization
  test).
- **Minimization convention.** The codebase minimizes (`solution_value` ascending,
  `target_score` is a floor). Dominance/hypervolume must follow the same
  convention (minimize all objectives; negate maximization objectives at the
  boundary).
- **Parallelism.** The `GenerationRunner` (ships fixed data once) is unaffected —
  archives live in the parent process; workers still just evaluate the objective.
  The objective now returns a small vector/descriptor instead of a scalar;
  negligible extra payload.
- **Cost.** Non-dominated sort is O(M·N²) per generation — fine for the bounded
  archive sizes here (100s). Exact hypervolume is the one thing to gate on M≤3 and
  compute sparingly (every k generations).
- **Local search with vector objectives.** `apply_local_optimization` (gradient/
  perturb) needs a scalar; it will optimize the *scalarized* objective (weighted/
  Tchebycheff) around a chosen reference direction, or be disabled in
  multi-objective mode. Decision to confirm with you.
- **Three distinct dimensionalities — don't conflate them.** (i) *decision-space*
  dim (`num_vars`) is **high** for this user; (ii) *objective/output* dim (`m`) is
  small (a few tracked outputs) — so the Pareto report and hypervolume stay cheap;
  (iii) *descriptor* dim (`d`) is low **by construction** (we project down to
  ≈2–4). High decision dim therefore does **not** blow up the archive or the
  report — it only shapes the *variation operator* and the *auto-descriptor*.
- **High-dim decision space reinforces the operator choice.** Isotropic mutation
  degrades badly as dimension grows (curse of dimensionality), which makes
  Iso+LineDD's *directional* (inter-elite) term the load-bearing part of variation
  here — the "elite hypervolume / interspecies correlation" effect it exploits is
  strongest exactly in high-dim genotypes. Good fit for the primary use.
- **Descriptor default is auto random projection** (§4.2). A meaningful domain
  descriptor illuminates better, but auto keeps it zero-setup and dimension-
  agnostic; a fixed seeded projection keeps cells stable across generations.
- **Direction confirmed (2026-07-11):** primary = **(b)** better exploration +
  convergence (MAP-Elites, scalar fitness + descriptor diversity + Iso+LineDD);
  secondary = **(a)** Pareto trade-off front as an *output report* over the tracked
  objectives, not as the search driver. **High-dimensional decision spaces** are
  the expected workload, so the defaults are **auto random-projection descriptors +
  `CVTArchive`**. Plan and phase order above reflect this.

---

## 7. References

*Multi-objective / Pareto*
1. Deb, Pratap, Agarwal, Meyarivan. *A Fast and Elitist Multiobjective Genetic
   Algorithm: NSGA-II.* IEEE TEC, 2002.
2. Zitzler, Laumanns, Thiele. *SPEA2: Improving the Strength Pareto EA.* 2001.
3. Zhang & Li. *MOEA/D: A Multiobjective EA Based on Decomposition.* IEEE TEC, 2007.
4. Deb & Jain. *NSGA-III.* IEEE TEC, 2014.
5. Zitzler & Thiele. *Multiobjective optimization using the hypervolume measure.* 1999.

*MAP-Elites / Quality-Diversity*
6. Mouret & Clune. *Illuminating search spaces by mapping elites.* arXiv:1504.04909, 2015.
   <https://arxiv.org/abs/1504.04909> · tutorial: <https://github.com/jbmouret/map_elites_tutorial>
7. Pugh, Soros, Stanley. *Quality Diversity: A New Frontier for Evolutionary
   Computation.* Frontiers in Robotics and AI, 2016.
8. Cully & Demiris. *Quality and Diversity Optimization: A Unifying Modular
   Framework.* IEEE TEC, 2018.
9. Vassiliades, Chatzilygeroudis, Mouret. *Using CVT to Scale Up MAP-Elites.*
   IEEE TEC, 2018.
10. Fontaine, Togelius, Nikolaidis, Hoover. *CMA-ME.* GECCO, 2020.

*The crossover / variation angle (core to this project)*
11. Vassiliades & Mouret. *Discovering the Elite Hypervolume by Leveraging
    Interspecies Correlation (Iso+LineDD).* GECCO, 2018.
    code: <https://github.com/resibots/vassiliades_2018_gecco>
12. Lehman & Stanley. *Abandoning Objectives: Evolution through the Search for
    Novelty Alone.* Evolutionary Computation, 2011.
13. Deb & Agrawal. *Simulated Binary Crossover for Continuous Search Space.* 1995.
14. *Discrete Gene Crossover Accelerates Solution Discovery in Quality-Diversity
    Algorithms.* arXiv:2602.13730 — recent evidence that recombination operators
    materially speed QD discovery; worth tracking for the Phase-2 operator choice.

*Recent multi-objective + QD (relevant to the §4.6 optional mode / the Pareto report)*
15. *Multi-Objective Quality-Diversity in Unstructured and Unbounded Spaces.*
    arXiv:2504.03715 — the emerging MO-QD combination our report layer approximates.

*Reference libraries / implementations*
16. Tjanaka et al. *pyribs: A Bare-Bones Python Library for Quality Diversity.*
    GECCO 2023, arXiv:2303.00191. Archive / Emitter / Scheduler split; MIT.
    <https://pyribs.org/> · <https://github.com/icaros-usc/pyribs>
17. Blank & Deb. *pymoo: Multi-Objective Optimization in Python.* IEEE Access,
    2020 (NSGA-II/III, MOEA/D, indicators; Apache-2.0).
18. *QDax: Accelerated Quality-Diversity* (JAX; hardware-accelerated QD/MAP-Elites).
    arXiv:2308.03665 — reference for the vectorized/batched archive+variation design.

*Citations verified against arXiv / official project pages (2026-07-11).*
