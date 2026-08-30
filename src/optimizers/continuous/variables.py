import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike
from scipy.special import ndtr, ndtri

from ..core.types import af64, f64
from ..core.variables import InputVariable as InputVariable  # re-exported
from ..core.random import rng as global_rng


class InputDiscreteVariable(InputVariable):
    """A variable restricted to a fixed set of choices.

    ``values`` is stored as float64 whatever dtype is passed in. Choices only
    ever leave this class by way of a solution vector, and every buffer that
    holds one is floating-point -- ``SolutionDeck.solution_archive``,
    ``CVTArchive``'s batch, ``OptimizerResult.solution_vector``, all float64 by
    default -- so an integer-valued variable was already widened one call later.
    Normalising at construction keeps the class inside the ``InputVariable``
    contract (``float`` for scalars, ``af64`` for populations) instead of
    returning numpy dtype unions that the base class does not promise and the
    pipeline does not preserve.
    """

    def __init__(
        self,
        name: str,
        values: ArrayLike,
        initial_value: float | None = None,
    ):
        super().__init__(name)
        self.values: af64 = np.asarray(values, dtype=f64)
        if self.values.size == 0:
            raise ValueError(
                f"variable {name!r} needs at least one value to choose from"
            )
        # Explicit None test: ``initial_value or ...`` re-drew at random whenever
        # the caller pinned the variable to a falsy choice (0.0 being the obvious
        # one), which silently ignored the argument.
        self.initial_value = (
            self.random_value() if initial_value is None else float(initial_value)
        )

    def __repr__(self) -> str:
        return f"DV:{self.name} in {self.values}"

    def __str__(self) -> str:
        return self.__repr__()

    def perturb_value(self, current_value: float, perturbation: float = 0.1) -> float:
        # Just randomly tweak to another choice.
        return self.initial_random_value()

    def perturb_values(
        self,
        current_values: af64,
        perturbation: float = 0.1,
        rng: Generator | None = None,
    ) -> af64:
        # Perturbing a discrete variable just re-draws a random choice, so draw
        # the whole population at once.
        if rng is None:
            rng = global_rng()
        n = np.asarray(current_values).shape[0]
        return self._choose(rng, n)

    def random_value(
        self,
        current_value: float = np.nan,
        other_values: af64 | None = None,
        learning_rate: float = 0.7,
    ) -> float:
        rng = global_rng()
        if other_values is not None:
            # Convert into a weighted count, but ensure every option has a non-zero probability
            all_values = np.concatenate((self.values, other_values))
            unique, counts = np.unique(all_values, return_counts=True)
            # Unity normalize - TODO - Utilize the learning rate to adjust the non-base weights
            p_count = counts / np.sum(counts)
            return float(rng.choice(self.values, p=p_count))
        return float(rng.choice(self.values))

    def random_values(
        self,
        current_values: af64,
        other_values: af64 | None = None,
        learning_rate: float = 0.7,
        rng: Generator | None = None,
    ) -> af64:
        # The discrete weighting depends only on the archive column, not on the
        # per-entry current value, so build the probability vector ONCE and draw
        # all samples in a single rng.choice instead of running np.unique per
        # sample. See report item #15.
        if rng is None:
            rng = global_rng()
        n = np.asarray(current_values).shape[0]
        if other_values is not None:
            all_values = np.concatenate((self.values, other_values))
            unique, counts = np.unique(all_values, return_counts=True)
            p_count = counts / np.sum(counts)
            return self._choose(rng, n, p=p_count)
        return self._choose(rng, n)

    def _choose(self, rng: Generator, n: int, p: af64 | None = None) -> af64:
        """One ``rng.choice`` draw, re-attached to the declared float64 type.

        ``Generator.choice`` is stubbed as returning ``dtype[generic]``; the
        ``asarray`` is a no-op at runtime because ``self.values`` is float64.
        """
        return np.asarray(rng.choice(self.values, size=n, p=p), dtype=f64)

    def initial_random_value(self, perturbation: float = 0.1) -> float:
        rng = global_rng()
        return float(rng.choice(self.values))

    def initial_random_values(
        self, n: int, perturbation: float = 0.1, rng: Generator | None = None
    ) -> af64:
        if rng is None:
            rng = global_rng()
        return self._choose(rng, n)

    def range_value(self, p: float) -> float:
        # Map p in [0,1] to the discrete values
        idx = int(p * len(self.values))
        idx = min(max(idx, 0), len(self.values) - 1)
        return float(self.values[idx])

    @property
    def lower_bound(self) -> float:
        return float(self.values.min())

    @property
    def upper_bound(self) -> float:
        return float(self.values.max())

    def get_nearest_value(self, x1: float) -> float:
        return float(self.values[np.argmin(np.abs(self.values - x1))])


class InputContinuousVariable(InputVariable):
    def __init__(
        self,
        name: str,
        lower_bound: float,
        upper_bound: float,
        initial_value: float = np.nan,
        perturbation: float = 0.1,
    ):
        super().__init__(name)
        if not lower_bound < upper_bound:
            raise ValueError(
                f"lower_bound ({lower_bound}) must be less than upper_bound ({upper_bound})"
                f" for variable {name!r}."
            )
        self.__lower_bound = lower_bound
        self.__upper_bound = upper_bound
        self.initial_value = self.initial_random_value()
        if not np.isnan(initial_value):
            # Use perturbation theory around the initial value
            self.initial_value = min(
                self.upper_bound,
                max(
                    self.lower_bound,
                    initial_value + perturbation * self.domain,
                ),
            )

    def __repr__(self) -> str:
        return f"CV:{self.name} in [{self.lower_bound}, {self.upper_bound}]"

    def __str__(self) -> str:
        return self.__repr__()

    def perturb_value(self, current_value: float, perturbation: float = 0.1) -> float:
        # Move it in a gaussian spread around the current value.
        sigma = self.domain * perturbation
        new_value = current_value + sigma * global_rng().normal()
        return max(min(self.upper_bound, new_value), self.lower_bound)

    def perturb_values(
        self,
        current_values: af64,
        perturbation: float = 0.1,
        rng: Generator | None = None,
    ) -> af64:
        # Vectorized gaussian perturbation for a whole population at once.
        if rng is None:
            rng = global_rng()
        cv = np.asarray(current_values, dtype=float)
        sigma = self.domain * perturbation
        return np.clip(
            cv + sigma * rng.normal(size=cv.shape),
            self.lower_bound,
            self.upper_bound,
        )

    def __get_truncated_normal(
        self,
        mean: float = 0.0,
        stdev: float = 1.0,
        low: float = 0.0,
        high: float = 10.0,
        rng: Generator | None = None,
    ) -> float:
        # Inverse-CDF (Gaussian) sampling of a truncated normal. This avoids
        # constructing a scipy ``truncnorm`` frozen distribution on every call
        # (which spent ~60% of ACO runtime building docstrings); the draw is
        # statistically identical. See docs/history/PERFORMANCE_REPORT.md item #1.
        if stdev <= 0.0:
            stdev = 1.0
        if rng is None:
            rng = global_rng()
        a = ndtr((low - mean) / stdev)
        b = ndtr((high - mean) / stdev)
        if b <= a:
            # Degenerate window (mean far outside [low, high]); fall back to the
            # nearer bound rather than dividing a zero-width interval.
            return float(min(max(mean, low), high))
        u = rng.uniform(a, b)
        x = mean + stdev * ndtri(u)
        # ndtri can return +/-inf at the extreme tails; clamp to the domain so
        # the result is always within [low, high] as truncnorm guaranteed.
        return float(min(max(x, low), high))

    def random_value(
        self,
        current_value: float = np.nan,
        other_values: af64 | None = None,
        learning_rate: float = 0.7,
    ) -> float:
        rng = global_rng()
        if other_values is not None:
            # TODO - Other than Manhattan distance, what other distance metrics can be used?
            d2 = np.sum(np.abs(other_values - current_value)) / len(other_values)
            return self.__get_truncated_normal(
                mean=current_value,
                stdev=learning_rate * d2,
                low=self.lower_bound,
                high=self.upper_bound,
                rng=rng,
            )
        return rng.uniform(self.lower_bound, self.upper_bound)

    def random_values(
        self,
        current_values: af64,
        other_values: af64 | None = None,
        learning_rate: float = 0.7,
        rng: Generator | None = None,
    ) -> af64:
        # Vectorized truncated-normal sampling: one draw per entry of
        # current_values, each centered on its own value with spread derived
        # from the (shared) archive column. Equivalent to calling random_value
        # once per entry, but done as array ops. See report item #5.
        if rng is None:
            rng = global_rng()
        cv = np.asarray(current_values, dtype=float)
        if other_values is None:
            return rng.uniform(self.lower_bound, self.upper_bound, size=cv.shape)
        # Mean absolute deviation of the archive column from each center.
        #
        # The direct broadcast (``|other_values[None,:] - cv[:,None]|``, then
        # mean over axis 1) materializes an (n_ants, archive_size) temporary
        # and is O(n_ants * archive_size) *per variable* -- at a large
        # population/archive this dwarfs everything else in ACO (measured:
        # ~90% of total wall-clock at population=1000/archive=3000/dim=30, see
        # docs/history/PERF_CONTINUOUS_REPORT.md addendum). Sort the archive column once and
        # use the standard prefix-sum identity for L1 distance to a sorted
        # array: for sorted ``s`` (prefix sums ``S``) and a query ``c`` with
        # rank ``m`` (count of elements < c),
        #   sum_j |s_j - c| = c*(2m - n) - 2*S[m] + S[n]
        # This is O(archive_size log archive_size) once (the sort) plus
        # O(n_ants log archive_size) (searchsorted for every ant's rank) --
        # independent of archive_size for the per-ant cost, instead of linear
        # in it. Verified numerically identical to the direct computation
        # (max abs diff ~1e-15, float64 rounding) over 200 randomized trials.
        n_archive = other_values.shape[0]
        sorted_other = np.sort(other_values)
        prefix = np.concatenate(([0.0], np.cumsum(sorted_other)))
        total = prefix[-1]
        rank = np.searchsorted(sorted_other, cv)
        d2 = (cv * (2 * rank - n_archive) - 2 * prefix[rank] + total) / n_archive
        stdev = learning_rate * d2
        stdev = np.where(stdev <= 0.0, 1.0, stdev)
        low, high = self.lower_bound, self.upper_bound
        a = ndtr((low - cv) / stdev)
        b = ndtr((high - cv) / stdev)
        span = b - a
        # Avoid a zero-width window for degenerate centers; sampled below.
        safe_span = np.where(span <= 0.0, 1.0, span)
        u = a + rng.uniform(size=cv.shape) * safe_span
        x = cv + stdev * ndtri(u)
        # Degenerate windows fall back to the clamped center.
        x = np.where(span <= 0.0, cv, x)
        return np.clip(x, low, high)

    def initial_random_value(
        self, perturbation: float = 0.1, rng: Generator | None = None
    ) -> float:
        if rng is None:
            rng = global_rng()
        return rng.uniform(self.lower_bound, self.upper_bound)

    def initial_random_values(
        self, n: int, perturbation: float = 0.1, rng: Generator | None = None
    ) -> af64:
        if rng is None:
            rng = global_rng()
        return rng.uniform(self.lower_bound, self.upper_bound, size=n)

    def range_value(self, p: float) -> float:
        # Map p in [0,1] to the variable range
        return self.lower_bound + p * (self.upper_bound - self.lower_bound)

    @property
    def lower_bound(self) -> float:
        return self.__lower_bound

    @property
    def upper_bound(self) -> float:
        return self.__upper_bound


def print_optimal_solution(x: af64, variables: list[InputContinuousVariable]) -> None:
    print("Optimal solution:")
    for ij, var in enumerate(variables):
        print(f"{var.name}: {x[ij]}")
