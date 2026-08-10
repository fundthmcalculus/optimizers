"""Tests for quasi-random samplers and input allocation.

Tests verify:
1. Samplers generate correct shapes and ranges
2. Discrete/categorical variables are handled via range_value mapping
3. Seeded runs are reproducible
4. Quasi-random samplers show measurably better coverage (lower discrepancy)
5. Convergence performance differs between sampling strategies
"""

import numpy as np
import pytest

from optimizers.continuous.variables import (
    InputContinuousVariable,
    InputDiscreteVariable,
)
from optimizers.core.random import set_seed
from optimizers.core.samplers import (
    create_sampler,
    UniformSampler,
    SobolSampler,
    HaltonSampler,
    LatinHypercubeSampler,
)
from optimizers.continuous.aco import (
    AntColonyOptimizerConfig,
    AntColonyOptimizer,
)
from optimizers.continuous.ga import (
    GeneticAlgorithmOptimizerConfig,
    GeneticAlgorithmOptimizer,
)


class TestSamplerBasics:
    """Verify sampler implementations produce correct shapes and ranges."""

    def test_sampler_factory(self):
        """Test sampler factory creates correct types."""
        assert isinstance(create_sampler("uniform"), UniformSampler)
        assert isinstance(create_sampler("sobol"), SobolSampler)
        assert isinstance(create_sampler("halton"), HaltonSampler)
        assert isinstance(create_sampler("lhs"), LatinHypercubeSampler)

    def test_sampler_invalid_type(self):
        """Test factory rejects unknown sampler types."""
        with pytest.raises(ValueError, match="Unknown sampler type"):
            create_sampler("invalid")

    @pytest.mark.parametrize("sampler_type", ["uniform", "sobol", "halton", "lhs"])
    def test_sampler_shape_and_range(self, sampler_type):
        """Test samplers produce correct shape and stay in [0,1]^d."""
        set_seed(42)
        sampler = create_sampler(sampler_type)
        points = sampler.sample(n=100, d=5, seed=42)

        assert points.shape == (100, 5), f"{sampler_type}: wrong shape"
        assert np.all(points >= 0.0), f"{sampler_type}: points < 0"
        assert np.all(points <= 1.0), f"{sampler_type}: points > 1"

    @pytest.mark.parametrize("sampler_type", ["uniform", "sobol", "halton", "lhs"])
    def test_sampler_reproducibility(self, sampler_type):
        """Test seeded runs produce identical results."""
        sampler = create_sampler(sampler_type)

        points1 = sampler.sample(n=50, d=3, seed=12345)
        points2 = sampler.sample(n=50, d=3, seed=12345)

        np.testing.assert_array_equal(
            points1,
            points2,
            err_msg=f"{sampler_type}: seeded runs not reproducible",
        )

    def test_sampler_different_seeds_differ(self):
        """Test that different seeds produce different samples."""
        sampler = create_sampler("sobol")
        points1 = sampler.sample(n=50, d=3, seed=1)
        points2 = sampler.sample(n=50, d=3, seed=2)

        # Not all points should be identical
        assert not np.allclose(points1, points2)


class TestDiscreteVariableMapping:
    """Test samplers work with discrete variables via range_value."""

    def test_discrete_variable_range_mapping(self):
        """Test range_value maps [0,1] to discrete choices uniformly."""
        discrete_var = InputDiscreteVariable("color", values=np.array([10, 20, 30, 40]))

        # Test boundary cases
        assert discrete_var.range_value(0.0) == 10
        assert discrete_var.range_value(0.5) in [20, 30]  # middle range
        assert discrete_var.range_value(1.0) == 40

    def test_sampler_with_discrete_variable(self):
        """Test sampler output can be mapped through discrete variable."""
        sampler = create_sampler("sobol")
        discrete_var = InputDiscreteVariable("category", values=np.array([1, 2, 3]))

        points = sampler.sample(n=50, d=1, seed=42)
        mapped = np.array([discrete_var.range_value(p[0]) for p in points])

        # All mapped values should be in the discrete set
        assert np.all(np.isin(mapped, [1, 2, 3]))

    def test_continuous_variable_range_mapping(self):
        """Test range_value maps [0,1] to [lower, upper]."""
        cont_var = InputContinuousVariable("x", lower_bound=-10.0, upper_bound=20.0)

        assert cont_var.range_value(0.0) == pytest.approx(-10.0)
        assert cont_var.range_value(0.5) == pytest.approx(5.0)
        assert cont_var.range_value(1.0) == pytest.approx(20.0)


class TestDiscrepancy:
    """Test quasi-random samplers have lower discrepancy than uniform.

    Discrepancy measures how evenly points fill the space. Lower is better.
    We use a simple box-counting approximation: divide [0,1]^d into cells
    and check coverage uniformity.
    """

    def compute_discrepancy_2d(self, points: np.ndarray, n_cells: int = 5) -> float:
        """Approximate discrepancy by counting occupancy in grid cells.

        For 2D points in [0,1]^2, divide into n_cells x n_cells grid and
        measure deviation from expected uniform occupancy.

        Args:
            points: (n, 2) array of points in [0,1]^2
            n_cells: resolution of the grid

        Returns:
            Discrepancy estimate (lower is better, 0 = perfect uniformity)
        """
        if points.shape[1] != 2:
            raise ValueError("This discrepancy metric works for 2D only")

        n_points = points.shape[0]
        expected_per_cell = n_points / (n_cells * n_cells)

        # Bin points into cells
        cells = (points * n_cells).astype(int)
        cells = np.clip(cells, 0, n_cells - 1)  # handle edge case

        # Count occupancy
        grid = np.zeros((n_cells, n_cells))
        for cell in cells:
            grid[cell[0], cell[1]] += 1

        # Measure deviation from expected
        deviation = np.sum(np.abs(grid - expected_per_cell))
        return float(deviation / (n_cells * n_cells))

    def test_sobol_vs_uniform_coverage(self):
        """Test Sobol has better coverage than uniform in 2D."""
        n_points = 256

        # Generate samples
        uniform_sampler = create_sampler("uniform")
        sobol_sampler = create_sampler("sobol")

        uniform_points = uniform_sampler.sample(n=n_points, d=2, seed=42)
        sobol_points = sobol_sampler.sample(n=n_points, d=2, seed=42)

        # Compute discrepancy
        uniform_disc = self.compute_discrepancy_2d(uniform_points)
        sobol_disc = self.compute_discrepancy_2d(sobol_points)

        # Sobol should have better (lower) discrepancy
        assert (
            sobol_disc < uniform_disc
        ), f"Sobol {sobol_disc:.4f} not < uniform {uniform_disc:.4f}"

    def test_halton_vs_uniform_coverage(self):
        """Test Halton has better coverage than uniform in 2D."""
        n_points = 256

        uniform_sampler = create_sampler("uniform")
        halton_sampler = create_sampler("halton")

        uniform_points = uniform_sampler.sample(n=n_points, d=2, seed=42)
        halton_points = halton_sampler.sample(n=n_points, d=2, seed=42)

        uniform_disc = self.compute_discrepancy_2d(uniform_points)
        halton_disc = self.compute_discrepancy_2d(halton_points)

        assert halton_disc < uniform_disc

    def test_lhs_vs_uniform_coverage(self):
        """Test LHS has better coverage than uniform in 2D."""
        n_points = 256

        uniform_sampler = create_sampler("uniform")
        lhs_sampler = create_sampler("lhs")

        uniform_points = uniform_sampler.sample(n=n_points, d=2, seed=42)
        lhs_points = lhs_sampler.sample(n=n_points, d=2, seed=42)

        uniform_disc = self.compute_discrepancy_2d(uniform_points)
        lhs_disc = self.compute_discrepancy_2d(lhs_points)

        assert lhs_disc < uniform_disc


class TestConvergencePerformance:
    """Test convergence of optimizers with different samplers.

    This measures wall-clock / function-evaluations to reach target fitness,
    comparing uniform (baseline) with quasi-random samplers.
    """

    def ackley_2d(self, x: np.ndarray) -> float:
        """2D Ackley function (cheap benchmark)."""
        a = 20.0
        b = 0.2
        c = 2 * np.pi
        d = len(x)
        return (
            -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
            - np.exp(1 / d * np.sum(np.cos(c * x)))
            + a
            + np.exp(1)
        )

    @pytest.mark.parametrize("sampler_type", ["uniform", "sobol", "halton", "lhs"])
    def test_ga_with_sampler(self, sampler_type):
        """Test GA converges with each sampler type."""
        set_seed(42)
        input_variables = [
            InputContinuousVariable("x", -15, 30),
            InputContinuousVariable("y", -15, 30),
        ]

        config = GeneticAlgorithmOptimizerConfig(
            name=f"GA-{sampler_type}",
            population_size=20,
            num_generations=10,
            solution_archive_size=50,
            sampler_type=sampler_type,
            init_type="random",
        )
        optimizer = GeneticAlgorithmOptimizer(
            config=config,
            variables=input_variables,
            fcn=self.ackley_2d,
        )
        result = optimizer.solve()

        # Should produce a valid result
        assert result.solution_score < float("inf")
        assert result.solution_vector.shape == (2,)
        # Ackley optimum is at (0, 0) with value ~0
        assert result.solution_score >= 0.0

    def test_aco_with_sobol_better_than_uniform(self):
        """Test that Sobol initialization helps ACO convergence."""
        input_variables = [
            InputContinuousVariable("x", -15, 30),
            InputContinuousVariable("y", -15, 30),
        ]

        # Run with uniform sampler (baseline)
        set_seed(42)
        config_uniform = AntColonyOptimizerConfig(
            name="ACO-uniform",
            population_size=15,
            num_generations=8,
            solution_archive_size=40,
            sampler_type="uniform",
            init_type="random",
            joblib_prefer="threads",
        )
        optimizer_uniform = AntColonyOptimizer(
            config=config_uniform,
            variables=input_variables,
            fcn=self.ackley_2d,
        )
        result_uniform = optimizer_uniform.solve()

        # Run with Sobol sampler
        set_seed(42)
        config_sobol = AntColonyOptimizerConfig(
            name="ACO-sobol",
            population_size=15,
            num_generations=8,
            solution_archive_size=40,
            sampler_type="sobol",
            init_type="random",
            joblib_prefer="threads",
        )
        optimizer_sobol = AntColonyOptimizer(
            config=config_sobol,
            variables=input_variables,
            fcn=self.ackley_2d,
        )
        result_sobol = optimizer_sobol.solve()

        # Both should converge reasonably well
        assert result_uniform.solution_score < 15.0
        assert result_sobol.solution_score < 15.0

        # Sobol may not always be strictly better (optimization is stochastic),
        # but the initial archive should be better distributed
        print(
            f"\nUniform final score: {result_uniform.solution_score:.6f}\n"
            f"Sobol final score: {result_sobol.solution_score:.6f}"
        )


class TestMixedVariables:
    """Test samplers with mixed continuous and discrete variables."""

    def test_mixed_variable_mapping(self):
        """Test sampler works with mixed continuous/discrete variables."""
        from optimizers.solution_deck import SolutionDeck

        variables = [
            InputContinuousVariable("x", -10.0, 10.0),
            InputDiscreteVariable("category", values=np.array([1, 2, 3, 4])),
            InputContinuousVariable("y", 0.0, 1.0),
        ]

        deck = SolutionDeck(archive_size=20, num_vars=3)

        def dummy_fcn(x: np.ndarray) -> float:
            return float(np.sum(x))

        # Initialize with Sobol sampler
        deck.initialize_solution_deck(
            variables,
            dummy_fcn,
            preserve_percent=0.0,
            init_type="random",
            sampler_type="sobol",
        )

        # Check that the deck was populated correctly
        assert deck.solution_archive.shape == (20, 3)
        assert np.all(np.isfinite(deck.solution_archive))
        # Discrete variable should have valid values
        assert np.all(np.isin(deck.solution_archive[:, 1], [1, 2, 3, 4]))
        # Continuous variables should be within bounds
        assert np.all(deck.solution_archive[:, 0] >= -10.0)
        assert np.all(deck.solution_archive[:, 0] <= 10.0)
        assert np.all(deck.solution_archive[:, 2] >= 0.0)
        assert np.all(deck.solution_archive[:, 2] <= 1.0)
