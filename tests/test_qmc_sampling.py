"""Tests for quasi-random (low-discrepancy) sampling in initial population generation."""

import warnings
import numpy as np
import pytest
from scipy.stats import qmc

from optimizers.continuous.variables import (
    InputContinuousVariable,
    InputDiscreteVariable,
)
from optimizers.continuous.ga import (
    GeneticAlgorithmOptimizer,
    GeneticAlgorithmOptimizerConfig,
)
from optimizers.sampling.qmc import generate_qmc_samples, Sobol, Halton, LatinHypercube
from optimizers.solution_deck import SolutionDeck
from optimizers.core.random import set_seed


def simple_objective(x):
    """Simple objective for testing."""
    return np.sum(x**2)


class TestQMCGeneration:
    """Tests for QMC sample generation."""

    def test_sobol_samples(self):
        """Test Sobol sequence generation."""
        samples = generate_qmc_samples(n=32, d=2, sampler="sobol")
        assert samples.shape == (32, 2)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_halton_samples(self):
        """Test Halton sequence generation."""
        samples = generate_qmc_samples(n=32, d=2, sampler="halton")
        assert samples.shape == (32, 2)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_lhs_samples(self):
        """Test Latin Hypercube Sampling."""
        samples = generate_qmc_samples(n=32, d=2, sampler="lhs")
        assert samples.shape == (32, 2)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_sobol_non_power_of_2(self):
        """Test Sobol with non-power-of-2 sample count works correctly."""
        # This should work without raising errors even for non-power-of-2 sizes
        # The internal generate_qmc_samples suppresses warnings, so they don't leak to users
        samples = generate_qmc_samples(n=33, d=3, sampler="sobol")
        assert samples.shape == (33, 3)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_invalid_sampler(self):
        """Test that invalid sampler raises ValueError."""
        with pytest.raises(ValueError, match="Unknown sampler"):
            generate_qmc_samples(n=10, d=2, sampler="invalid")

    def test_seeding_reproducibility_sobol(self):
        """Test that seeded Sobol generates identical samples."""
        set_seed(42)
        samples1 = generate_qmc_samples(n=20, d=3, sampler="sobol", seed=42)
        samples2 = generate_qmc_samples(n=20, d=3, sampler="sobol", seed=42)
        np.testing.assert_allclose(samples1, samples2)

    def test_seeding_reproducibility_halton(self):
        """Test that seeded Halton generates identical samples."""
        samples1 = generate_qmc_samples(n=20, d=3, sampler="halton", seed=123)
        samples2 = generate_qmc_samples(n=20, d=3, sampler="halton", seed=123)
        np.testing.assert_allclose(samples1, samples2)

    def test_seeding_reproducibility_lhs(self):
        """Test that seeded LHS generates identical samples."""
        samples1 = generate_qmc_samples(n=20, d=3, sampler="lhs", seed=456)
        samples2 = generate_qmc_samples(n=20, d=3, sampler="lhs", seed=456)
        np.testing.assert_allclose(samples1, samples2)

    def test_different_seeds_different_samples(self):
        """Test that different seeds produce different samples."""
        samples1 = generate_qmc_samples(n=20, d=2, sampler="sobol", seed=1)
        samples2 = generate_qmc_samples(n=20, d=2, sampler="sobol", seed=2)
        assert not np.allclose(samples1, samples2)


class TestDiscrepancy:
    """Tests for low-discrepancy properties of QMC samplers."""

    def compute_discrepancy(self, points):
        """Compute discrepancy using scipy's implementation.

        Lower values indicate better uniformity.
        """
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        return qmc.discrepancy(points)

    def test_qmc_lower_discrepancy_than_uniform(self):
        """Test that QMC samplers have lower discrepancy than uniform random."""
        set_seed(42)
        n_samples = 64  # Use power of 2 for Sobol
        n_trials = 5

        # Generate uniform random samples
        uniform_discrepancies = []
        for _ in range(n_trials):
            uniform_points = np.random.uniform(0, 1, size=(n_samples, 2))
            disc = self.compute_discrepancy(uniform_points)
            uniform_discrepancies.append(disc)

        # Generate Sobol samples
        sobol_discrepancies = []
        for trial in range(n_trials):
            sobol_points = generate_qmc_samples(
                n=n_samples, d=2, sampler="sobol", seed=trial
            )
            disc = self.compute_discrepancy(sobol_points)
            sobol_discrepancies.append(disc)

        # Sobol should have lower average discrepancy than uniform
        avg_uniform = np.mean(uniform_discrepancies)
        avg_sobol = np.mean(sobol_discrepancies)
        assert (
            avg_sobol < avg_uniform
        ), f"Sobol ({avg_sobol:.6f}) should beat uniform ({avg_uniform:.6f})"

    def test_halton_lower_discrepancy_than_uniform(self):
        """Test that Halton has lower discrepancy than uniform random."""
        set_seed(42)
        n_samples = 100
        n_trials = 5

        uniform_discrepancies = []
        for _ in range(n_trials):
            uniform_points = np.random.uniform(0, 1, size=(n_samples, 2))
            disc = self.compute_discrepancy(uniform_points)
            uniform_discrepancies.append(disc)

        halton_discrepancies = []
        for trial in range(n_trials):
            halton_points = generate_qmc_samples(
                n=n_samples, d=2, sampler="halton", seed=trial
            )
            disc = self.compute_discrepancy(halton_points)
            halton_discrepancies.append(disc)

        avg_uniform = np.mean(uniform_discrepancies)
        avg_halton = np.mean(halton_discrepancies)
        assert (
            avg_halton < avg_uniform
        ), f"Halton ({avg_halton:.6f}) should beat uniform ({avg_uniform:.6f})"

    def test_lhs_lower_discrepancy_than_uniform(self):
        """Test that LHS has lower discrepancy than uniform random."""
        set_seed(42)
        n_samples = 100
        n_trials = 5

        uniform_discrepancies = []
        for _ in range(n_trials):
            uniform_points = np.random.uniform(0, 1, size=(n_samples, 2))
            disc = self.compute_discrepancy(uniform_points)
            uniform_discrepancies.append(disc)

        lhs_discrepancies = []
        for trial in range(n_trials):
            lhs_points = generate_qmc_samples(
                n=n_samples, d=2, sampler="lhs", seed=trial
            )
            disc = self.compute_discrepancy(lhs_points)
            lhs_discrepancies.append(disc)

        avg_uniform = np.mean(uniform_discrepancies)
        avg_lhs = np.mean(lhs_discrepancies)
        assert (
            avg_lhs < avg_uniform
        ), f"LHS ({avg_lhs:.6f}) should beat uniform ({avg_uniform:.6f})"


class TestInitializationWithQMC:
    """Tests for initializing SolutionDeck with QMC samplers."""

    def test_solution_deck_initialization_sobol(self):
        """Test SolutionDeck initialization with Sobol sampler."""
        variables = [
            InputContinuousVariable("x", -5.0, 5.0),
            InputContinuousVariable("y", -5.0, 5.0),
        ]
        deck = SolutionDeck(archive_size=50, num_vars=2)

        def eval_fcn(x):
            return np.sum(x**2)

        deck.initialize_solution_deck(
            variables, eval_fcn, preserve_percent=0.0, init_type="sobol"
        )

        # Check that deck was populated
        assert np.all(np.isfinite(deck.solution_value))
        assert deck.solution_archive.shape == (50, 2)
        # All values should be within bounds
        assert np.all(deck.solution_archive >= -5.0)
        assert np.all(deck.solution_archive <= 5.0)

    def test_solution_deck_initialization_halton(self):
        """Test SolutionDeck initialization with Halton sampler."""
        variables = [
            InputContinuousVariable("x", 0.0, 10.0),
            InputContinuousVariable("y", 0.0, 10.0),
        ]
        deck = SolutionDeck(archive_size=40, num_vars=2)

        deck.initialize_solution_deck(
            variables, lambda x: np.sum(x), preserve_percent=0.0, init_type="halton"
        )

        assert np.all(np.isfinite(deck.solution_value))
        assert deck.solution_archive.shape == (40, 2)

    def test_solution_deck_initialization_lhs(self):
        """Test SolutionDeck initialization with LHS sampler."""
        variables = [
            InputContinuousVariable("x", -1.0, 1.0),
            InputContinuousVariable("y", -1.0, 1.0),
        ]
        deck = SolutionDeck(archive_size=60, num_vars=2)

        deck.initialize_solution_deck(
            variables, lambda x: np.sum(x**2), preserve_percent=0.0, init_type="lhs"
        )

        assert np.all(np.isfinite(deck.solution_value))
        assert deck.solution_archive.shape == (60, 2)

    def test_discrete_variable_mapping(self):
        """Test that discrete variables are correctly mapped through range_value."""
        continuous_vars = [InputContinuousVariable("x", 0.0, 10.0)]
        discrete_vars = [InputDiscreteVariable("choice", values=np.array([1, 5, 10]))]
        all_vars = continuous_vars + discrete_vars
        deck = SolutionDeck(archive_size=10, num_vars=2)

        deck.initialize_solution_deck(
            all_vars, lambda x: x[0] + x[1], preserve_percent=0.0, init_type="sobol"
        )

        # Check discrete variable is mapped to one of the valid values
        discrete_values = deck.solution_archive[:, 1]
        assert np.all(np.isin(discrete_values, [1, 5, 10]))

    def test_preserve_percent_with_qmc(self):
        """Test that preserve_percent works with QMC samplers."""
        variables = [
            InputContinuousVariable("x", 0.0, 1.0),
            InputContinuousVariable("y", 0.0, 1.0),
        ]
        deck = SolutionDeck(archive_size=100, num_vars=2)

        set_seed(42)
        deck.initialize_solution_deck(
            variables,
            lambda x: np.sum(x**2),
            preserve_percent=0.3,
            init_type="sobol",
        )

        # Non-preserved rows should have been initialized and evaluated
        num_preserve = int(100 * 0.3)
        # Non-preserved portion should have reasonable values in bounds
        assert np.all(deck.solution_archive[num_preserve:] >= 0.0)
        assert np.all(deck.solution_archive[num_preserve:] <= 1.0)
        assert np.all(np.isfinite(deck.solution_value[num_preserve:]))

    def test_reproducibility_with_seeding(self):
        """Test that seeded runs produce identical archives."""
        variables = [
            InputContinuousVariable("x", -1.0, 1.0),
            InputContinuousVariable("y", -1.0, 1.0),
        ]

        def run_with_seed(seed):
            set_seed(seed)
            deck = SolutionDeck(archive_size=50, num_vars=2)
            deck.initialize_solution_deck(
                variables,
                lambda x: np.sum(x**2),
                preserve_percent=0.0,
                init_type="sobol",
            )
            return deck.solution_archive.copy()

        arch1 = run_with_seed(123)
        arch2 = run_with_seed(123)

        np.testing.assert_allclose(arch1, arch2)

    def test_config_default_is_uniform(self):
        """Test that default config uses uniform sampling."""
        from optimizers.continuous.ga import GeneticAlgorithmOptimizerConfig

        config = GeneticAlgorithmOptimizerConfig()
        assert config.init_sampler == "uniform"

    def test_optimizer_uses_config_sampler(self):
        """Test that optimizer uses the configured init_sampler."""
        variables = [
            InputContinuousVariable("x", -5.0, 5.0),
            InputContinuousVariable("y", -5.0, 5.0),
        ]

        config = GeneticAlgorithmOptimizerConfig(
            num_generations=2,
            population_size=10,
            solution_archive_size=20,
            init_sampler="sobol",
        )
        optimizer = GeneticAlgorithmOptimizer(
            config=config,
            variables=variables,
            fcn=simple_objective,
        )

        # Run one generation to ensure initialization worked
        result = optimizer.solve()
        assert result is not None
        assert np.isfinite(result.solution_score)


class TestConfigOption:
    """Tests for the init_sampler config option."""

    def test_config_init_sampler_uniform(self):
        """Test that init_sampler='uniform' is accepted."""
        config = GeneticAlgorithmOptimizerConfig(init_sampler="uniform")
        assert config.init_sampler == "uniform"

    def test_config_init_sampler_sobol(self):
        """Test that init_sampler='sobol' is accepted."""
        config = GeneticAlgorithmOptimizerConfig(init_sampler="sobol")
        assert config.init_sampler == "sobol"

    def test_config_init_sampler_halton(self):
        """Test that init_sampler='halton' is accepted."""
        config = GeneticAlgorithmOptimizerConfig(init_sampler="halton")
        assert config.init_sampler == "halton"

    def test_config_init_sampler_lhs(self):
        """Test that init_sampler='lhs' is accepted."""
        config = GeneticAlgorithmOptimizerConfig(init_sampler="lhs")
        assert config.init_sampler == "lhs"

    def test_config_init_sampler_invalid(self):
        """Test that invalid init_sampler is caught at validation time."""
        # The config accepts the value, but validation happens at runtime
        config = GeneticAlgorithmOptimizerConfig()
        # Manually set an invalid value
        config.init_sampler = "invalid"

        variables = [InputContinuousVariable("x", 0.0, 1.0)]
        optimizer = GeneticAlgorithmOptimizer(
            config=config,
            variables=variables,
            fcn=lambda x: x[0],
        )

        # Should raise when trying to solve (during initialization)
        with pytest.raises(ValueError, match="Invalid"):
            optimizer.solve()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
