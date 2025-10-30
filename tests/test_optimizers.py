import pytest
import matplotlib.pyplot as plt
import numpy as np

from optimizers.continuous.aco import AntColonyOptimizerConfig, AntColonyOptimizer
from optimizers.continuous.ga import (
    GeneticAlgorithmOptimizerConfig,
    GeneticAlgorithmOptimizer,
)
from optimizers.continuous.gd import (
    GradientDescentOptimizer,
    GradientDescentOptimizerConfig,
)
from optimizers.core.base import IOptimizerConfig
from optimizers.continuous.optimizer_strategy import MultiTypeOptimizer, GroupedVariableOptimizerConfig, \
    InputVariableGroup, GroupedVariableOptimizer
from optimizers.continuous.pso import (
    ParticleSwarmOptimizerConfig,
    ParticleSwarmOptimizer,
)
from optimizers.solution_deck import SolutionDeck
from optimizers.continuous.variables import (
    InputContinuousVariable,
    InputDiscreteVariable,
    InputVariable,
)
from optimizers.core.types import AF, F


def optim_ackley(x: AF) -> F:
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


def optim_rosenbrock(x: AF) -> F:
    # https://en.wikipedia.org/wiki/Rosenbrock_function
    return np.sum(100 * (x[0::2] ** 2 - x[1::2]) ** 2 + (x[0::2] - 1) ** 2)


def optim_para(x: AF) -> F:
    # N-dimensional parabola
    return np.sqrt(np.sum(np.power(x - 1.414, 2)))


def test_aco():
    input_variables = [
        InputContinuousVariable("x", -15, 30),
        InputContinuousVariable("y", -15, 30),
    ]

    config = AntColonyOptimizerConfig(
        name="Ant Colony Optimizer",
        population_size=30,
        num_generations=50,
        solution_archive_size=100,
        learning_rate=0.5,
        q=1.0,
        local_grad_optim="grad",
        joblib_prefer="processes",
    )
    optimizer = AntColonyOptimizer(
        config=config,
        variables=input_variables,
        fcn=optim_ackley,
    )
    best_solution = optimizer.solve()
    print(
        f"Best solution: {best_solution.solution_vector} with value: {best_solution.solution_score}"
    )
    assert pytest.approx(best_solution.solution_score) == optim_ackley(
        best_solution.solution_vector
    )


def test_ga():
    input_variables = [
        InputContinuousVariable("x", -15, 30),
        InputContinuousVariable("y", -15, 30),
    ]

    config = GeneticAlgorithmOptimizerConfig(
        name="GA Optimizer",
    )
    optimizer = GeneticAlgorithmOptimizer(
        config=config,
        variables=input_variables,
        fcn=optim_ackley,
    )
    best_solution = optimizer.solve()
    print(
        f"Best solution: {best_solution.solution_vector} with value: {best_solution.solution_score}"
    )
    assert pytest.approx(best_solution.solution_score) == optim_ackley(
        best_solution.solution_vector
    )


def test_group_optimize():
    input_variables = [
        InputContinuousVariable("x", -15, 30),
        InputContinuousVariable("y", -15, 30),
    ]

    config = GroupedVariableOptimizerConfig(
        name="Grouped Var Optimizer",
        groups=[InputVariableGroup(name="x",variables=["x"], optimizer_type="aco"),
                InputVariableGroup(name="y",variables=["y"], optimizer_type="ga")],
    )
    optimizer = GroupedVariableOptimizer(
        config=config,
        variables=input_variables,
        fcn=optim_ackley,
    )
    best_solution = optimizer.solve()
    print(
        f"Best solution: {best_solution.solution_vector} with value: {best_solution.solution_score}"
    )
    assert pytest.approx(best_solution.solution_score) == optim_ackley(
        best_solution.solution_vector
    )


def test_multi_optimizer():
    input_variables = [
        InputContinuousVariable("x", -15, 30),
        InputContinuousVariable("y", -15, 30),
    ]

    config = IOptimizerConfig(
        name="Various-types Optimizer",
    )
    optimizer = MultiTypeOptimizer(
        config=config,
        variables=input_variables,
        fcn=optim_ackley,
    )
    best_solution = optimizer.solve()
    print(
        f"Best solution: {best_solution.solution_vector} with value: {best_solution.solution_score}"
    )
    assert pytest.approx(best_solution.solution_score) == optim_ackley(
        best_solution.solution_vector
    )


def test_gd():
    n_dim = 10
    input_variables: list[InputVariable] = [
        InputContinuousVariable(f"cv-{ij}", lower_bound=-5, upper_bound=5)
        for ij in range(n_dim)
    ]
    # Append some discrete variables
    input_variables.extend(
        [
            InputDiscreteVariable(
                f"dv-{ij}", values=np.linspace(0, 2, num=40), initial_value=0
            )
            for ij in range(n_dim)
        ]
    )
    gd_soln = GradientDescentOptimizer(
        config=GradientDescentOptimizerConfig(
            name="GD-optim",
            parallel_discrete_search=True,
            discrete_search_size=40,
            n_jobs=4,
            joblib_prefer="processes",
        ),
        variables=input_variables,
        fcn=optim_ackley,
    )
    soln = gd_soln.solve()
    print("Best solution:", soln)
    assert soln is not None


def test_rosenbrock():
    n_dim = 10
    input_variables = [
        InputContinuousVariable(f"cv-{ij}", lower_bound=-5, upper_bound=5)
        for ij in range(n_dim)
    ]
    pso_soln = ParticleSwarmOptimizer(
        config=ParticleSwarmOptimizerConfig(
            name="PSO-optim",
        ),
        variables=input_variables,
        fcn=optim_rosenbrock,
    )
    soln = pso_soln.solve()
    print("Best solution:", soln)
    assert soln is not None


def test_fibonacci():
    n_dim = 3
    n_deck = 100
    input_variables = [
        InputContinuousVariable(f"cv-{ij}", lower_bound=-5, upper_bound=5)
        for ij in range(n_dim)
    ]
    soln_deck = SolutionDeck(archive_size=n_deck, num_vars=n_dim)
    soln_deck.initialize_solution_deck(
        input_variables, optim_para, init_type="fibonacci"
    )

    # Verify solution deck was created correctly
    assert soln_deck.archive_size == n_deck
    assert soln_deck.num_vars == n_dim

    # Plot the solution deck points
    solutions = soln_deck.solution_archive
    plt.figure(figsize=(8, 8))
    ax = plt.figure(figsize=(8, 8)).add_subplot(projection="3d")
    ax.plot(solutions[:, 0], solutions[:, 1], solutions[:, 2], c="blue", alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("Fibonacci Solution Distribution")
    plt.grid(True)
    plt.show()
    # plt.savefig('fibonacci_solutions.png')
    # plt.close()
