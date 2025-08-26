import pytest

from aco import AntColonyOptimizerConfig, AntColonyOptimizer
from ga import (
    GeneticAlgorithmOptimizerConfig,
    GeneticAlgorithmOptimizer,
)
from gd import (
    GradientDescentOptimizer,
    GradientDescentOptimizerConfig,
)
from pso import ParticleSwarmOptimizerConfig, ParticleSwarmOptimizer
from variables import InputContinuousVariable, InputDiscreteVariable
from opt_types import *


def optim_ackley(x: af64) -> f64:
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


def optim_rosenbrock(x: af64) -> f64:
    # https://en.wikipedia.org/wiki/Rosenbrock_function
    return np.sum(100 * (x[0::2] ** 2 - x[1::2]) ** 2 + (x[0::2] - 1) ** 2)


def optim_para(x: af64) -> f64:
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
        local_grad_optim=True,
        joblib_prefer="processes",
    )
    optimizer = AntColonyOptimizer(
        name="ACO-AckleyFunction",
        config=config,
        variables=input_variables,
        fcn=optim_ackley,
    )
    best_solution = optimizer.solve()
    print(
        f"Best solution: {best_solution.solution_vector} with value: {best_solution.solution_score}"
    )
    assert pytest.approx(best_solution.solution_score) == optim_ackley(best_solution.solution_vector)

def test_ga():
    input_variables = [
        InputContinuousVariable("x", -15, 30),
        InputContinuousVariable("y", -15, 30),
    ]

    config = GeneticAlgorithmOptimizerConfig(
        name="GA Optimizer",
        joblib_prefer="threads",
    )
    optimizer = GeneticAlgorithmOptimizer(
        name="GA-AckleyFunction",
        config=config,
        variables=input_variables,
        fcn=optim_ackley,
    )
    best_solution = optimizer.solve()
    print(
        f"Best solution: {best_solution.solution_vector} with value: {best_solution.solution_score}"
    )
    assert pytest.approx(best_solution.solution_score) == optim_ackley(best_solution.solution_vector)

def test_pso():
    input_variables = [
        InputContinuousVariable("x", -15, 30),
        InputContinuousVariable("y", -15, 30),
    ]

    config = ParticleSwarmOptimizerConfig(
        name="PSO Optimizer",
        joblib_prefer="threads",
    )
    optimizer = ParticleSwarmOptimizer(
        name="PSO-AckleyFunction",
        config=config,
        variables=input_variables,
        fcn=optim_ackley,
    )
    best_solution = optimizer.solve()
    print(
        f"Best solution: {best_solution.solution_vector} with value: {best_solution.solution_score}"
    )
    assert pytest.approx(best_solution.solution_score) == optim_ackley(best_solution.solution_vector)

def test_gd():
    n_dim = 10
    input_variables = [
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
        "GD",
        config=GradientDescentOptimizerConfig(
            name="GD-optim",
            parallel_discrete_search=True,
            discrete_search_size=40,
            joblib_num_procs=4,
            joblib_prefer="processes",
            num_generations=60,
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
        "PSO",
        config=ParticleSwarmOptimizerConfig(
            name="PSO-optim",
            joblib_prefer="processes",
            joblib_num_procs=4,
            num_generations=60,
            local_grad_optim="single-var-grad",
        ),
        variables=input_variables,
        fcn=optim_rosenbrock,
    )
    best_solution = pso_soln.solve()
    print("Best solution:", best_solution)
    assert pytest.approx(best_solution.solution_score, abs=0.1) == 0.0


