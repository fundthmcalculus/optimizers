import sys
import time

from optimizers.continuous.variables import InputContinuousVariable
from src.optimizers import AntColonyOptimizerConfig, AntColonyOptimizer
from tests.test_optimizers import optim_ackley


def main():
    print(
        f"GIL Enabled={sys._is_gil_enabled() if hasattr(sys, '_is_gil_enabled') else 'Unknown'}"
    )
    num_runs = 1

    input_variables = [
        InputContinuousVariable(f"x{ij}", -15, 30) for ij in range(1, 10)
    ]

    config = AntColonyOptimizerConfig(
        name="Ant Colony Optimizer",
        population_size=50,
        num_generations=25,
        solution_archive_size=200,
        learning_rate=0.5,
        q=1.0,
        local_grad_optim="single-var-grad",
        joblib_prefer="processes",
        n_jobs=4,
    )
    optimizer = AntColonyOptimizer(
        config=config,
        variables=input_variables,
        fcn=optim_ackley,
    )

    # Process-based execution
    start_time = time.perf_counter()
    for _ in range(num_runs):
        best_solution = optimizer.solve()
    process_time = time.perf_counter() - start_time

    config.joblib_prefer = "threads"

    # Thread-based execution
    start_time = time.perf_counter()
    for _ in range(num_runs):
        best_solution = optimizer.solve()
    thread_time = time.perf_counter() - start_time

    print(
        f"{num_runs}-run Process-based execution time: {process_time/num_runs:.4f} seconds"
    )
    print(
        f"{num_runs}-run Thread-based execution time: {thread_time/num_runs:.4f} seconds"
    )


if __name__ == "__main__":
    main()
