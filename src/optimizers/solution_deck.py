from typing import Any, Callable, Literal, Optional, Tuple
import numpy as np
from functools import lru_cache
from kmodes.kmodes import KModes

from .opt_types import f64, af64, ab8, b8
from .variables import InputContinuousVariable, InputVariables

# Some type hinting
InputArguments = dict[str, Any]
GoalFcn = Callable[[af64, Optional[InputArguments]], f64]
LocalOptimType = Literal["none", "grad", "single-var-grad", "perturb"]
InitializationType = Literal["random", "fibonacci"]


class SolutionDeck:
    def __init__(self, archive_size: int, num_vars: int, dtype=f64):
        self.solution_archive = np.empty((archive_size, num_vars), dtype=dtype)
        self.solution_value = np.empty((archive_size,), dtype=dtype)
        self.is_local_optima = np.empty((archive_size,), dtype=b8)
        # TODO - Add support for constraints
        # self.solution_constraints = np.empty((archive_size,), dtype=dtype)  # Vector for constraints
        self.archive_size = archive_size
        self.num_vars = num_vars

    def append(
        self, solutions: af64, values: af64, local_optima: bool | b8 | ab8 = False
    ):
        assert (
            solutions.shape[0] == values.shape[0]
        ), f"Batch size mismatch on append, solutions={solutions.shape}, values={values.shape}"
        if isinstance(local_optima, bool) or isinstance(local_optima, np.bool_):
            self.is_local_optima = np.hstack(
                [
                    self.is_local_optima,
                    np.full(solutions.shape[0], local_optima, dtype=b8),
                ]
            )
        else:
            assert (
                solutions.shape[0] == local_optima.shape[0]
            ), f"Batch size mismatch on append, solutions={solutions.shape}, local_optima={local_optima.shape}"
            self.is_local_optima = np.hstack([self.is_local_optima, local_optima])

        self.solution_archive = np.vstack([self.solution_archive, solutions])
        self.solution_value = np.hstack([self.solution_value, values])

    def initialize_solution_deck(
        self,
        variables: InputVariables,
        eval_fcn: GoalFcn,
        preserve_percent: float = 0.0,
        init_type: InitializationType = "random",
    ) -> None:
        if len(variables) != self.num_vars:
            raise ValueError(
                "Number of variables does not match the initialized deck size."
            )
        num_preserve = int(self.archive_size * preserve_percent)
        fibb_spiral_points: af64 = None
        if init_type == "fibonacci" and num_preserve < self.archive_size:
            fibb_spiral_points = fibonacci_sphere_points(
                self.archive_size - num_preserve, self.num_vars
            )
        for k in range(self.archive_size):
            for i, variable in enumerate(variables):
                if k >= num_preserve:
                    if init_type == "random":
                        self.solution_archive[k, i] = variable.initial_random_value()
                    elif init_type == "fibonacci":
                        # http://www.math.vanderbilt.edu/saffeb/texts/161.pdf
                        self.solution_archive[k, i] = variable.range_value(
                            fibb_spiral_points[k - num_preserve, i]
                        )
                    else:
                        raise ValueError(f"Unknown initialization type: {init_type}")
            if k >= num_preserve:
                self.solution_value[k] = eval_fcn(self.solution_archive[k])
                self.is_local_optima[k] = False  # Initially, none are local optima

    def deduplicate(self, abs_err: f64 = 1e-4, rel_err: f64 = 1e-2) -> None:
        """Deduplicate solutions in the archive based on closeness. Keeps the best solutions.
        Args:
            abs_err (f64): Absolute tolerance for closeness across all dimensions.
            rel_err (f64): Relative tolerance for closeness across all dimensions.
        """
        # TODO - Handle the case of discrete variables with manhattan distance?
        # Sort first
        self.sort()
        # Deduplicate solutions (worst to best)
        for i_row in range(len(self.solution_archive) - 1, 0, -1):
            for j_row in range(i_row - 1, 0, -1):
                if len(self.solution_value) == self.archive_size:
                    return
                if np.allclose(
                    self.solution_archive[i_row],
                    self.solution_archive[j_row],
                    rtol=rel_err,
                    atol=abs_err,
                ):
                    self.solution_archive = np.delete(
                        self.solution_archive, i_row, axis=0
                    )
                    self.solution_value = np.delete(self.solution_value, i_row, axis=0)
                    self.is_local_optima = np.delete(
                        self.is_local_optima, i_row, axis=0
                    )
                else:
                    # Because sorted, we can break early
                    break

    def sort(self) -> None:
        idx = np.argsort(self.solution_value)
        self.solution_archive = self.solution_archive[idx]
        self.solution_value = self.solution_value[idx]
        self.is_local_optima = self.is_local_optima[idx]

    def __len__(self) -> int:
        return self.solution_archive.shape[0]

    def get(self, idx) -> tuple[af64, f64, b8]:
        return (
            self.solution_archive[idx],
            self.solution_value[idx],
            self.is_local_optima[idx],
        )

    def get_best(self) -> tuple[af64, f64, b8]:
        self.sort()
        return self.get(0)

    def get_clusters(self, n_clusters: int = -1) -> tuple[af64, af64]:
        if n_clusters == -1:
            n_clusters = self.solution_archive.shape[0]
        k_modes_model = KModes(n_clusters=n_clusters, random_state=42)
        cluster_labels = k_modes_model.fit_predict(self.solution_archive)
        return cluster_labels, k_modes_model.mode_indicies_


@lru_cache(maxsize=16)
def fibonacci_sphere_points(n: int, k: int) -> np.ndarray:
    """
    Generate N points uniformly distributed on the k-dimensional unit sphere using a Fibonacci spiral,
    then scale the rectangular coordinates to be on the unit hyper-cube [0,1]^k.

    Args:
        n (int): Number of points.
        k (int): Dimension of the sphere.

    Returns:
        np.ndarray: Array of shape (n, k) with coordinates in [0,1].
    """
    if k < 2:
        raise ValueError("Dimension k must be at least 2.")
    if n < 1:
        raise ValueError("Number of points n must be at least 1.")
    points = np.ones((n, k), dtype=np.float64)
    phi = np.pi * (np.sqrt(5.0) + 1.0)  # golden angle
    indices = np.arange(0, n) + 0.5
    theta = np.arccos(1 - 2 * indices / n)
    points[:, 0] *= np.cos(theta)
    points[:, 1] *= np.sin(theta)
    for i in range(n):
        for j in range(k):
            if j == 0:
                points[i, j] = np.cos(theta[j] * i)
                points[i, 1:] = np.sin(theta[1:] * i)
            elif j > 0:
                points[i, j] = points[i, j - 1] * np.sin(phi * i)

    # Normalize to unit sphere
    points = points / 2.0 + 0.5  # Scale to [0,1]
    return points
