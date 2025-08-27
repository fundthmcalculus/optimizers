import numpy as np
from opt_types import f64, af64, ab8, b8
from optimizer_base import GoalFcn, InputVariables

class SolutionDeck:
    def __init__(self, archive_size: int, num_vars: int, dtype=f64):
        self.solution_archive = np.empty((archive_size, num_vars), dtype=dtype)
        self.solution_value = np.empty((archive_size,), dtype=dtype)
        self.is_local_optima = np.empty((archive_size,), dtype=b8)
        # TODO - Add support for constraints
        # self.solution_constraints = np.empty((archive_size,), dtype=dtype)  # Vector for constraints
        self.archive_size = archive_size
        self.num_vars = num_vars

    def append(self, solutions: af64, values: af64, local_optima: ab8):
        assert solutions.shape[0] == values.shape[0] == local_optima.shape[0], "Batch size mismatch"

        self.solution_archive = np.vstack([self.solution_archive, solutions])
        self.solution_value = np.hstack([self.solution_value, values])
        self.is_local_optima = np.hstack([self.is_local_optima, local_optima])

    def initialize_solution_deck(self, variables: InputVariables, eval_fcn: GoalFcn, preserve_percent: float = 0.0) -> None:
        if len(variables) != self.num_vars:
            raise ValueError("Number of variables does not match the initialized deck size.")
        num_preserve = int(self.archive_size * preserve_percent)
        for k in range(self.archive_size):
            for i, variable in enumerate(variables):
                if k >= num_preserve:
                    self.solution_archive[k, i] = variable.initial_random_value()
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
                if np.allclose(self.solution_archive[i_row], self.solution_archive[j_row], rtol=rel_err, atol=abs_err):
                    self.solution_archive = np.delete(self.solution_archive, i_row, axis=0)
                    self.solution_value = np.delete(self.solution_value, i_row, axis=0)
                    self.is_local_optima = np.delete(self.is_local_optima, i_row, axis=0)
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
        return (self.solution_archive[idx], self.solution_value[idx], self.is_local_optima[idx])