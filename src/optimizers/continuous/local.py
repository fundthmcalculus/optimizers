from .gd import solve_gd_for_1var, solve_gd_from_x0
from optimizers.solution_deck import WrappedGoalFcn, LocalOptimType
from .variables import InputVariable, InputDiscreteVariable
from optimizers.core.types import af64
from typing import get_args


def apply_local_optimization(
    fcn: WrappedGoalFcn,
    local_optim: LocalOptimType,
    new_solution: af64,
    variables: list[InputVariable],
):
    if local_optim == "grad":
        new_solution, new_value = full_grad_optim(fcn, new_solution, variables)
    elif local_optim == "perturb":
        new_solution, new_value = local_perturb_optim(fcn, new_solution, variables)
    elif local_optim == "single-var-grad":
        new_solution, new_value = single_var_grad_optim(fcn, new_solution, variables)
    elif local_optim == "none":
        # Evaluate the new solution
        new_value = fcn(new_solution)
    else:
        allowed = ", ".join(repr(x) for x in get_args(LocalOptimType))
        raise ValueError(
            f"Invalid local_optim={local_optim!r}. Allowed options: {allowed}"
        )
    return new_solution, new_value


def local_perturb_optim(
    fcn: WrappedGoalFcn, new_solution: af64, variables: list[InputVariable]
):
    new_value = fcn(new_solution)
    # One variable at a time, do a stepwise optimization.
    for i, variable in enumerate(variables):
        old_var_value = new_solution[i]
        new_solution[i] = variable.perturb_value(old_var_value)
        # Evaluate the new solution value
        tmp_soln_value = fcn(new_solution)
        if tmp_soln_value > new_value:
            new_solution[i] = old_var_value
        else:
            new_value = tmp_soln_value
    return new_solution, new_value


def full_grad_optim(
    fcn: WrappedGoalFcn, new_solution: af64, variables: list[InputVariable]
):
    # Configure a continuous only gradient optimizer around this (ACO handles the discrete variable searching already)
    # Don't fire this off on another process, to prevent overloading the OS.
    result = solve_gd_from_x0(new_solution, variables, fcn)
    new_value = result.solution_score
    new_solution = result.solution_vector
    return new_solution, new_value


def single_var_grad_optim(
    fcn: WrappedGoalFcn, new_solution: af64, variables: list[InputVariable]
):
    new_value = fcn(new_solution)
    for var_idx, variable in enumerate(variables):
        if isinstance(variable, InputDiscreteVariable):
            continue
        result = solve_gd_for_1var(new_solution, variables, var_idx, fcn)
        new_value = result.solution_score
        new_solution = result.solution_vector
    return new_solution, new_value
