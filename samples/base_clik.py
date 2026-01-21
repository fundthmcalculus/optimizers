import os

import numpy as np

from optimizers import (
    plot_convergence,
    AntColonyOptimizer,
    AntColonyOptimizerConfig,
    GeneticAlgorithmOptimizer,
    GeneticAlgorithmOptimizerConfig,
)
from optimizers.continuous.step import StepWiseOptimizer, StepWiseOptimizerConfig
from optimizers.continuous.gd import (
    GradientDescentOptimizer,
    GradientDescentOptimizerConfig,
)
from optimizers.continuous.variables import InputContinuousVariable

from remi.src.clik_functions import get_gam2_vals, calc_clik_params
from remi.src.dynamics import inverse_dynamics
from remi.src.kinematics import (
    calc_capture_point_position,
    calc_end_effector_position,
    calc_end_effector_velocity,
    calc_capture_point_velocity,
    calc_capture_point_acceleration,
    calc_J,
    calc_J_dot,
)

np.seterr(invalid='ignore', divide='ignore', over='ignore')

# TODO - This allows us to change the optimizer type at the drop of a hat
# Optimizer = GradientDescentOptimizer
# OptimizerConfig = GradientDescentOptimizerConfig
# Optimizer = StepWiseOptimizer
# OptimizerConfig = StepWiseOptimizerConfig
# Optimizer = AntColonyOptimizer
# OptimizerConfig = AntColonyOptimizerConfig
Optimizer = GeneticAlgorithmOptimizer
OptimizerConfig = GeneticAlgorithmOptimizerConfig

# LOCAL PACKAGE IMPORT
from remi.src.system import System
from remi.src.visualize import plot_states, plot_controls, animate

# Physical Parameters
r_s = np.array([-1.25, 0.0])
r_t = np.array([1.25, 0.0])
rho = np.array([0.5, 0.5, 0.5, 0.5])
m = np.array([250.0, 25.0, 25.0, 180.0])
I = np.array([25.0, 2.5, 2.5, 18.0])
d = np.zeros(4)

# Simulation Settings
t_dur = 5.0
step_size = 0.1
tol = 0.01
max_tau = (np.inf, np.inf, np.inf, 0.0)

# Initial Conditions
y0 = np.array(
    [
        np.pi / 3.0,  # theta_s
        -0.3,  # theta_1
        -0.1,  # theta_2
        0.0,  # theta_t
        0.0,  # theta_dot_s
        0.0,  # theta_dot_1
        0.0,  # theta_dot_2
        0.2,
    ]
)  # theta_dot_t

# Put parameters and settings in dict
parameters = dict(r_s=r_s, r_t=r_t, rho=rho, m=m, I=I, d=d)

settings = dict(t_dur=t_dur, step_size=step_size, tol=tol, max_tau=max_tau)

# Define system
sys1 = System(y0, parameters, settings)


def calculate_cost(sol):
    # NOTE - Lifted from TruEAI version
    if sol.status == -1:
        return 1e100
    elif sol.status == 0:
        penalty = 1e10
    else:
        penalty = 0.0

    # Don't want oscillatory behavior upon reaching target
    if np.any(np.abs(sol.u[-1]) > 1.0):
        penalty += 1e10

    isu = np.sum(sol.u**2) * 0.1
    Ts = sol.t[-1]

    if not np.isfinite(isu):
        return 5e7

    return 5_000.0 * Ts + isu + penalty


# CLIK control goes here...
def controls(t, y, control_params: np.ndarray | None = None) -> np.ndarray:
    bounds = np.ones(8)  # or 10
    bounds[0::2] = 0.0  # [0,1]
    bounds[:4] *= 0.001
    bounds[4:] *= 0.2

    # Don't want the bounds to be the same, if so then fuzzy isn't
    # used
    if (
        bounds[0] == bounds[1]
        or bounds[2] == bounds[3]
        or bounds[4] == bounds[5]
        or bounds[6] == bounds[7]
    ):  # or\
        #    bounds[8] == bounds[9]:
        return 1e99

    p_bounds = sorted((bounds[0], bounds[1]))
    w_bounds = sorted((bounds[2], bounds[3]))
    kp_bounds = sorted((bounds[4], bounds[5]))
    kd_bounds = sorted((bounds[6], bounds[7]))
    # kn_bounds = sorted((bounds[8], bounds[9]))

    if p_bounds[0] == 0.0:
        p_bounds[0] = 0.00001
    if w_bounds[0] == 0.0:
        w_bounds[0] = 0.00001
    if kp_bounds[0] == 0.0:
        kp_bounds[0] = 0.001
    if kd_bounds[0] == 0.0:
        kd_bounds[0] = 0.001
    # if kn_bounds[0] == 0.:
    #     kn_bounds[0] == 0.001

    # System limits
    q_bar = np.zeros(3)
    q_max = np.array(
        [2.0 * np.pi, np.pi / 2.0, np.pi]  # theta_s limit  # theta_1 limit
    )  # theta_2 limit
    q_min = -q_max

    # FIS bounds
    max_th1 = q_max[1]
    max_th2 = q_max[2]
    max_dth1 = 2.0
    max_dth2 = 2.0
    # max_eN = 2.
    # Normalize w FIS by distance btwn satellites
    max_djk = np.sqrt((r_s[0] - r_t[0]) ** 2 + (r_s[1] - r_t[1]) ** 2)
    max_djk_dot = 1.0  # kind of a guesstimate

    # Specific states
    q = y[:3]
    qdot = y[4:-1]

    # Compute error and derivative error
    xd = calc_capture_point_position(y, rho, r_t)
    xe = calc_end_effector_position(y, rho, r_s)
    vd = calc_capture_point_velocity(y, rho, r_t)
    ve = calc_end_effector_velocity(y, rho, r_s)

    e = xd - xe
    e_dot = vd - ve

    # Capture point acceleration used in qddot_P calc
    ad = calc_capture_point_acceleration(y, 0.0, rho, r_t)

    # Calc and cache Jacobian information
    J = calc_J(y, rho, r_s)
    J_dot = calc_J_dot(y, rho, r_s)

    M = J @ J.T
    M_inv = np.linalg.pinv(M)
    M_dot = J_dot @ J.T + J @ J_dot.T
    Jp = J.T @ M_inv
    Jp_dot = (J_dot.T @ M_inv) - (J.T @ M_inv @ M_dot @ M_inv)

    # Calculate parameters for gam2 calculation and for inference input
    pos, vel, djk, djk_dot = get_gam2_vals(y, rho, r_s, r_t)

    # TODO - INFER P, W, KP, KD, AND KN HERE
    # p2 = fiss[0]([q[1] / max_th1, qdot[1] / max_dth1])
    # p3 = fiss[0]([q[2] / max_th2, qdot[2] / max_dth2])
    #
    # w11 = fiss[1]([djk[0][0] / max_djk, djk_dot[0][0] / max_djk_dot])
    # w12 = fiss[1]([djk[0][1] / max_djk, djk_dot[0][1] / max_djk_dot])
    # w21 = fiss[1]([djk[1][0] / max_djk, djk_dot[1][0] / max_djk_dot])
    # w22 = fiss[1]([djk[1][1] / max_djk, djk_dot[1][1] / max_djk_dot])
    # w31 = fiss[1]([djk[2][0] / max_djk, djk_dot[2][0] / max_djk_dot])
    # w32 = fiss[1]([djk[2][1] / max_djk, djk_dot[2][1] / max_djk_dot])
    #
    # p_rank = fiss[2]([e[0, 0] / max_djk, e[1, 0] / max_djk])
    # d_rank = fiss[3]([e_dot[0, 0] / max_djk_dot, e_dot[1, 0] / max_djk_dot])
    #
    # KP = fiss[4]([p_rank, d_rank])
    # KD = fiss[5]([p_rank, d_rank])

    # KP = fiss[2]([e[0, 0]/max_djk, e[1, 0]/max_djk])
    # KD = fiss[3]([e_dot[0, 0]/max_djk_dot, e_dot[1, 0]/max_djk_dot])
    # KN = fiss[4]([])

    if control_params is not None:
        KD = control_params[0]
        KN = 1.0
        KP = control_params[1]
        p1 = 2.225e-308
        p2 = control_params[2]
        p3 = control_params[3]
        w = control_params[4:]
        p = np.array([p1, p2, p3])

        # TODO - Put gains on the appropriate range
        # p2 = (p_bounds[1] - p_bounds[0]) * p2 + p_bounds[0]
        # p3 = (p_bounds[1] - p_bounds[0]) * p3 + p_bounds[0]
        #
        # w = (w_bounds[1] - w_bounds[0]) * w + w_bounds[0]
        #
        # KP = (kp_bounds[1] - kp_bounds[0]) * KP + kp_bounds[0]
        # KD = (kd_bounds[1] - kd_bounds[0]) * KD + kd_bounds[0]

        # p = np.array([1e-308, p2, p3])
    else:
        KD, KN, KP, p, w = get_control_params()

    # Calculate parameters for qddot_S
    lam, lam_dot = calc_clik_params(y, rho, r_s, pos, vel, p, q_bar, q_max, q_min, w)
    edot_N = (np.eye(3) - Jp @ J) @ (lam.squeeze() - qdot)

    # Apply gains and previous information to get primary desired
    # acceleration and secondary desired acceleration
    qddot_P = (Jp @ (ad + KD * e_dot + KP * e - (J_dot @ qdot)[:, None])).squeeze()
    qddot_S = (np.eye(3) - Jp @ J) @ (lam_dot.squeeze() + KN * edot_N) - (
        Jp @ J_dot @ Jp + Jp_dot
    ) @ J @ (lam.squeeze() - qdot)

    # Calculate complete desired acceleration
    qddot = qddot_P + qddot_S

    # Compute inverse dynamics to find proper control input
    tau = inverse_dynamics(y, np.hstack((qddot, 0.0)), r_s, r_t, rho, m, I, d)
    return tau


def get_control_params() -> tuple[float, float, float, np.ndarray, np.ndarray]:
    # INFER P, W, KP, KD, AND KN HERE
    p = np.array([2.225073858507201e-308, 1e-2, 1e-2])  # (0, 0.1)
    w = np.ones(6) * 2.0  # w11 w12 w21 w22 w31 w32   # (0, 2)
    KD = 5.0  # (0, 20)
    KP = 5.0  # (0, 20)
    KN = 1.0  # (0, 20) (is a constant)
    return KD, KN, KP, p, w


def event(t, y, tol):
    ee = calc_end_effector_position(y, rho, r_s)
    capt = calc_capture_point_position(y, rho, r_t)
    dist = np.sqrt((ee[0] - capt[0]) ** 2 + (ee[1] - capt[1]) ** 2)

    return dist <= tol


def simulate(x0: np.ndarray | None = None):
    def controls2(t, y):
        return controls(t, y, x0)

    sys1.set_controller(controls2)
    sys1.set_event(event)
    sol = sys1.run()
    return sol


def optimize_simulate(x0: np.ndarray) -> np.float64:
    try:

        def controls2(t, y):
            return controls(t, y, x0)

        sys1.set_controller(controls2)
        sys1.set_event(event)
        sol = sys1.run()
        return calculate_cost(sol)
    except:
        print("Exception in simulation optimization")
        return 1e100


def fuzzy_optimize():
    param_optim = OptimizerConfig(
        name="CLIK Controller",
        joblib_prefer="processes",
        local_grad_optim="perturb",  # this slows things down!
        population_size=32,
        solution_archive_size=64,
        n_jobs=8,
    )

    w_l = 1.9
    w_u = 2.1

    k_l = 4.9
    k_u = 5.1

    p_l = 0.005
    p_u = 0.02

    variables = [
        InputContinuousVariable(
            name="KD", lower_bound=k_l, upper_bound=k_u, initial_value=5.0
        ),
        InputContinuousVariable(
            name="KP", lower_bound=k_l, upper_bound=k_u, initial_value=5.0
        ),
        InputContinuousVariable(
            name="p2", lower_bound=p_l, upper_bound=p_u, initial_value=0.01
        ),
        InputContinuousVariable(
            name="p3", lower_bound=p_l, upper_bound=p_u, initial_value=0.01
        ),
        InputContinuousVariable(
            name="w1", lower_bound=w_l, upper_bound=w_u, initial_value=2.0
        ),
        InputContinuousVariable(
            name="w2", lower_bound=w_l, upper_bound=w_u, initial_value=2.0
        ),
        InputContinuousVariable(
            name="w3", lower_bound=w_l, upper_bound=w_u, initial_value=2.0
        ),
        InputContinuousVariable(
            name="w4", lower_bound=w_l, upper_bound=w_u, initial_value=2.0
        ),
        InputContinuousVariable(
            name="w5", lower_bound=w_l, upper_bound=w_u, initial_value=2.0
        ),
        InputContinuousVariable(
            name="w6", lower_bound=w_l, upper_bound=w_u, initial_value=2.0
        ),
    ]

    optim = Optimizer(
        config=param_optim,
        fcn=optimize_simulate,
        variables=variables,
    )
    results = optim.solve()
    return results


def main():
    print("Optimizing Fixed Control parameters")
    results = fuzzy_optimize()
    print("Optimized Fixed Control parameters:", results)
    plot_convergence(results.solution_history, "Optimization")

    sol = simulate(results.solution_vector)
    plot_states(sol.t, sol.y, save=False, show=True)

    try:
        plot_controls(sol.t[:-1], sol.u, save=False, show=True)
    except:
        print("Failed to plot last-time controls")
        plot_controls(sol.t, sol.u, save=False, show=True)

    animate(
        sol.y, parameters, os.getcwd(), blit=True
    )  # , frames=[i for i in range(len(sol.t)) if i%10 == 0])


if __name__ == "__main__":
    main()
