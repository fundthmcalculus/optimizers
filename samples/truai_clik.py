"""Defines the fitness function of GFS controller."""

import numpy as np

from remi.kinematics import (
    calc_capture_point_position,
    calc_end_effector_position,
    calc_end_effector_velocity,
    calc_capture_point_velocity,
    calc_capture_point_acceleration,
    calc_J,
    calc_J_dot,
)
from remi.clik_functions import get_gam2_vals, calc_clik_params
from remi.dynamics import inverse_dynamics

from eve import (
    FitnessCPU,
    ChromosomeBlueprint,
    GeneBlueprint,
    GeneKind,
    FisBlueprint,
    TrainCenters,
    FisSettings,
)
from psion import Wav


def calculate_cost(sol):
    if sol.status == -1:
        return 1e100
    elif sol.status == 0:
        penalty = 1e10
    else:
        penalty = 0.0

    # Don't want oscilatory behavior upon reaching target
    if np.any(np.abs(sol.u[-1]) > 1.0):
        penalty += 1e10

    isu = np.sum(sol.u**2) * 0.1
    Ts = sol.t[-1]

    if not np.isfinite(isu):
        return 5e7

    return 5_000.0 * Ts + isu + penalty


class Fitness(FitnessCPU):
    def __init__(self, system):
        super().__init__()
        self.chromosome_blueprint = ChromosomeBlueprint(
            [
                FisBlueprint(
                    FisSettings(
                        name="p_fis",
                        shape=(3, 3),
                        train_centers=TrainCenters.off,
                        variable_shape=False,
                        precision=0.01,
                        input_range=(-1.0, 1),
                        output_range=(0.0, 1.0),
                    )
                ),
                FisBlueprint(
                    FisSettings(
                        name="w_fis",
                        shape=(3, 3),
                        train_centers=TrainCenters.off,
                        variable_shape=False,
                        precision=0.01,
                        input_range=(0.0, 1.0),
                        output_range=(0.0, 1.0),
                    )
                ),
                FisBlueprint(
                    FisSettings(
                        name="p_rank_fis",
                        shape=(3, 3),
                        train_centers=TrainCenters.off,
                        variable_shape=False,
                        precision=0.01,
                        input_range=(-1.0, 1.0),
                        output_range=(0.0, 1.0),
                    )
                ),
                FisBlueprint(
                    FisSettings(
                        name="d_rank_fis",
                        shape=(3, 3),
                        train_centers=TrainCenters.off,
                        variable_shape=False,
                        precision=0.01,
                        input_range=(-1.0, 1.0),
                        output_range=(0.0, 1.0),
                    )
                ),
                FisBlueprint(
                    FisSettings(
                        name="kp_fis",
                        shape=(3, 3),
                        train_centers=TrainCenters.off,
                        variable_shape=False,
                        precision=0.01,
                        input_range=(0.0, 1.0),
                        output_range=(0.0, 1.0),
                    )
                ),
                FisBlueprint(
                    FisSettings(
                        name="kd_fis",
                        shape=(3, 3),
                        train_centers=TrainCenters.off,
                        variable_shape=False,
                        precision=0.01,
                        input_range=(0.0, 1.0),
                        output_range=(0.0, 1.0),
                    )
                ),
                # FisBlueprint(FisSettings(name='kn_fis',
                #                          shape=(3,3),
                #                          train_centers=TrainCenters.off,
                #                          variable_shape=False,
                #                          precision=0.01,
                #                          input_range=(-1., 1.),
                #                          output_range=(0., 1.))),
                GeneBlueprint(
                    name="bounds", alleles=[50], shape=[8], kind=GeneKind.repeating
                ),
            ]
        )

        self.data["system"] = system

    def evaluate(self, chromosome):
        names = [
            "p_fis",
            "w_fis",
            "p_rank_fis",
            "d_rank_fis",
            "kp_fis",
            "kd_fis",
        ]  # , 'kn_fis']
        fiss = [
            Wav(chromosome.fis(name).centers(), chromosome.fis(name).rules())
            for name in names
        ]

        bounds = np.array(chromosome.gene("bounds").dna, dtype=float)
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

        sys = self.data["system"]
        rho = sys.rho
        r_s = sys.r_s
        r_t = sys.r_t
        m = sys.m
        I = sys.I
        d = sys.d

        def controls(t, y):
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

            # INFER P, W, KP, KD, AND KN HERE
            p2 = fiss[0]([q[1] / max_th1, qdot[1] / max_dth1])
            p3 = fiss[0]([q[2] / max_th2, qdot[2] / max_dth2])

            w11 = fiss[1]([djk[0][0] / max_djk, djk_dot[0][0] / max_djk_dot])
            w12 = fiss[1]([djk[0][1] / max_djk, djk_dot[0][1] / max_djk_dot])
            w21 = fiss[1]([djk[1][0] / max_djk, djk_dot[1][0] / max_djk_dot])
            w22 = fiss[1]([djk[1][1] / max_djk, djk_dot[1][1] / max_djk_dot])
            w31 = fiss[1]([djk[2][0] / max_djk, djk_dot[2][0] / max_djk_dot])
            w32 = fiss[1]([djk[2][1] / max_djk, djk_dot[2][1] / max_djk_dot])

            p_rank = fiss[2]([e[0, 0] / max_djk, e[1, 0] / max_djk])
            d_rank = fiss[3]([e_dot[0, 0] / max_djk_dot, e_dot[1, 0] / max_djk_dot])

            KP = fiss[4]([p_rank, d_rank])
            KD = fiss[5]([p_rank, d_rank])

            # KP = fiss[2]([e[0, 0]/max_djk, e[1, 0]/max_djk])
            # KD = fiss[3]([e_dot[0, 0]/max_djk_dot, e_dot[1, 0]/max_djk_dot])
            # KN = fiss[4]([])

            # Put gains on appropriate range
            p2 = (p_bounds[1] - p_bounds[0]) * p2 + p_bounds[0]
            p3 = (p_bounds[1] - p_bounds[0]) * p3 + p_bounds[0]

            w11 = (w_bounds[1] - w_bounds[0]) * w11 + w_bounds[0]
            w12 = (w_bounds[1] - w_bounds[0]) * w12 + w_bounds[0]
            w21 = (w_bounds[1] - w_bounds[0]) * w21 + w_bounds[0]
            w22 = (w_bounds[1] - w_bounds[0]) * w22 + w_bounds[0]
            w31 = (w_bounds[1] - w_bounds[0]) * w31 + w_bounds[0]
            w32 = (w_bounds[1] - w_bounds[0]) * w32 + w_bounds[0]

            KP = (kp_bounds[1] - kp_bounds[0]) * KP + kp_bounds[0]
            KD = (kd_bounds[1] - kd_bounds[0]) * KD + kd_bounds[0]

            p = np.array([1e-308, p2, p3])
            w = np.array([w11, w12, w21, w22, w31, w32])

            # p = np.array([1e-308, 1e-2, 1e-2])
            # w = np.ones(6)*2. # w11 w12 w21 w22 w31 w32
            # KD = 5.
            # KP = 5.
            KN = 1.0

            # Calculate parameters for qddot_S
            lam, lam_dot = calc_clik_params(
                y, rho, r_s, pos, vel, p, q_bar, q_max, q_min, w
            )
            edot_N = (np.eye(3) - Jp @ J) @ (lam.squeeze() - qdot)

            # Apply gains and previous information to get primary desired
            # acceleration and secondary desired acceleration
            qddot_P = (
                Jp @ (ad + KD * e_dot + KP * e - (J_dot @ qdot)[:, None])
            ).squeeze()
            qddot_S = (np.eye(3) - Jp @ J) @ (lam_dot.squeeze() + KN * edot_N) - (
                Jp @ J_dot @ Jp + Jp_dot
            ) @ J @ (lam.squeeze() - qdot)

            # Calculate complete desired acceleration
            qddot = qddot_P + qddot_S

            # Compute inverse dynamics to find proper control input
            tau = inverse_dynamics(y, np.hstack((qddot, 0.0)), r_s, r_t, rho, m, I, d)
            return tau

        sys.set_controller(controls)
        sol = sys.run()

        return calculate_cost(sol)
