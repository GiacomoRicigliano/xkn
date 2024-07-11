import numpy as np
import sys
from scipy.interpolate import interp1d, interp2d

from . import extrapolation_2d as expol
from . import utils


class Thermalization(object):

    def __init__(self, therm_model):
        if therm_model in ["BKWM", "BKWM_dens"]:
            self.therm_efficiency = BKWM_therm_efficiency
            self.therm_efficiency_params = self.therm_efficiency_params_2d
            # 2-D
            x = [np.log10(1.0e-3), np.log10(5e-3), np.log10(1e-2), np.log10(5e-2)]
            y = [0.1, 0.2, 0.3]
            a = [
                [2.01, 0.81, 0.56, 0.27],
                [4.52, 1.90, 1.31, 0.55],
                [8.16, 3.20, 2.19, 0.95],
            ]
            b = [
                [0.28, 0.19, 0.17, 0.10],
                [0.62, 0.28, 0.21, 0.13],
                [1.19, 0.45, 0.31, 0.15],
            ]
            d = [
                [1.12, 0.86, 0.74, 0.60],
                [1.39, 1.21, 1.13, 0.90],
                [1.52, 1.39, 1.32, 1.13],
            ]
            # define the interpolation functions
            self.fa = interp2d(x, y, a, kind="linear")
            self.fb = interp2d(x, y, b, kind="linear")
            self.fd = interp2d(x, y, d, kind="linear")

        elif therm_model == "BKWM_1d":
            self.therm_efficiency = BKWM_therm_efficiency
            self.therm_efficiency_params = self.therm_efficiency_params_1d
            # 1-D
            x_barnes = [
                0.011,
                0.025,
                0.0556,
                0.1,
                0.111,
                0.125,
                0.25,
                0.5,
                0.5556,
                1.0,
                1.25,
                5.0,
            ]
            a_barnes = [
                8.16,
                4.52,
                3.20,
                2.01,
                2.19,
                1.90,
                1.31,
                0.81,
                0.95,
                0.56,
                0.55,
                0.27,
            ]
            b_barnes = [
                1.19,
                0.62,
                0.45,
                0.28,
                0.31,
                0.28,
                0.21,
                0.19,
                0.15,
                0.17,
                0.13,
                0.10,
            ]
            d_barnes = [
                1.52,
                1.39,
                1.39,
                1.12,
                1.32,
                1.21,
                1.13,
                0.86,
                1.13,
                0.74,
                0.90,
                0.60,
            ]
            # define the interpolation functions
            self.fa_1d = interp1d(
                x_barnes,
                a_barnes,
                bounds_error=False,
                fill_value=(a_barnes[0], a_barnes[-1]),
            )
            self.fb_1d = interp1d(
                x_barnes,
                b_barnes,
                bounds_error=False,
                fill_value=(b_barnes[0], b_barnes[-1]),
            )
            self.fd_1d = interp1d(
                x_barnes,
                d_barnes,
                bounds_error=False,
                fill_value=(d_barnes[0], d_barnes[-1]),
            )

        elif therm_model == "power_law":
            self.therm_efficiency = power_law_therm_efficiency

        elif therm_model == "cnst":
            self.therm_efficiency = cnst_therm_efficiency

        else:
            sys.exit("Unknown thermalization efficiency model\n")

    def __call__(self, **kwargs):
        return self.therm_efficiency(self, **kwargs)

    def therm_efficiency_params_2d(self, omegas, mass_ej, vel):
        # assign the values of the mass and velocity
        xnew = np.log10(utils.fourpi / omegas * mass_ej)  # mass     [Msun]
        ynew = vel  # velocity [c]
        # compute the parameters by linear interpolation in the table
        return [func(xnew, ynew) for func in [self.fa, self.fb, self.fd]]

    def therm_efficiency_params_1d(self, omegas, mass_ej, vel):
        # assign the value of x=m/v^2
        xnew = utils.fourpi / omegas * mass_ej / vel**2
        # compute the parameters by 1-d interpolation
        return [np.array(func(xnew)) for func in [self.fa_1d, self.fb_1d, self.fd_1d]]


def BKWM_therm_efficiency(cls, **kwargs):
    if any(
        [
            kwargs["times"] is None,
            kwargs["omegas"] is None,
            kwargs["mass_ej"] is None,
            kwargs["vel"] is None,
        ]
    ):
        sys.exit(
            "Error. For thermalization efficiency, user must specify times, angles, mass and velocity.\n"
        )
    coeffs = cls.therm_efficiency_params(
        kwargs["omegas"], kwargs["mass_ej"], kwargs["vel"]
    )
    coeffs = [np.diag(coeff) for coeff in coeffs]
    times_days = kwargs["times"] * utils.sec2day
    _, times_days = np.meshgrid(coeffs[0], times_days)
    tmp = 2.0 * coeffs[1] * times_days ** coeffs[2]
    tmp = 0.36 * (np.exp(-coeffs[0] * times_days) + np.log(1.0 + tmp) / tmp)
    return tmp.T


def power_law_therm_efficiency(cls, **kwargs):
    if any(
        [kwargs["times"] is None, kwargs["cnst_eff"] is None, kwargs["idx_eff"] is None]
    ):
        sys.exit(
            "power_law_therm_efficiency: Error, for power-law thermalization efficiency, user must specify times, thermalization constant and thermalization index.\n"
        )
    return kwargs["cnst_eff"] / kwargs["times"] ** kwargs["idx_eff"]


def cnst_therm_efficiency(cls, **kwargs):
    if (
        kwargs["cnst_eff"] is None
        or kwargs["cnst_eff"] < 0.0
        or kwargs["cnst_eff"] > 1.0
    ):
        sys.exit(
            'cnst_therm_efficiency: Error, parameter "cnst_eff"={}. Cannot be None and has to be inside [0,1].\n'.format(
                kwargs["cnst_eff"]
            )
        )
    return kwargs["cnst_eff"]
