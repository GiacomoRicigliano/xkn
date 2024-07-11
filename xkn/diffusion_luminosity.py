import sys
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator

from . import nuclear_heat as nh
from .utils import c, day2sec
from .incomplete_gamma import scaled_upper_gamma

# Note: np.frompyfunc returns dtype=object, while np.vectorize handles the type
# correctly. In this particular applications, np.frompyfunc leads to a 50% slow
# down, probably because of the type mishandling, so np.vectorize is
# preferable.
sug = np.vectorize(scaled_upper_gamma, otypes=["float64"]) # scaled upper incomplete gamma function, i.e exp(z) * Gamma(s, z)


def generate_diff_lums(
    ye, entropy, tau, times, glob_vars, shell_params, glob_params, **kwargs
):
    if shell_params["heat_model"] == "RP":
        A_alphas = [
            nh.skynet_heating_params(YE, S, TAU) for YE, S, TAU in zip(ye, entropy, tau)
        ]
    elif shell_params["heat_model"] == "K":
        A_alphas = len(ye) * [
            (1.95e10 * glob_vars["eps0"] / 2e18, glob_params["alpha"])
        ]
    # TODO use correct heating fit params for PBR and LR
    elif shell_params["heat_model"] == "PBR":
        A_alphas = len(ye) * [
            (1.95e10 * glob_vars["eps0"] / 2e18, glob_params["alpha"])
        ]
    elif shell_params["heat_model"] == "LR":
        A_alphas = [
            nh.skynet_heating_params(YE, S, TAU) for YE, S, TAU in zip(ye, entropy, tau)
        ]
    else:
        sys.exit(
            "Wrong input name for heating rate model\n"
            + "Please use:\n"
            + '"RP" for Perego et al 2021\n'
            + '"PBR" for Perego et al 2017 ApJL\n'
            + '"LR" for Lippuner & Roberts 2016 ApJ\n'
            + '"K" for Korobkin 2015'
        )

    return [
        DiffusionLum(
            glob_params["t_0"],
            times,
            glob_params["T_0"],
            glob_params["cnst_eff"] * A,
            glob_params["idx_eff"] + alpha,
        )
        for (A, alpha) in A_alphas
    ]


# definition of luminosity class:
class DiffusionLum(object):
    # class parameters (cgs):

    N = 500  # number of terms in the luminosity semi-analytical expansion formula (for convergence)

    # array of expansion terms indices:
    no = np.arange(1, N + 1)
    n = no[
        :, np.newaxis
    ]  # reforms array on different axis with respect to time array to create matrix

    # array of term-curbing coefficients for initial conditions:
    S = np.append(
        1, np.zeros(np.size(n) - 1)
    )  # as initial condition on the time profile of the energy density, sets to zero all the terms in the expansion except for the first one
    S = S[:, np.newaxis]

    # array of alternate signs:
    sign = np.empty(N, int)
    sign[::2] = 1
    sign[1::2] = -1
    sign = sign[:, np.newaxis]

    # class instance definition:
    def __init__(self, t_0, time, T_0, A, alpha):

        # class instance parameters(cgs):
        self.t_0 = t_0
        self.t_f = time[-1]
        self.E_0 = (
            T_0**4 * 7.57e-15
        )  # initial outflow energy density [erg/cm^3] (a=7.57e-15 erg/(cm^3*K^4) radiation constant)
        self.A = A
        self.alpha = alpha
        self.t = time  # logarithmically spaced time array

        # array factors in the solution of the temporal differential equation:
        self.A_n_factor = (
            np.power(DiffusionLum.n, self.alpha - 3)
            * DiffusionLum.sign
            * np.power(np.pi, self.alpha - 3)
            * 2**0.5
            / np.power(2, self.alpha / 2)
            * self.A
            * np.power(t_0, -self.alpha / 2)
            / self.E_0
        )
        self.gamma_factor = -0.5 * (np.pi * DiffusionLum.n * self.t) ** 2 / t_0
        self.gamma_K_nt_factor = 0.5 * (np.pi * DiffusionLum.n) ** 2 * t_0

        # linear 1D interpolation of first function in temporal differential equation solution:
        self.Np = 300  # number of sample points x (interpolation precision)
        self.x_i = (
            -0.5
            * (np.pi * DiffusionLum.N * self.t_f) ** 2
            / (t_0 * 63661977.23675813 / 10000)
        )  # x mesh left extreme
        self.x_f = (
            -0.5 * (np.pi * t_0) ** 2 / (t_0 * 63661977.23675813 * 10000)
        )  # x mesh right extreme
        self.x = np.flip(
            -np.logspace(np.log10(-self.x_f), np.log10(-self.x_i), self.Np)
        )  # sample mesh
        self.f = interp1d(
            self.x, self.interpf(self.x), bounds_error=False, fill_value="extrapolate"
        )  # interpolating function (requires one argument)

        # linear 3D interpolation of second function in temporal differential equation solution:
        self.Np_K = np.array(
            [50, 100, 50]
        )  # number of sample points for first, second and third sample mesh (interpolation precision)
        self.t_K = np.logspace(
            np.log10(t_0), np.log10(self.t_f), self.Np_K[0]
        )  # first sample mesh
        self.n_K = np.linspace(1, DiffusionLum.N, self.Np_K[1])  # second sample mesh
        self.tau_0_K = np.logspace(
            np.log10(63661977.23675813 / 10000),
            np.log10(63661977.23675813 * 10000),
            self.Np_K[2],
        )  # third sample mesh
        self.G_K = self.interpfunc(
            *np.meshgrid(self.t_K, self.n_K, self.tau_0_K, indexing="ij", sparse=True)
        )  # function to be interpolated
        self.f_K = RegularGridInterpolator(
            (self.t_K, self.n_K, self.tau_0_K),
            self.G_K,
            bounds_error=False,
            fill_value=None,
        )  # interpolating function (requires three arguments)

    # function to be interpolated definition:
    def interpf(self, x):
        return np.cos(np.pi * 0.5 * self.alpha) * sug(
            1 - self.alpha / 2, x
        )  # function to be interpolated

    def interpfunc(self, t_K, n_K, tau_0_K):
        return (
            np.exp(0.5 * (np.pi * n_K) ** 2 * (self.t_0 - t_K**2 / self.t_0) / tau_0_K)
            * np.cos(np.pi * 0.5 * self.alpha)
            * sug(1 - self.alpha / 2, -0.5 * (np.pi * n_K) ** 2 * self.t_0 / tau_0_K)
        )

    # solution of the temporal differential equation function definition:
    def solution(self, tau_0, rho_0):
        A_n = self.A_n_factor * np.power(tau_0, 1 - self.alpha / 2) * rho_0
        K_nt = DiffusionLum.S * np.exp(
            (self.gamma_factor + self.gamma_K_nt_factor) / tau_0
        ) - A_n * self.f_K((self.t, DiffusionLum.n, tau_0))
        return K_nt + A_n * self.f(self.gamma_factor / tau_0)

    # luminosity function definition:
    def calc_lum(self, v_max, k, M):
        rho_0 = M / (
            4 / 3 * np.pi * (v_max * self.t_0) ** 3
        )  # initial outflow density [g/cm^3]
        tau_0 = (
            3 * k * rho_0 * (v_max * self.t_0) ** 2 / c
        )  # collective factor in the solution of the temporal differential equation
        phi_nt = self.solution(
            tau_0, rho_0
        )  # solution of the temporal differential equation (matrix generated using t and n arrays)
        T = (
            DiffusionLum.sign * DiffusionLum.n * phi_nt
        )  # matrix generated using t and n arrays
        return (
            np.sum(T, axis=0)
            * 4
            * np.pi**2
            * c
            * v_max
            * 2**0.5
            * self.t_0
            * self.E_0
            / (3 * k * rho_0)
        )  # sum over n of matrix elements and factor multiplication (expansion formula)
