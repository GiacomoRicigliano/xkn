import sys
import hashlib
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator

from . import nuclear_heat as nh
from .utils import c, day2sec
from .incomplete_gamma import scaled_upper_gamma_vec

# Scaled_upper_gamma directly vectorized with numpy:
sug = scaled_upper_gamma_vec

def generate_diff_lums(ye, entropy, tau, times, glob_vars, shell_params, glob_params, **kwargs):
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
            nh.skynet_heating_parax
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

    result = []
    for A, alpha in A_alphas:
        A_eff   = glob_params["cnst_eff"] * A * glob_vars["nuc_corr"] * glob_params["t_0"]**(-alpha)
        alpha_eff = glob_params["idx_eff"] + alpha
        key = _difflum_cache_key(glob_params["t_0"], times, glob_params["T_0"],
                                 A_eff, alpha_eff)
        if key not in _DIFFLUM_CACHE:
            _DIFFLUM_CACHE[key] = DiffusionLum(
                glob_params["t_0"],
                times,
                glob_params["T_0"],
                A_eff,
                alpha_eff,
            )
        result.append(_DIFFLUM_CACHE[key])
    return result

# Definition of luminosity class (DiffusionLum)
class DiffusionLum_direct:
    # Class-level arrays:
    N    = 500
    n    = np.arange(1, N + 1, dtype=float)[:, None]     # (500, 1)
    sign = ((-1) ** (n + 1)).astype(int)
    S    = np.where(n == 1, 1.0, 0.0)

    def __init__(self, t_0, time, T_0, A, alpha):
        self.t_0   = t_0
        self.t_f   = time[-1]
        self.E_0   = T_0**4 * 7.57e-15          # radiation energy density [erg/cm³]
        self.A     = A
        self.alpha = alpha
        self.t     = time

        # Precompute fixed (n, t) arrays — these never change for a given instance.
        # No interpolation grids are built; calc_lum calls sug_vec directly.
        self.gamma_factor      = -0.5 * (np.pi * self.n * self.t) ** 2 / t_0  # (500, 60)
        self.gamma_K_nt_factor =  0.5 * (np.pi * self.n) ** 2 * t_0            # (500, 1)

        # Scalar factor used in every calc_lum call
        self.A_n_factor = (
            self.n**(alpha - 3) * self.sign
            * np.pi**(alpha - 3) * 2**0.5 / 2**(alpha / 2)
            * A * t_0**(alpha / 2) / self.E_0
        )

    def calc_lum(self, v_max: float, k: float, M: float) -> np.ndarray:
        c     = 3e10
        rho_0 = M / (4/3 * np.pi * (v_max * self.t_0)**3)
        tau_0 = 3 * k * rho_0 * (v_max * self.t_0)**2 / c
        A_n   = self.A_n_factor * tau_0**(1 - self.alpha/2) * rho_0
        s     = 1 - self.alpha / 2
        cos_a = np.cos(np.pi * 0.5 * self.alpha)

        # Two direct sug evaluations replace both the 3-D RGI (76% of old cost)
        # and the 1-D interp1d (13% of old cost).
        # F_K: (500, 1) — no time dependence, very cheap
        F_K = cos_a * scaled_upper_gamma_vec(s, -self.gamma_K_nt_factor / tau_0)
        # f:   (500, 60) — main cost; replaces the 1-D interp over 30 k points
        f   = cos_a * scaled_upper_gamma_vec(s,  self.gamma_factor / tau_0)

        exp_term = np.exp((self.gamma_factor + self.gamma_K_nt_factor) / tau_0)
        phi = exp_term * (self.S - A_n * F_K) + A_n * f
        T   = self.sign * self.n * phi

        return (
            np.sum(T, axis=0)
            * 4 * np.pi**2 * c * v_max * 2**0.5 * self.t_0 * self.E_0
            / (3 * k * rho_0)
        )

# Alternative implementation of luminosity class (DiffusionLum) using interpolation

# This can speed up a little but can introduce significant errors.
# The key observation is that sug(1−α/2, −γ_Knt/τ₀) depends only on n and τ₀,
# not on t. The t-dependence enters only through the exp prefactor — which is
# already computed in solution() as exp[(γ_f + γ_Knt)/τ₀]
# The direct formula is preferable - no interpolation grids.

# ---------------------------------------------------------------------------
# Module-level cache for DiffusionLum_interp objects.
#
# DiffusionLum_interp.__init__ is expensive: it builds a 300-point 1-D interpolation
# table and a 50x100x50 = 250,000-point 3-D RegularGridInterpolator. These
# depend only on (t_0, T_0, A_eff, alpha_eff, times), not on per-call ejecta
# parameters such as opacity or mass.  In parameter-estimation / MCMC workflows
# (thousands of calls at fixed t_0, T_0, and time grid) this rebuild dominated
# the ricigliano_lippold model runtime.  The cache eliminates it entirely after
# the first call at a given parameter combination.
# ---------------------------------------------------------------------------
_DIFFLUM_CACHE: dict = {}

def _difflum_cache_key(t_0: float, times: np.ndarray, T_0: float,
                       A: float, alpha: float) -> str:
    """Stable MD5 key from DiffusionLum constructor arguments."""
    return hashlib.md5(
        b"%r|%r|%r|%r|%r" % (t_0, tuple(times.tolist()), T_0, float(A), float(alpha))
    ).hexdigest()
    
class DiffusionLum_interp(object):
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
            * np.power(t_0, self.alpha / 2)
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
        )  

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

# ---------------------------------------------------------------------------
# Choose here which DiffusionLum
# ---------------------------------------------------------------------------
#DiffusionLum = DiffusionLum_interp 
DiffusionLum = DiffusionLum_direct # default

