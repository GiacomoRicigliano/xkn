import numpy as np
import os
import sys
from scipy.interpolate import RegularGridInterpolator

from .utils import smoothclamp, oneoverpi, sec2day, day2sec


class NuclearHeat(object):

    def __init__(self, heat_model):
        if heat_model == "RP":
            self.heat_rate = heat_rate_RP
        elif heat_model == "PBR":
            self.heat_rate = heat_rate_PBR
        elif heat_model == "LR":
            self.heat_rate = heat_rate_LR
        elif heat_model == "K":
            self.heat_rate = heat_rate_K
        else:
            sys.exit(
                "Wrong input name for heating rate model\n"
                + "Please use:\n"
                + '"RP" for Perego et al 2021\n'
                + '"PBR" for Perego et al 2017 ApJL\n'
                + '"LR" for Lippuner & Roberts 2016 ApJ\n'
                + '"K" for Korobkin 2015'
            )

    def __call__(
        self,
        times,
        omegas,
        mass_ej,
        vel_rms,
        alpha,
        t0eps,
        sigma0,
        eps0,
        cnst_eff,
        idx_eff,
        thermalization,
        kappa_2_ye,
        heating_function,
        **kwargs
    ):
        return self.heat_rate(
            times,
            omegas,
            mass_ej,
            vel_rms,
            alpha,
            t0eps,
            sigma0,
            eps0,
            cnst_eff,
            idx_eff,
            thermalization,
            kappa_2_ye,
            heating_function,
            **kwargs
        )


########
# Ricigliano heating rates (implemented by G. Ricigliano, based on Perego et al 2021)
########
class SkynetFits(object):
    # file reading
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "interp_tables",
        "skynet_fit_parameters.dat",
    )
    tau_raw, entropy_raw, ye_raw, A_raw, alpha_raw = np.loadtxt(
        filename, unpack=True, usecols=(0, 1, 2, 3, 4)
    )

    taus = np.unique(tau_raw)
    entropys = np.unique(entropy_raw)
    yes = np.unique(ye_raw)

    As = A_raw.reshape((len(taus), len(entropys), len(yes)))
    alphas = alpha_raw.reshape((len(taus), len(entropys), len(yes)))

    # Linear 3D interpolation of the parameters
    A_interp = RegularGridInterpolator(
        (taus, entropys, yes), As, bounds_error=False, fill_value=None, method="nearest"
    )
    alpha_interp = RegularGridInterpolator(
        (taus, entropys, yes),
        alphas,
        bounds_error=False,
        fill_value=None,
        method="nearest",
    )


# function calculating touple of parameters for given set of inputs
def skynet_heating_params(ye, s, tau):  # units: s[k_B/baryon] tau[ms]
    A = SkynetFits.A_interp((tau, s, ye))
    alpha = SkynetFits.alpha_interp((tau, s, ye))
    return A * day2sec**alpha, alpha  # units: A[erg/s/g]


def heat_rate_RP(
    times,
    omegas,
    mass_ej,
    vel_rms,
    alpha,
    t0eps,
    sigma0,
    eps0,
    cnst_eff,
    idx_eff,
    thermalization,
    kappa_2_ye,
    heating_function,
    **kwargs
):
    A, alpha = skynet_heating_params(kwargs["ye"], kwargs["s"], kwargs["tau"])
    times_grid, alpha = np.meshgrid(times, alpha)
    if np.isscalar(A):
        A = A[None]
    eps_th = thermalization(
        times=times,
        omegas=omegas,
        mass_ej=mass_ej,
        vel=vel_rms,
        cnst_eff=cnst_eff,
        idx_eff=idx_eff,
        **kwargs
    )
    return np.array(eps_th) * A[:, None] / times_grid**alpha


########
# Perego heating rates (implemented by A. Perego, based on Martin et al 2015)
########
def calc_eps_nuc(times_day, kappas, a_eps_nuc, b_eps_nuc, t_eps_nuc):
    tmp = np.zeros_like(times_day)
    mask = 4.0 * times_day - 4.0 > -20
    tmp[mask] = 4.0 * times_day[mask] - 4.0
    tmp[np.logical_not(mask)] = -20.0
    tmp[tmp > 20.0] = 20.0
    tmp = a_eps_nuc + b_eps_nuc / (1.0 + np.exp(tmp))  # t_eps_nuc still missing!
    tmp, weight = np.meshgrid(tmp, smoothclamp(kappas, 1.0, 10.0, 1.0, 0.0))
    return (1.0 - weight) + weight * tmp


def heat_rate_PBR(
    times,
    omegas,
    mass_ej,
    vel_rms,
    alpha,
    t0eps,
    sigma0,
    eps0,
    cnst_eff,
    idx_eff,
    thermalization,
    kappa_2_ye,
    heating_function,
    **kwargs
):
    eps_nuc = calc_eps_nuc(
        times * sec2day,
        kwargs["opacity"],
        kwargs["cnst_a_eps_nuc"],
        kwargs["cnst_b_eps_nuc"],
        kwargs["cnst_t_eps_nuc"],
    )
    eps_th = thermalization(
        times=times,
        omegas=omegas,
        mass_ej=mass_ej,
        vel=vel_rms,
        cnst_eff=cnst_eff,
        idx_eff=idx_eff,
        **kwargs
    )
    return eps0 * (
        (0.5 - oneoverpi * np.arctan((times - t0eps) / sigma0)) ** alpha
        * (2.0 * eps_nuc * eps_th)
    )


########
# Lippuner & Roberts heating rates (implemented by D. Vescovi)
########
def heat_rate_LR(
    times,
    omegas,
    mass_ej,
    vel_rms,
    alpha,
    t0eps,
    sigma0,
    eps0,
    cnst_eff,
    idx_eff,
    thermalization,
    kappa_2_ye,
    heating_function,
    **kwargs
):
    # convert the times from seconds to days
    # get the ye from the opacity according to Tanaka et al 2019
    eps_nuc = 10.0 ** heating_function(
        kappa_2_ye("opacity", kwargs["opacity"]), times * sec2day
    )
    eps_th = thermalization(
        times=times,
        omegas=omegas,
        mass_ej=mass_ej,
        vel=vel_rms,
        cnst_eff=cnst_eff,
        idx_eff=idx_eff,
        **kwargs
    )
    return (2.0 * eps0 / 2.0e18) * eps_nuc * eps_th


########
# Korobkin heating rates (implemented by A. Perego)
########
def heat_rate_K(
    times,
    omegas,
    mass_ej,
    vel_rms,
    alpha,
    t0eps,
    sigma0,
    eps0,
    cnst_eff,
    idx_eff,
    thermalization,
    kappa_2_ye,
    heating_function,
    **kwargs
):
    eps_th = thermalization(
        times=times,
        omegas=omegas,
        mass_ej=mass_ej,
        vel=vel_rms,
        cnst_eff=cnst_eff,
        idx_eff=idx_eff,
        **kwargs
    )
    return (
        eps0
        * (0.5 - oneoverpi * np.arctan((times - t0eps) / sigma0)) ** alpha
        * (2.0 * eps_th)
    )
