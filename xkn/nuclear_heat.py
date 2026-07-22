"""
nuclear_heat.py
===============
Nuclear heating-rate models for kilonova ejecta.

Each model is exposed as a standalone function (``heat_rate_*``) and
collected under the :class:`NuclearHeat` dispatcher class.

Supported models
----------------
``"RP"``
    Ricigliano/Perego: power-law fits to Skynet nuclear-network results
    (Perego et al. 2021).  Parameters are interpolated from a 3-D table
    ``(τ, s, Y_e)`` via :class:`SkynetFits`.

``"PBR"``
    Perego, Bernuzzi & Radice (2017 ApJL): analytic fit with a logistic
    nuclear-efficiency correction.

``"LR"``
    Lippuner & Roberts (2016 ApJ): tabulated heating fits, read from
    ``interp_tables/hires_sym0_results``.

``"K"``
    Korobkin (2015): simple power-law heating.
"""

import os
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .utils import smoothclamp, oneoverpi, sec2day, day2sec


# ---------------------------------------------------------------------------
# Dispatcher class
# ---------------------------------------------------------------------------

class NuclearHeat:
    """Dispatcher that routes calls to the selected heating-rate function.

    Parameters
    ----------
    heat_model : str
        One of ``{"RP", "PBR", "LR", "K"}``.

    Raises
    ------
    SystemExit
        If *heat_model* is not recognised.
    """

    def __init__(self, heat_model: str) -> None:
        _models = {
            "RP":  heat_rate_RP,
            "PBR": heat_rate_PBR,
            "LR":  heat_rate_LR,
            "K":   heat_rate_K,
        }
        if heat_model not in _models:
            sys.exit(
                f"Unknown heating-rate model: {heat_model!r}\n"
                "Please use:\n"
                '  "RP"  for Perego et al 2021\n'
                '  "PBR" for Perego et al 2017 ApJL\n'
                '  "LR"  for Lippuner & Roberts 2016 ApJ\n'
                '  "K"   for Korobkin 2015'
            )
        self.heat_rate = _models[heat_model]

    def __call__(self, times, omegas, mass_ej, vel_rms, alpha, t0eps, sigma0, eps0,
                 cnst_eff, idx_eff, thermalization, kappa_2_ye, heating_function,
                 **kwargs):
        """Evaluate the nuclear heating rate.

        All positional arguments are forwarded unchanged to the selected
        ``heat_rate_*`` function.  See the individual functions for full
        parameter descriptions.
        """
        return self.heat_rate(
            times, omegas, mass_ej, vel_rms, alpha, t0eps, sigma0, eps0,
            cnst_eff, idx_eff, thermalization, kappa_2_ye, heating_function,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Ricigliano / Perego (RP) model
# ---------------------------------------------------------------------------

class SkynetFits:
    """3-D interpolation table of Skynet nuclear-network fit parameters.

    The table is read once at class-definition time from
    ``interp_tables/epsdatafit.dat`` and stored as
    :class:`~scipy.interpolate.RegularGridInterpolator` objects
    :attr:`A_interp` and :attr:`alpha_interp`.

    Attributes
    ----------
    A_interp : RegularGridInterpolator
        Interpolator for the heating amplitude *A* over the grid
        ``(τ [ms], s [k_B/baryon], Y_e)``.
    alpha_interp : RegularGridInterpolator
        Interpolator for the heating power-law index *α* over the same
        grid.
    """

    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "interp_tables",
        "epsdatafit.dat",
    )
    tau_raw, entropy_raw, ye_raw, A_raw, alpha_raw = np.loadtxt(
        filename, unpack=True, usecols=(0, 1, 2, 3, 4)
    )

    taus      = np.unique(tau_raw)
    entropys  = np.unique(entropy_raw)
    yes       = np.unique(ye_raw)

    As     = A_raw.reshape((len(taus), len(entropys), len(yes)))
    alphas = alpha_raw.reshape((len(taus), len(entropys), len(yes)))

    _kw = dict(bounds_error=False, fill_value=None, method="linear")
    A_interp = RegularGridInterpolator((taus, entropys, yes), As, **_kw)
    alpha_interp = RegularGridInterpolator((taus, entropys, yes), alphas, **_kw)


def skynet_heating_params(
    ye: float,
    s: float,
    tau: float,
) -> tuple:
    """Return Skynet heating parameters for a single ``(Y_e, s, τ)`` point.

    Parameters
    ----------
    ye : float
        Electron fraction.
    s : float
        Entropy [k_B/baryon].
    tau : float
        Expansion timescale [ms].

    Returns
    -------
    A : float
        Heating amplitude [erg/s/g].
    alpha : float
        Heating power-law index.
    """
    A     = SkynetFits.A_interp((tau, s, ye))
    alpha = SkynetFits.alpha_interp((tau, s, ye))
    return A * day2sec ** alpha, alpha


def skynet_heating_params_batch(
    ye_arr: np.ndarray,
    s_arr: np.ndarray,
    tau_arr: np.ndarray,
) -> tuple:
    """Vectorized heating parameters for all angular bins in one call.

    Replaces a scalar loop over angular bins with a single batched
    :class:`~scipy.interpolate.RegularGridInterpolator` query on an
    ``(n_angles, 3)`` array, eliminating ``n_angles − 1`` redundant
    Python→C dispatch overheads.

    Parameters
    ----------
    ye_arr : array_like, shape (n_angles,)
        Electron fractions.
    s_arr : array_like, shape (n_angles,)
        Entropies [k_B/baryon].
    tau_arr : array_like, shape (n_angles,)
        Expansion timescales [ms].

    Returns
    -------
    A_arr : ndarray, shape (n_angles,)
        Heating amplitudes [erg/s/g].
    alpha_arr : ndarray, shape (n_angles,)
        Heating power-law indices.
    """
    ye_arr  = np.asarray(ye_arr,  dtype=float)
    s_arr   = np.asarray(s_arr,   dtype=float)
    tau_arr = np.asarray(tau_arr, dtype=float)
    pts     = np.column_stack([tau_arr, s_arr, ye_arr])   # (n_angles, 3)
    A_arr     = SkynetFits.A_interp(pts)
    alpha_arr = SkynetFits.alpha_interp(pts)
    return A_arr * day2sec ** alpha_arr, alpha_arr


def heat_rate_RP(
    times, omegas, mass_ej, vel_rms, alpha, t0eps, sigma0, eps0,
    cnst_eff, idx_eff, thermalization, kappa_2_ye, heating_function,
    **kwargs,
):
    """Ricigliano/Perego nuclear heating rate (Perego et al. 2021).

    Parameters
    ----------
    times : ndarray, shape (n_times,)
        Observer times [s].
    omegas : ndarray, shape (n_angles,)
        Solid-angle widths per angular bin [sr].
    mass_ej : ndarray, shape (n_angles,)
        Ejecta mass per bin [Msun].
    vel_rms : ndarray, shape (n_angles,)
        RMS velocity per bin [c].
    thermalization : callable
        Thermalization efficiency object (see :mod:`thermalization`).
    **kwargs
        Must include ``ye``, ``s``, ``tau`` as arrays of shape
        ``(n_angles,)``.

    Returns
    -------
    ndarray, shape (n_angles, n_times)
        Specific heating rate ``ε_nuc · ε_th`` [erg/s/g].
    """
    # Batched RGI call replaces a scalar loop over n_angles
    A, alpha = skynet_heating_params_batch(kwargs["ye"], kwargs["s"], kwargs["tau"])
    times_grid, alpha = np.meshgrid(times, alpha)
    if np.isscalar(A):
        A = A[None]

    eps_th = thermalization(
        times=times, omegas=omegas, mass_ej=mass_ej, vel=vel_rms,
        cnst_eff=cnst_eff, idx_eff=idx_eff, **kwargs,
    )
    # eps_th is already an ndarray; no redundant np.array() wrapping needed
    return eps_th * A[:, None] / times_grid ** alpha


# ---------------------------------------------------------------------------
# Perego, Bernuzzi & Radice (PBR) model
# ---------------------------------------------------------------------------

def calc_eps_nuc(
    times_day: np.ndarray,
    kappas: np.ndarray,
    a_eps_nuc: float,
    b_eps_nuc: float,
    t_eps_nuc: float,
) -> np.ndarray:
    """Nuclear heating efficiency factor (Martin et al. 2015 / Perego et al. 2017).

    Parameters
    ----------
    times_day : ndarray, shape (n_times,)
        Observer times [days].
    kappas : ndarray, shape (n_angles,)
        Opacities per angular bin [cm²/g].
    a_eps_nuc, b_eps_nuc, t_eps_nuc : float
        Fitting constants.

    Returns
    -------
    ndarray, shape (n_angles, n_times)
    """
    tmp = np.zeros_like(times_day)
    mask = 4.0 * times_day - 4.0 > -20
    tmp[mask]               = 4.0 * times_day[mask] - 4.0
    tmp[~mask]              = -20.0
    tmp[tmp > 20.0]         = 20.0
    tmp = a_eps_nuc + b_eps_nuc / (1.0 + np.exp(tmp))
    tmp, weight = np.meshgrid(tmp, smoothclamp(kappas, 1.0, 10.0, 1.0, 0.0))
    return (1.0 - weight) + weight * tmp


def heat_rate_PBR(
    times, omegas, mass_ej, vel_rms, alpha, t0eps, sigma0, eps0,
    cnst_eff, idx_eff, thermalization, kappa_2_ye, heating_function,
    **kwargs,
):
    """Perego, Bernuzzi & Radice (2017 ApJL) nuclear heating rate.

    Parameters
    ----------
    times : ndarray, shape (n_times,)
        Observer times [s].
    eps0 : float
        Base specific heating rate [erg/s/g].
    thermalization : callable
        Thermalization efficiency object.
    **kwargs
        Must include ``opacity``, ``cnst_a_eps_nuc``, ``cnst_b_eps_nuc``,
        ``cnst_t_eps_nuc``.

    Returns
    -------
    ndarray, shape (n_angles, n_times)
    """
    eps_nuc = calc_eps_nuc(
        times * sec2day,
        kwargs["opacity"],
        kwargs["cnst_a_eps_nuc"],
        kwargs["cnst_b_eps_nuc"],
        kwargs["cnst_t_eps_nuc"],
    )
    eps_th = thermalization(
        times=times, omegas=omegas, mass_ej=mass_ej, vel=vel_rms,
        cnst_eff=cnst_eff, idx_eff=idx_eff, **kwargs,
    )
    return eps0 * (
        (0.5 - oneoverpi * np.arctan((times - t0eps) / sigma0)) ** alpha
        * (2.0 * eps_nuc * eps_th)
    )


# ---------------------------------------------------------------------------
# Lippuner & Roberts (LR) model
# ---------------------------------------------------------------------------

def heat_rate_LR(
    times, omegas, mass_ej, vel_rms, alpha, t0eps, sigma0, eps0,
    cnst_eff, idx_eff, thermalization, kappa_2_ye, heating_function,
    **kwargs,
):
    """Lippuner & Roberts (2016 ApJ) nuclear heating rate.

    Parameters
    ----------
    times : ndarray, shape (n_times,)
        Observer times [s].
    heating_function : callable
        Tabulated heating function ``f(Y_e, t_days)`` from
        :class:`~heating_function.HeatingFunction`.
    kappa_2_ye : callable
        Opacity-to-electron-fraction converter from
        :class:`~kappa_2_ye.Kappa2Ye`.
    **kwargs
        Must include ``opacity``.

    Returns
    -------
    ndarray, shape (n_angles, n_times)
    """
    eps_nuc = 10.0 ** heating_function(
        kappa_2_ye("opacity", kwargs["opacity"]), times * sec2day
    )
    eps_th = thermalization(
        times=times, omegas=omegas, mass_ej=mass_ej, vel=vel_rms,
        cnst_eff=cnst_eff, idx_eff=idx_eff, **kwargs,
    )
    return (2.0 * eps0 / 2.0e18) * eps_nuc * eps_th


# ---------------------------------------------------------------------------
# Korobkin (K) model
# ---------------------------------------------------------------------------

def heat_rate_K(
    times, omegas, mass_ej, vel_rms, alpha, t0eps, sigma0, eps0,
    cnst_eff, idx_eff, thermalization, kappa_2_ye, heating_function,
    **kwargs,
):
    """Korobkin (2015) nuclear heating rate.

    Parameters
    ----------
    times : ndarray, shape (n_times,)
        Observer times [s].
    eps0 : float
        Base specific heating rate [erg/s/g].
    thermalization : callable
        Thermalization efficiency object.

    Returns
    -------
    ndarray, shape (n_angles, n_times)
    """
    eps_th = thermalization(
        times=times, omegas=omegas, mass_ej=mass_ej, vel=vel_rms,
        cnst_eff=cnst_eff, idx_eff=idx_eff, **kwargs,
    )
    return (
        eps0
        * (0.5 - oneoverpi * np.arctan((times - t0eps) / sigma0)) ** alpha
        * (2.0 * eps_th)
    )
