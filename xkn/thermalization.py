"""
thermalization.py
=================
Thermalization efficiency models for kilonova ejecta.

The main entry point is the :class:`Thermalization` class, which acts as a
factory: the constructor selects the appropriate efficiency function based on
the ``therm_model`` string, and calling the instance dispatches to that
function.

Supported models
----------------
``"BKWM"`` / ``"BKWM_dens"``
    Barnes, Kasen, Wu & Martínez-Pinedo (2016) 2-D table interpolation in
    ``(velocity, log10-mass)`` space.  Uses
    :meth:`~Thermalization.therm_efficiency_params_2d`.

``"BKWM_1d"``
    1-D variant of the BKWM model interpolating in the combined variable
    ``x = (4π/ω) · m / v²``.  Uses
    :meth:`~Thermalization.therm_efficiency_params_1d`.

``"power_law"``
    Simple power-law: ``ε_th = cnst_eff / t^{idx_eff}``.

``"cnst"``
    Constant thermalization efficiency ``ε_th = cnst_eff ∈ [0, 1]``.
"""

import sys

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator as RGI

from . import extrapolation_2d as expol
from . import utils


class Thermalization:
    """Factory class for thermalization efficiency models.

    Parameters
    ----------
    therm_model : str
        Name of the thermalization model.  One of
        ``{"BKWM", "BKWM_dens", "BKWM_1d", "power_law", "cnst"}``.

    Raises
    ------
    SystemExit
        If *therm_model* is not one of the recognised strings.
    """

    def __init__(self, therm_model: str) -> None:
        if therm_model in ("BKWM", "BKWM_dens"):
            self.therm_efficiency        = BKWM_therm_efficiency
            self.therm_efficiency_params = self.therm_efficiency_params_2d

            # Table data from Barnes et al. (2016), Table 1
            # x-axis: log10(mass / Msun), y-axis: velocity / c
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
            # RegularGridInterpolator replaces the removed interp2d
            # (dropped in SciPy 1.14). Table axes: (velocity, log10-mass).
            # Element-wise query evaluates exactly n_angles points instead of
            # the n_angles² outer-product grid that interp2d produced.
            self._rgi_a = RGI(
                (y, x), np.array(a), method="linear",
                bounds_error=False, fill_value=None,
            )
            self._rgi_b = RGI(
                (y, x), np.array(b), method="linear",
                bounds_error=False, fill_value=None,
            )
            self._rgi_d = RGI(
                (y, x), np.array(d), method="linear",
                bounds_error=False, fill_value=None,
            )

        elif therm_model == "BKWM_1d":
            self.therm_efficiency        = BKWM_therm_efficiency
            self.therm_efficiency_params = self.therm_efficiency_params_1d

            # Table data from Barnes et al. (2016)
            x_barnes = [0.011, 0.025, 0.0556, 0.1, 0.111, 0.125, 0.25,
                        0.5,   0.5556, 1.0,   1.25, 5.0]
            a_barnes = [8.16, 4.52, 3.20, 2.01, 2.19, 1.90, 1.31,
                        0.81, 0.95,  0.56, 0.55, 0.27]
            b_barnes = [1.19, 0.62, 0.45, 0.28, 0.31, 0.28, 0.21,
                        0.19, 0.15,  0.17, 0.13, 0.10]
            d_barnes = [1.52, 1.39, 1.39, 1.12, 1.32, 1.21, 1.13,
                        0.86, 1.13,  0.74, 0.90, 0.60]

            kw = dict(bounds_error=False)
            self.fa_1d = interp1d(x_barnes, a_barnes, fill_value=(a_barnes[0], a_barnes[-1]), **kw)
            self.fb_1d = interp1d(x_barnes, b_barnes, fill_value=(b_barnes[0], b_barnes[-1]), **kw)
            self.fd_1d = interp1d(x_barnes, d_barnes, fill_value=(d_barnes[0], d_barnes[-1]), **kw)

        elif therm_model == "power_law":
            self.therm_efficiency = power_law_therm_efficiency

        elif therm_model == "cnst":
            self.therm_efficiency = cnst_therm_efficiency

        else:
            sys.exit(
                f"Unknown thermalization efficiency model: {therm_model!r}\n"
                "Supported models: 'BKWM', 'BKWM_dens', 'BKWM_1d', 'power_law', 'cnst'"
            )

    def __call__(self, **kwargs):
        """Dispatch to the selected thermalization efficiency function."""
        return self.therm_efficiency(self, **kwargs)

    # ------------------------------------------------------------------
    # Element-wise RGI helper
    # ------------------------------------------------------------------

    @staticmethod
    def _eval_rgi(
        rgi: RGI,
        xn: np.ndarray,
        yn: np.ndarray,
    ) -> np.ndarray:
        """Query *rgi* element-wise at paired ``(xn[i], yn[i])`` points.

        The table axes are ``(velocity, log10-mass) = (y, x)``.  Stacking
        ``[yn, xn]`` column-wise produces a ``(n_angles, 2)`` query array,
        evaluating exactly ``n_angles`` points.

        Parameters
        ----------
        rgi : RegularGridInterpolator
            The 2-D interpolator to query.
        xn : array_like
            Log10-mass values.
        yn : array_like
            Velocity values.

        Returns
        -------
        numpy.ndarray, shape (n_angles,)
        """
        xn, yn = np.atleast_1d(xn), np.atleast_1d(yn)
        return rgi(np.column_stack([yn, xn]))

    def fa(self, xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
        """Interpolate coefficient *a* at ``(log10-mass, velocity)`` pairs."""
        return self._eval_rgi(self._rgi_a, xn, yn)

    def fb(self, xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
        """Interpolate coefficient *b* at ``(log10-mass, velocity)`` pairs."""
        return self._eval_rgi(self._rgi_b, xn, yn)

    def fd(self, xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
        """Interpolate coefficient *d* at ``(log10-mass, velocity)`` pairs."""
        return self._eval_rgi(self._rgi_d, xn, yn)

    # ------------------------------------------------------------------
    # Parameter retrievers
    # ------------------------------------------------------------------

    def therm_efficiency_params_1d(
        self,
        omegas: np.ndarray,
        mass_ej: np.ndarray,
        vel: np.ndarray,
    ) -> list:
        """Return BKWM coefficients via 1-D interpolation in ``x = m / v²``.

        Parameters
        ----------
        omegas : ndarray, shape (n_angles,)
            Solid-angle widths of each angular bin [sr].
        mass_ej : ndarray, shape (n_angles,)
            Ejecta mass per bin [Msun].
        vel : ndarray, shape (n_angles,)
            RMS velocity per bin [c].

        Returns
        -------
        list of ndarray
            ``[a, b, d]``, each shape ``(n_angles,)``.
        """
        xnew = utils.fourpi / omegas * mass_ej / vel ** 2
        return [np.array(func(xnew)) for func in [self.fa_1d, self.fb_1d, self.fd_1d]]

    def therm_efficiency_params_2d(
        self,
        omegas: np.ndarray,
        mass_ej: np.ndarray,
        vel: np.ndarray,
    ) -> list:
        """Return BKWM coefficients via 2-D interpolation in ``(log10-mass, vel)``.

        Parameters
        ----------
        omegas : ndarray, shape (n_angles,)
            Solid-angle widths of each angular bin [sr].
        mass_ej : ndarray, shape (n_angles,)
            Ejecta mass per bin [Msun].
        vel : ndarray, shape (n_angles,)
            RMS velocity per bin [c].

        Returns
        -------
        list of ndarray
            ``[a, b, d]``, each shape ``(n_angles,)``.
        """
        xnew = np.log10(utils.fourpi / omegas * mass_ej)   # log10-mass [Msun]
        ynew = vel                                           # velocity   [c]
        return [func(xnew, ynew) for func in [self.fa, self.fb, self.fd]]


# ---------------------------------------------------------------------------
# Efficiency functions
# ---------------------------------------------------------------------------

def BKWM_therm_efficiency(cls: Thermalization, **kwargs) -> np.ndarray:
    """Barnes, Kasen, Wu & Martínez-Pinedo (2016) thermalization efficiency.

    Computes the angle- and time-resolved thermalization efficiency using
    the fitting formula from Barnes et al. (2016)::

        ε_th(t) = 0.36 · [exp(−a · t_d)
                           + ln(1 + 2 b t_d^d) / (2 b t_d^d)]

    where *t_d* is time in days and ``(a, b, d)`` are tabulated
    coefficients depending on the ejecta mass and velocity.

    Parameters
    ----------
    cls : Thermalization
        Thermalization instance carrying the interpolators and the
        coefficient-lookup method.
    **kwargs
        Must contain ``times``, ``omegas``, ``mass_ej``, ``vel``.

    Returns
    -------
    numpy.ndarray, shape (n_angles, n_times)
        Thermalization efficiency per angular bin and time step.

    Raises
    ------
    SystemExit
        If any of the required keyword arguments is ``None``.
    """
    required = ("times", "omegas", "mass_ej", "vel")
    if any(kwargs.get(k) is None for k in required):
        sys.exit(
            "BKWM_therm_efficiency: must supply times, omegas, mass_ej, vel.\n"
        )

    coeffs     = cls.therm_efficiency_params(kwargs["omegas"], kwargs["mass_ej"], kwargs["vel"])
    times_days = kwargs["times"] * utils.sec2day

    # Broadcast coeffs[0] (shape n_angles) against times_days (shape n_times)
    _, times_days = np.meshgrid(coeffs[0], times_days)
    tmp = 2.0 * coeffs[1] * times_days ** coeffs[2]
    tmp = 0.36 * (np.exp(-coeffs[0] * times_days) + np.log(1.0 + tmp) / tmp)
    return tmp.T   # (n_angles, n_times)


def power_law_therm_efficiency(cls: Thermalization, **kwargs) -> np.ndarray:
    """Power-law thermalization: ``ε_th = cnst_eff / t^{idx_eff}``.

    Parameters
    ----------
    cls : Thermalization
        Unused; kept for a uniform calling convention.
    **kwargs
        Must contain ``times``, ``cnst_eff``, ``idx_eff``.

    Returns
    -------
    numpy.ndarray, shape (n_times,)

    Raises
    ------
    SystemExit
        If any required keyword argument is ``None``.
    """
    if any(kwargs.get(k) is None for k in ("times", "cnst_eff", "idx_eff")):
        sys.exit(
            "power_law_therm_efficiency: must supply times, cnst_eff, idx_eff.\n"
        )
    return kwargs["cnst_eff"] / kwargs["times"] ** kwargs["idx_eff"]


def cnst_therm_efficiency(cls: Thermalization, **kwargs) -> float:
    """Constant thermalization efficiency ``ε_th = cnst_eff ∈ [0, 1]``.

    Parameters
    ----------
    cls : Thermalization
        Unused; kept for a uniform calling convention.
    **kwargs
        Must contain ``cnst_eff`` with a value in ``[0, 1]``.

    Returns
    -------
    float

    Raises
    ------
    SystemExit
        If ``cnst_eff`` is ``None`` or outside ``[0, 1]``.
    """
    cnst = kwargs.get("cnst_eff")
    if cnst is None or not (0.0 <= cnst <= 1.0):
        sys.exit(
            f'cnst_therm_efficiency: "cnst_eff" = {cnst} is invalid. '
            "Must be a float in [0, 1].\n"
        )
    return cnst
