"""
mkn.py
======
Top-level kilonova model class (:class:`MKN`) and helper utilities.

:class:`MKN` orchestrates the full light-curve pipeline:

1. Angular discretisation of the ejecta via
   :class:`~angular_distribution.AngularDistribution`.
2. Per-component bolometric luminosity via :class:`~ejecta.Ejecta`.
3. Projection onto the observer sky plane via
   :class:`~utils.ObserverProjection`.
4. Broadband magnitude computation via :mod:`filters`.
"""

import sys
import logging
from copy import deepcopy
from warnings import filterwarnings

filterwarnings("ignore")

import numpy as np

from . import filters as flt
from .angular_distribution import AngularDistribution
from .ejecta import Ejecta
from .plotting import plot_magnitudes
from .utils import (
    Mpc2cm,
    ObserverProjection,
    init_times,
    time_safe,
    check_dict_variables,
    Redshift,
)


class MKN:
    """Top-level kilonova model class.

    Orchestrates the full multi-component kilonova pipeline from ejecta
    parameters to broadband magnitudes and log-likelihood.

    Parameters
    ----------
    shell_params : dict
        Per-component parameter dictionaries keyed by component name
        (e.g. ``"dynamical"``, ``"secular"``, ``"wind"``).  Each value is
        a dict of component-level parameters as read from the config file.
    glob_params : dict
        Global model parameters (angular grid, filter settings, time grid,
        heating model constants, …).
    inj_dict : dict or None, optional
        If provided, generates synthetic injection data from a secondary
        MKN run and uses it as mock observations.  Keys: ``shell_params``,
        ``glob_params``, ``mkn_vars``, ``seed``, ``sigma_min``,
        ``sigma_max``.
    log_name : str, optional
        Logger name (default ``"MKN"``).
    log_level : str, optional
        Logging level string (default ``"INFO"``).

    Attributes
    ----------
    ejecta : Ejecta
        Multi-component ejecta object.
    angles : ndarray, shape (n_angles, 2)
        Lower and upper polar-angle boundaries for each angular bin [rad].
    omegas : ndarray, shape (n_angles,)
        Solid-angle width of each bin [sr].
    times : ndarray
        Full observer time array [s].
    glob_params : dict
        Copy of the global parameter dictionary.
    lams : list
        Filter wavelengths [nm] used for magnitude computation.
    mag : dict
        Observed magnitude data (``None`` when ``filter_usage`` is
        ``"properties"``).
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        shell_params: dict,
        glob_params: dict,
        inj_dict=None,
        log_name: str = "MKN",
        log_level: str = "INFO",
    ) -> None:
        self.set_logger(name=log_name, level=log_level)
        self.set_ejecta(list(shell_params.keys()), shell_params)
        self.set_glob_params(glob_params)
        self.gen_inj_data(inj_dict)
        self.logger.info("--- MKN object fully initialized. ---")

    # ------------------------------------------------------------------
    # Setters used during initialisation
    # ------------------------------------------------------------------

    def set_logger(self, name="MKN", level="INFO", stdout=True, logfile=None):
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s : %(message)s"
        )
        if stdout:
            logging.basicConfig(stream=sys.stdout)
        if logfile is not None:
            logging.basicConfig(filename=logfile, filemode="w")
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

    def set_ejecta(self, shell_names, shell_params):
        self.shell_names = shell_names
        self.shell_params = shell_params
        self.ejecta = Ejecta(shell_names, shell_params)
        self.logger.info("Initialized ejecta.")

    def set_glob_params(self, glob_params):
        self.glob_params = glob_params
        self.set_flux_factor_func()
        self.set_angles_omegas()
        self.set_filter_data()
        self.set_redshift()
        self.set_times()

    def set_flux_factor_func(self):
        check_dict_variables(
            dic=(self.glob_params, ["slices_dist", "slices_num"]),
            logger=self.logger,
            strict=True,
            label="set_flux_factor_func",
        )
        self.flux_factor_func = ObserverProjection(
            self.glob_params["slices_num"], self.glob_params["slices_dist"]
        )
        self.logger.info("Initialized flux_factor_func.")
        self.logger.debug(
            f"   settings: slices_dist={self.glob_params['slices_dist']}, slices_num={self.glob_params['slices_num']}."
        )

    def set_angles_omegas(self):
        check_dict_variables(
            dic=(self.glob_params, ["slices_dist", "slices_num", "omega_frac"]),
            logger=self.logger,
            strict=True,
            label="set_angles_omegas",
        )
        self.angles, self.omegas = AngularDistribution(self.glob_params["slices_dist"])(
            self.glob_params["slices_num"] / 2, self.glob_params["omega_frac"]
        )
        self.logger.info("Initialized angular distribution.")
        self.logger.debug(
            f"   settings: slices_dist={self.glob_params['slices_dist']}, slices_num={self.glob_params['slices_num']}, omega_frac={self.glob_params['omega_frac']}."
        )

    def set_filter_data(self):
        if check_dict_variables(
            dic=(
                self.glob_params,
                ["filter_usage", "t_min", "t_max", "t_start_filter"],
            ),
            logger=self.logger,
            strict=False,
            label="set_filter_data",
        ):
            self.dic_filt_full, self.lams_full, self.mag_full = flt.read_filters(
                self.glob_params["filter_usage"],
                self.glob_params["filter_data_path"],
                self.glob_params["t_min"],
                self.glob_params["t_max"],
                self.glob_params["t_start_filter"],
                self.glob_params["t_type_data"],
                filter_dict=self.glob_params["filter_dictionary"],
                filter_dict_path=self.glob_params["filter_dictionary_path"],
                dered_correction=self.glob_params["dered_correction"],
                R_V=self.glob_params["R_V"],
                EBV=self.glob_params["EBV"],
                A_V=self.glob_params["A_V"],
                upper_limits=self.glob_params["upper_limits"],
            )

            self.dic_filt, self.lams, self.mag = flt.limit_mags(
                *flt.limit_lams(
                    self.dic_filt_full,
                    self.lams_full,
                    self.mag_full,
                    lam_list=self.glob_params["lam_list"],
                    lam_min=self.glob_params["lam_min"],
                    lam_max=self.glob_params["lam_max"],
                ),
                mag_min=self.glob_params["mag_min"],
                mag_max=self.glob_params["mag_max"],
            )
            self.logger.info(f"Initialized filter data from local data.")
            self.logger.debug(
                f"   settings: path={self.glob_params['filter_data_path']}."
            )
        else:
            (
                self.dic_filt_full,
                self.lams_full,
                self.mag_full,
                self.dic_filt,
                self.lams,
                self.mag,
            ) = (None, None, None, None, None, None)
            self.logger.info("Did not initialize filter data from local data.")

    def set_redshift(self):
        check_dict_variables(
            dic=(self.glob_params, ["cosmology"]),
            logger=self.logger,
            strict=True,
            label="set_redshift",
            allow_none=True,
        )
        self.redshift = Redshift(self.glob_params["cosmology"])
        self.logger.info("Initialized redshift")
        self.logger.debug(f"   settings: cosmology={self.glob_params['cosmology']}.")

    def set_times(self):
        if check_dict_variables(
            dic=(
                self.glob_params,
                ["t_scale", "t_min", "t_max", "t_num", "t_start_filter"],
            ),
            logger=None,
        ) or check_dict_variables(
            dic=(self.glob_params, ["t_scale", "t_start_filter"]),
            var=([self.mag], ["mag"]),
            logger=None,
        ):
            if check_dict_variables(dic=(self.glob_params, ["t_toll"]), logger=None):
                toll = self.glob_params["t_toll"]
            else:
                toll = 0.1
            self.times = init_times(
                self.glob_params["t_scale"],
                self.glob_params["t_min"],
                self.glob_params["t_max"],
                self.glob_params["t_num"],
                self.glob_params["t_start_filter"],
                self.glob_params["t_type_data"],
                self.mag,
                toll,
            )
            self.logger.info("Initialized times.")
            self.logger.debug(
                f"   settings: t_scale={self.glob_params['t_scale']}, t_min={self.glob_params['t_min']}, t_max={self.glob_params['t_max']}, t_num={self.glob_params['t_num']}, t_start_filter={self.glob_params['t_start_filter']}, t_toll={toll}"
            )
        else:
            check_dict_variables(
                dic=(
                    self.glob_params,
                    ["t_scale", "t_min", "t_max", "t_num", "t_start_filter"],
                ),
                var=([self.mag], ["mag"]),
                logger=self.logger,
                strict=True,
                label="set_times",
            )

    # ------------------------------------------------------------------
    # Injection synthetic-data generation
    # ------------------------------------------------------------------

    def prep_inj_mag(
        self,
        new_mags: dict,
        measures: bool = False,
        seed=None,
        sigma_min=None,
        sigma_max=None,
    ) -> dict:
        """Prepare synthetic injection magnitudes with optional noise.

        Parameters
        ----------
        new_mags : dict
            Noiseless model magnitudes from :meth:`calc_magnitudes`.
        measures : bool, optional
            Whether the time grid is derived from observational epochs.
        seed : int or None, optional
            Random seed for reproducible noise realisations.
        sigma_min, sigma_max : float or None, optional
            Bounds on the per-datapoint magnitude uncertainty.

        Returns
        -------
        dict
            Magnitude dictionary with injected noise.
        """
        mag = flt.prep_inj_mag(
            new_mags,
            mag=self.mag,
            measures=measures,
            t_start_filter=self.glob_params["t_start_filter"],
            seed=seed,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        if not self.mag:
            for lam in mag:
                mag[lam]["name"] = self.dic_filt[lam]["name"]
        return mag

    def gen_inj_data(self, inj_dict: dict) -> None:
        """Generate and store synthetic injection data (if requested).

        Parameters
        ----------
        inj_dict : dict or None
            Injection configuration.  Set to ``None`` to skip injection.
        """
        self.inj_dict = inj_dict
        if inj_dict is None:
            return
        mkn = MKN(
            inj_dict["shell_params"],
            inj_dict["glob_params"],
            log_name="INJ-MKN",
            log_level="WARNING",
        )
        measures = inj_dict["glob_params"]["t_scale"] == "measures"
        mag = mkn.prep_inj_mag(
            mkn.calc_magnitudes(inj_dict["mkn_vars"], measures=measures),
            measures=measures,
            seed=inj_dict["seed"],
            sigma_min=inj_dict["sigma_min"],
            sigma_max=inj_dict["sigma_max"],
        )
        self.dic_filt, self.lams, self.mag = flt.limit_mags(
            *flt.limit_lams(
                mkn.dic_filt,
                mkn.lams,
                mag,
                lam_list=inj_dict["glob_params"]["lam_list"],
                lam_min=inj_dict["glob_params"]["lam_min"],
                lam_max=inj_dict["glob_params"]["lam_max"],
            ),
            mag_min=inj_dict["glob_params"]["mag_min"],
            mag_max=inj_dict["glob_params"]["mag_max"],
        )
        self.logger.info("Initialized injection.")
        return mag

    # ------------------------------------------------------------------
    # Observer-frame / source-frame time helpers
    # ------------------------------------------------------------------

    def time_source(self, mkn_vars: dict) -> np.ndarray:
        """Return the source-frame time array, truncated below ``t_0``.

        Parameters
        ----------
        mkn_vars : dict
            Variable dictionary containing ``glob["distance"]``.

        Returns
        -------
        ndarray
            Source-frame times [s] with ``t < t_0`` removed.
        """
        return time_safe(
            self.times / (1.0 + self.redshift(mkn_vars["glob"]["distance"])),
            self.glob_params["t_0"],
        )

    def time_observer(self, mkn_vars: dict) -> np.ndarray:
        """Return the observer-frame time array, truncated for consistency.

        Truncates the observer times so that their length matches the
        source-frame array returned by :meth:`time_source`.

        Parameters
        ----------
        mkn_vars : dict
            Variable dictionary containing ``glob["distance"]``.

        Returns
        -------
        ndarray
            Observer-frame times [s].
        """
        return time_safe(
            self.times,
            self.glob_params["t_0"]
            * (1.0 + self.redshift(mkn_vars["glob"]["distance"])),
        )

    # ------------------------------------------------------------------
    # Core model calculations
    # ------------------------------------------------------------------

    def calc_flux_factors(self, mkn_vars: dict) -> np.ndarray:
        """Return the observer-projection flux factors for all angular bins.

        Parameters
        ----------
        mkn_vars : dict
            Variable dictionary containing ``glob["view_angle"]`` [rad].

        Returns
        -------
        ndarray, shape (n_angles,)
            Dimensionless projection factors in ``[0, 1]``.
        """
        angle = mkn_vars["glob"]["view_angle"]
        if angle > np.pi / 2:
            return self.flux_factor_func(np.degrees(np.pi - angle))
        return self.flux_factor_func(np.degrees(angle))

    def calc_lightcurve_vars(self, mkn_vars: dict) -> tuple:
        """Compute bolometric light-curve variables for all components.

        Delegates to :meth:`~ejecta.Ejecta.calc_lightcurve_vars` and
        returns a tuple of the following arrays (all indexed by
        ``(n_angles, n_times)`` unless noted):

        * ``lum_bol``      — total bolometric luminosity [erg/s]
        * ``lum_photo``    — photospheric contribution [erg/s]
        * ``radius_photo`` — photospheric radius [cm]
        * ``T_photo``      — photospheric temperature [K]
        * ``lum_shells``   — thin-shell luminosity (or ``None``)
        * ``T_shells``     — thin-shell temperature (or ``None``)
        * ``lum_bol_raw``  — total luminosity without thin-shell correction

        Parameters
        ----------
        mkn_vars : dict
            Variable dictionary.

        Returns
        -------
        tuple of ndarray
            Seven-element tuple as described above.
        """
        return self.ejecta.calc_lightcurve_vars(
            self.angles,
            self.omegas,
            self.time_source(mkn_vars),
            mkn_vars,
            mkn_vars["glob"],
            self.glob_params,
            logger=self.logger,
        )

    def calc_magnitudes(self, mkn_vars: dict, measures: bool = False) -> dict:
        """Compute broadband magnitudes in all configured filters.

        Calls :meth:`calc_lightcurve_vars` first, then passes the
        photometric output to :func:`filters.calc_magnitudes`.

        Parameters
        ----------
        mkn_vars : dict
            Variable dictionary.
        measures : bool, optional
            If ``True``, evaluate the model only at observed epochs.

        Returns
        -------
        dict
            Keys are filter wavelengths [nm]; values are sub-dicts with
            keys ``"time"`` (ndarray [s]) and ``"mag"`` (ndarray [AB mag]).
        """
        self.calc_lightcurve_vars(mkn_vars)
        return flt.calc_magnitudes(
            self.calc_flux_factors(mkn_vars),
            self.time_observer(mkn_vars),
            self.lams,
            self.dic_filt,
            mkn_vars["glob"]["distance"] * Mpc2cm,
            self.redshift(mkn_vars["glob"]["distance"]),
            self.ejecta.radius_photo,
            T_photo=self.ejecta.T_photo,
            lum_shells=self.ejecta.lum_shells,
            T_shells=self.ejecta.T_shells,
            omegas=self.omegas,
            measures=measures,
            mag=self.mag,
            t_start_filter=self.glob_params["t_start_filter"],
            t_type_data=self.glob_params["t_type_data"],
        )

    # ------------------------------------------------------------------
    # Residuals and log-likelihood
    # ------------------------------------------------------------------

    def calc_residuals(self, mkn_vars: dict) -> dict:
        """Compute per-filter magnitude residuals between model and data.

        Calls :meth:`calc_lightcurve_vars` internally to update all
        photometric quantities before delegating to
        :func:`filters.calc_residuals`.

        Parameters
        ----------
        mkn_vars : dict
            Variable dictionary as returned by
            :meth:`~config.MKNConfig.get_vars`.

        Returns
        -------
        dict
            Keys are filter wavelengths; values are 1-D arrays of
            normalised residuals ``(model − data) / σ``.
        """
        self.calc_lightcurve_vars(mkn_vars)
        return flt.calc_residuals(
            self.calc_flux_factors(mkn_vars),
            self.time_observer(mkn_vars),
            self.lams,
            self.dic_filt,
            mkn_vars["glob"]["distance"] * Mpc2cm,
            self.redshift(mkn_vars["glob"]["distance"]),
            self.mag,
            self.glob_params["t_start_filter"],
            self.glob_params["t_type_data"],
            self.ejecta.radius_photo,
            T_photo=self.ejecta.T_photo,
            lum_shells=self.ejecta.lum_shells,
            T_shells=self.ejecta.T_shells,
            omegas=self.omegas,
            sigma_sys=mkn_vars["glob"]["sigma_sys"],
        )

    def calc_log_like(self, mkn_vars: dict) -> float:
        """Compute the Gaussian log-likelihood of the model given the data.

        Evaluates::

            ln L = −½ Σ_i r_i²  +  ln L_norm

        where ``r_i`` are the normalised residuals from
        :meth:`calc_residuals` and ``ln L_norm`` is the Gaussian
        normalisation from :meth:`calc_log_like_normalization`.

        Parameters
        ----------
        mkn_vars : dict
            Variable dictionary.

        Returns
        -------
        float
            Log-likelihood value.

        Notes
        -----
        # TODO: consider removing the normalisation term so that perfect
        agreement gives ``ln L = 0``.
        """
        residuals = self.calc_residuals(mkn_vars).values()
        total_sq  = np.sum(np.concatenate([r ** 2 for r in residuals]))
        return -0.5 * total_sq + self.calc_log_like_normalization(mkn_vars)

    def calc_log_like_normalization(self, mkn_vars: dict) -> float:
        """Compute the Gaussian log-likelihood normalisation constant.

        Returns::

            −½ · N_filters · Σ_λ ln(2π (σ_λ² + σ_sys²))

        Parameters
        ----------
        mkn_vars : dict
            Variable dictionary; must contain ``glob["sigma_sys"]``.

        Returns
        -------
        float
        """
        log_terms = np.concatenate([
            np.log(
                2.0 * np.pi * (
                    self.mag[lam]["sigma"] ** 2
                    + mkn_vars["glob"]["sigma_sys"] ** 2
                )
            )
            for lam in self.lams
        ])
        return -0.5 * len(self.lams) * np.sum(log_terms)

    # ------------------------------------------------------------------
    # Isotropised luminosity (consistency check)
    # ------------------------------------------------------------------

    def calc_lum_iso(
        self,
        mkn_vars: dict,
        t_scale=None,
        t_num=None,
        t_min=None,
        t_max=None,
    ) -> np.ndarray:
        """Compute the angle-averaged (isotropised) bolometric luminosity.

        Convenience wrapper around :func:`calc_lum_iso_fake_filters` for
        a quick sanity check of energy conservation.

        Parameters
        ----------
        mkn_vars : dict
            Variable dictionary.
        t_scale, t_num, t_min, t_max : optional
            Override the time-grid settings from ``glob_params``.

        Returns
        -------
        ndarray
            Isotropised bolometric luminosity as a function of time [erg/s].
        """
        return calc_lum_iso_fake_filters(
            self.shell_params,
            self.glob_params,
            mkn_vars,
            t_scale=t_scale,
            t_num=t_num,
            t_min=t_min,
            t_max=t_max,
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_magnitudes(
        self,
        mkn_vars: dict,
        ax=None,
        filename=None,
        title=None,
        titlesize: int = 30,
        hsize: int = 16,
        wsize: int = 9,
        labelsize: int = 30,
        ticksize: int = 26,
        legendsize: int = 12,
        legend_geom: list = None,
    ) -> None:
        """Plot model magnitudes and (optionally) observational data.

        Thin wrapper around :func:`~plotting.plot_magnitudes`.

        Parameters
        ----------
        mkn_vars : dict
            Variable dictionary.
        ax : matplotlib.axes.Axes or None, optional
            Existing axes to draw on; a new figure is created if ``None``.
        filename : str or None, optional
            Save the figure to this path instead of displaying it.
        title : str or None, optional
            Figure title.
        titlesize, hsize, wsize, labelsize, ticksize, legendsize : int
            Font and figure-size settings.
        legend_geom : list of 4 ints, optional
            ``[x, y, loc, ncol]`` for :meth:`matplotlib.axes.Axes.legend`.
        """
        if legend_geom is None:
            legend_geom = [0, 0, 3, 4]
        plot_magnitudes(
            self,
            mkn_vars,
            ax=ax,
            filename=filename,
            title=title,
            titlesize=titlesize,
            hsize=hsize,
            wsize=wsize,
            labelsize=labelsize,
            ticksize=ticksize,
            legendsize=legendsize,
            legend_geom=legend_geom,
        )


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------

def gen_inj_dict(
    shell_params: dict,
    glob_params: dict,
    mkn_vars: dict,
    seed=None,
    sigma_min=None,
    sigma_max=None,
) -> dict:
    """Build an injection-configuration dictionary for :class:`MKN`.

    Parameters
    ----------
    shell_params : dict
        Per-component shell parameters for the injected signal.
    glob_params : dict
        Global parameters for the injected signal.
    mkn_vars : dict
        Variable dictionary for the injected signal.
    seed : int or None, optional
        Random seed for noise generation.
    sigma_min, sigma_max : float or None, optional
        Bounds on the per-datapoint magnitude uncertainty.

    Returns
    -------
    dict
        Ready to pass as ``inj_dict`` to :class:`MKN`.
    """
    return {
        "shell_params": deepcopy(shell_params),
        "glob_params":  deepcopy(glob_params),
        "mkn_vars":     deepcopy(mkn_vars),
        "seed":         seed,
        "sigma_min":    sigma_min,
        "sigma_max":    sigma_max,
    }


def calc_lum_iso_fake_filters(
    shell_params: dict,
    glob_params: dict,
    mkn_vars: dict,
    t_scale=None,
    t_num=None,
    t_min=None,
    t_max=None,
) -> np.ndarray:
    """Compute isotropised luminosity using a dedicated ``"iso_calc"`` filter set.

    Parameters
    ----------
    shell_params : dict
        Per-component shell parameters.
    glob_params : dict
        Global parameters (a deep copy is modified internally).
    mkn_vars : dict
        Variable dictionary.
    t_scale, t_num, t_min, t_max : optional
        Override time-grid settings.

    Returns
    -------
    ndarray
        Isotropised bolometric luminosity [erg/s].
    """
    glob_params_lum_iso = deepcopy(glob_params)
    glob_params_lum_iso["filter_dictionary"] = "iso_calc"

    if t_scale is not None:
        glob_params_lum_iso["t_scale"] = t_scale
    if t_num is not None:
        glob_params_lum_iso["t_num"] = t_num
    if t_min is not None:
        glob_params_lum_iso["t_min"] = t_min
    if t_max is not None:
        glob_params_lum_iso["t_max"] = t_max

    mkn = MKN(
        shell_params, glob_params_lum_iso, log_name="LUM-ISO-MKN", log_level="WARNING"
    )
    if glob_params_lum_iso["t_scale"] not in ["lin", "log"]:
        mkn.logger.error(
            "lum_iso calculation has to be done with t_usage = lin or log! ... Exiting."
        )
        sys.exit()
    mkn.calc_lightcurve_vars(mkn_vars)
    return (
        mkn.time_observer(mkn_vars),
        mkn.time_source(mkn_vars),
        flt.calc_lum_iso_from_bol(
            mkn.ejecta.lum_bol, mkn.calc_flux_factors(mkn_vars), mkn.omegas
        ),
        flt.calc_lum_iso_from_mags(
            mkn.calc_magnitudes(mkn_vars, measures=False),
            mkn.dic_filt,
            mkn_vars["glob"]["distance"],
        ),
    )