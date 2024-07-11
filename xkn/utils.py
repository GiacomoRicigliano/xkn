import os
import sys
from copy import deepcopy

import astropy.units as apu
import numpy as np
from astropy.cosmology import Planck18, z_at_value
from scipy.interpolate import InterpolatedUnivariateSpline

from . import __path__ as mkn_path

mkn_path = mkn_path[0]

# ---units---
fourpi = 12.5663706144
oneoverpi = 0.31830988618
c = 2.99792458e10  # [cm/s]
c2 = 8.98755178737e20  # [cm^2/s^2]
Msun = 1.98855e33  # [g]
sec2day = 1.157407407e-5  # [day/s]
day2sec = 86400.0  # [sec/day]
sigma_SB = 5.6704e-5  # [erg/cm^2/s/K^4]
fourpisigma_SB = 7.125634793e-4  # [erg/cm^2/s/K^4]
h = 6.6260755e-27  # [erg*s]
kB = 1.380658e-16  # [erg/K]
pc2cm = 3.085678e18  # [cm/pc]
Mpc2cm = 1e6 * pc2cm  # [cm/pc]
sec2hour = 2.777778e-4  # [hr/s]
day2hour = 24.0  # [hr/day]
small = 1.0e-10  # [-]
huge = 1.0e30  # [-]


class ObserverProjection(object):

    def __init__(self, slices_num, slices_dist):
        assert slices_num in [12, 18, 24, 30]
        self.slices_num = slices_num
        self.slices_dist = slices_dist
        self.data_path = os.path.join(
            mkn_path, "flux_factor_data", f"{self.slices_dist}_{self.slices_num}.dat"
        )
        self.flux_interpolant = self.read_flux_factors()

    def __call__(self, angle):
        return np.array([f(angle) for f in self.flux_interpolant])

    def read_flux_factors(self):
        flux_factors = np.loadtxt(self.data_path).T
        return [
            InterpolatedUnivariateSpline(flux_factors[0], f) for f in flux_factors[1:]
        ]


class ExpansionModelSingleSpherical(object):

    def __init__(self, expansion_model):
        if expansion_model == "GK":
            self.expansion_model_single_spherical = self.GK_expansion_model
        else:
            sys.exit("No other expansion model presently implemented\n")

    def __call__(
        self,
        omegas,
        mass_ej,
        vel_rms,
        opacity,
        vel_min,
        vel_num_pts,
        vel_scale,
        vel_law,
        **kwargs,
    ):
        return self.expansion_model_single_spherical(
            omegas,
            mass_ej,
            vel_rms,
            opacity,
            vel_min,
            vel_num_pts,
            vel_scale,
            vel_law,
            **kwargs,
        )

    def func_vel(self, x):
        x2 = x * x
        x3 = x2 * x  # 2.1875 = 35./16.
        x5 = x3 * x2  # 1.3125 = 105./80.
        x7 = x5 * x2  # 0.3125 = 35./112.
        return 0.3125 * x7 - 1.3125 * x5 + 2.1875 * x3 - 2.1875 * x

    def GK_expansion_model(
        self,
        omegas,
        mass_ej,
        vel_rms,
        opacity,
        vel_min,
        vel_num,
        vel_scale,
        vel_law,
        **kwargs,
    ):
        if vel_law == "poly":
            vel_max = vel_rms * 3.0
        elif vel_law == "uniform":
            vel_max = vel_rms * np.sqrt(3)
        else:
            sys.exit(
                f'The vel_law "{vel_law}" is not recognized. Use "poly" or "uniform".'
            )

        if vel_scale == "lin":
            vel = np.linspace(vel_min, vel_max, vel_num)
        elif vel_scale == "log":
            vel = np.geomspace(vel_min, vel_max, vel_num)
        else:
            sys.exit(
                f'The vel_scale "{vel_scale}" is not recognized. Use "lin" or "log".'
            )

        if vel_law == "poly":
            m_vel = mass_ej * Msun * (1.0 + self.func_vel(vel / vel_max))  # [g]
        elif vel_law == "uniform":
            m_vel = mass_ej * Msun * (1.0 - (vel / vel_max))  # [g]

        t_diff = np.sqrt(opacity * m_vel / (omegas * vel * c2))  # [s]
        t_fs = t_diff * np.sqrt(1.5 / vel)

        return vel.T, m_vel.T, t_diff.T, t_fs.T


def init_times(
    t_scale,
    t_min=None,
    t_max=None,
    t_num=None,
    t_start_filter=None,
    mag=None,
    toll=0.05,
):
    assert None not in [t_min, t_max, t_num] or None not in [t_start_filter, mag]
    if t_scale == "lin":
        return np.linspace(t_min, t_max, num=t_num)
    elif t_scale == "log":
        return np.logspace(np.log10(t_min), np.log10(t_max), num=t_num)
    elif t_scale == "measures":
        return time_measures(mag, t_start_filter, toll)
    elif t_scale == "all_measures":
        return time_measures(mag, t_start_filter, None)
    else:
        sys.exit("Error! Wrong option for the time scale")


def time_measures(mag, t_start_filter, toll):
    all_time = (
        np.unique(
            np.sort(np.array([el for lam in mag for el in list(mag[lam]["time"])]))
        )
        - t_start_filter
    )
    if toll is None:
        return all_time * day2sec

    times = []
    ii = 0
    for i in range(1, len(all_time)):
        if all_time[i] < (1 + toll) * all_time[ii]:
            continue
        times.append(0.5 * (all_time[i] + all_time[ii]))
        ii = i
    times = [all_time[0]] + times + [all_time[-1]]
    return np.array(times) * day2sec


def time_safe(times, t_0):
    return times[times > t_0]


def smoothclamp_aux(t):
    return np.where(t < 0, 0, np.where(t <= 1, 3.0 * t**2 - 2.0 * t**3, 1))


def smoothclamp(x, x1, x2, y1, y2):
    return y1 + (y2 - y1) * smoothclamp_aux(np.log10(x / x1) / np.log10(x2 / x1))


def T_eff_calc(omegas, radius_photo, lum_bol):
    return np.where(
        radius_photo > 0,
        (lum_bol / omegas[:, None] / radius_photo**2 / sigma_SB) ** 0.25,
        0,
    )


def calc_Tfloor(mode, value, T_floor_LA, T_floor_Ni):
    if mode == "opacity":
        weight = smoothclamp(
            value, 1.0, 10.0, 1.0, 0.0
        )  # calculate from value of opacity
    elif mode == "ye":
        weight = smoothclamp(value, 0.3, 0.2, 1.0, 0.0)  # calculate from value of ye
    else:
        sys.exit("Please choose 'opacity' or 'ye' for calculating T_floor")
    return T_floor_Ni * weight + T_floor_LA * (1.0 - weight)


def R_early(times, vel_max, opacity, rho0, t0):  # early time approximation
    return (
        (1 - ((times / t0) ** 3 / (3 * opacity * rho0 * times * vel_max)) ** (0.25))
        * vel_max
        * times
    )


def parab(params, t):
    a, b, c = params
    return a * t**2 + b * t + c


def error(params, t, r):
    return parab(params, t) - r


def Mv_Woll(mass, ratio):
    return mass * (
        1 + 0.3125 * ratio**7 - 1.3125 * ratio**5 + 2.1875 * ratio**3 - 2.1875 * ratio
    )


def mass_scaled_thin(radius_photo, times, vel_woll, mass_scaled):
    return np.where(
        radius_photo / times <= vel_woll,
        Mv_Woll(mass_scaled, radius_photo / times / vel_woll),
        0,
    ).sum(axis=0)


def R_poly(radius, lum, times, vel_woll, mass_scaled, T_floor):
    return (
        radius**2
        - (
            1
            - mass_scaled_thin(radius[None], times[None], vel_woll, mass_scaled)
            / mass_scaled.sum(axis=0)
        )
        * lum
        / fourpisigma_SB
        / T_floor**4
    )


def find_floor_radius_roots(
    radius_photo, lum_bol, omegas, times, vel_woll, mass_scaled, T_floors, mask_floors
):
    alpha = lum_bol / omegas / sigma_SB / T_floors**4
    beta = alpha * mass_scaled / mass_scaled.sum(axis=0)
    gamma = 1 / times / vel_woll
    zeros = np.zeros_like(beta)
    ones = np.ones_like(beta)

    R_poly_2 = np.array(
        [zeros - alpha[None] / vel_woll.shape[0], zeros, ones / vel_woll.shape[0]]
    )
    R_poly_7 = np.array(
        [
            beta - alpha / vel_woll.shape[0],
            -2.1875 * beta * gamma,
            ones / vel_woll.shape[0],
            2.1875 * beta * gamma**3,
            zeros,
            -1.3125 * beta * gamma**5,
            zeros,
            0.3125 * beta * gamma**7,
        ]
    )

    for i, j in zip(*np.nonzero(mask_floors)):
        poly = np.sum(
            [
                (
                    R_poly_7[:, k, i, j]
                    if radius_photo[i, j] / times[0, j] <= vel_woll[k, i, 0]
                    else np.pad(R_poly_2[:, k, i, j], (0, 5))
                )
                for k in range(vel_woll.shape[0])
            ],
            axis=0,
        )
        radius_photo[i, j] = max_real_poly_root(poly)

    return radius_photo


def max_real_poly_root(poly):
    return max(
        0,
        np.amax(
            np.array(
                [
                    np.real(root)
                    for root in np.polynomial.polynomial.polyroots(poly)
                    if not np.imag(root) and np.real(root) > 0
                ]
            )
        ),
    )


class Redshift(object):

    def __init__(self, cosmo):
        self.cosmo = cosmo
        if self.cosmo is None:
            self.get_redshift = self.set_0
        elif self.cosmo == "Planck18":
            self.get_redshift = self.get_z
        else:
            sys.exit(
                'Redshift.__init__: Error in cosmology parameter. Choose None or "Planck18.'
            )

    def __call__(self, distance, z_min=0.0, z_max=2.0):
        return self.get_redshift(distance, z_min=z_min, z_max=z_max)

    def set_0(self, distance, z_min=0.0, z_max=2.0):
        return 0

    def get_z(self, distance, z_min=0.0, z_max=2.0):
        return float(
            z_at_value(Planck18.luminosity_distance, distance * apu.Mpc, z_min, z_max)
        )


#####


def check_dict_variables(
    dic=(), var=(), logger=None, strict=False, label="", allow_none=False
):
    # dic = (dict, key_list) --> check, if key in dict and if dict[key] is None
    if dic:
        missing_keys = [key for key in dic[1] if key not in dic[0].keys()]
        none_keys = [
            key for key in dic[1] if key in dic[0].keys() and dic[0][key] is None
        ]
        if allow_none:
            none_keys = []
        if missing_keys or none_keys:
            dic_bool = False
        else:
            dic_bool = True
    else:
        missing_keys = []
        none_keys = []
        dic_bool = True

    if var:
        none_vars = [var[1][i] for i in range(len(var[0])) if var[0][i] is None]
        if none_vars:
            var_bool = False
        else:
            var_bool = True
    else:
        none_vars = []
        var_bool = True

    if not dic_bool or not var_bool:
        if label:
            label += ": "
        msg = f"{label}Missing keys in dictionary: {missing_keys}, None-valued keys in dictionary: {none_keys}, or None-valued variables: {none_vars}!"
        if logger is not None:
            if strict:
                logger.error(msg)
                sys.exit()
            else:
                logger.warning(msg)

    return dic_bool and var_bool
