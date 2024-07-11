import os
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp2d


class HeatingFunction(object):

    def __init__(self, heat_model, s, tau):
        if heat_model == "PBR":
            self.heat_func = None
        elif heat_model == "LR":
            self.heat_func = interpolating_function(s, tau)
        elif heat_model == "RP":
            self.heat_func = None
        elif heat_model == "K":
            self.heat_func = None
        else:
            sys.exit(
                "Wrong input name for heating rate model\n"
                + "Please use:\n"
                + '"RP" for Perego et al 2021\n'
                + '"PBR" for Perego et al 2017 ApJL\n'
                + '"LR" for Lippuner & Roberts 2016 ApJ\n'
                + '"K" for Korobkin 2015'
            )

    def __call__(self, ye, time, **kwargs):
        if ye.shape:
            return self.heat_func(ye, time, **kwargs)[:, np.argsort(np.argsort(ye))].T
        else:
            return self.heat_func(ye, time, **kwargs)[:, 0]


# specifico il nome del file di input di Lippuner+ 2015
def interpolating_function(s, tau):
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "interp_tables",
        "hires_sym0_results",
    )
    _, s_ar, tau_ar, A, alpha, B1, beta1, B2, beta2, B3, beta3 = np.loadtxt(
        filename, unpack=True, usecols=(0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12)
    )

    ye = np.asarray(
        [
            0.01,
            0.04,
            0.07,
            0.10,
            0.13,
            0.16,
            0.19,
            0.22,
            0.25,
            0.29,
            0.32,
            0.35,
            0.38,
            0.41,
            0.44,
            0.47,
            0.50,
        ]
    )
    t_array = np.logspace(-2.0, 2.0, num=50, endpoint=True)

    # creo una matrice direttamente dell'heating rate
    idx = np.nonzero(
        np.logical_and(
            s_ar == find_nearest(s_ar, s), tau_ar == find_nearest(tau_ar, tau)
        )
    )[0]
    Q = [
        A[idx] * t ** (-alpha[idx])
        + B1[idx] * np.exp(-t / beta1[idx])
        + B2[idx] * np.exp(-t / beta2[idx])
        + B3[idx] * np.exp(-t / beta3[idx])
        for t in t_array
    ]
    Q = np.log10(np.array(Q))

    return interp2d(ye, t_array, Q, kind="linear")


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
