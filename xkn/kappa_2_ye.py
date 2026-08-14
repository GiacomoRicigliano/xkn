import numpy as np
import sys
from scipy.interpolate import RegularGridInterpolator, interp1d

def func_wu_opacities(ye, kappa_min=1, kappa_up=9):
    kappa = kappa_min + kappa_up / (1 + (4 * ye)** 12)
    return kappa

def func_wu_opacities_invert(kappa, kappa_min=1, kappa_up=9):
    ye = ((kappa_min + kappa_up - kappa) / (kappa - kappa_min))**(1/12)/4
    return ye

class Kappa2Ye(object):

    def __init__(self, ye_k_dep):
        if ye_k_dep == "TH":
            self.f_ye2k = func_interp_tanaka()
            self.f_k2ye = func_interp_tanaka_invert()
        elif ye_k_dep == "Wu":
            self.f_ye2k = func_wu_opacities
            self.f_k2ye = func_wu_opacities_invert
        else:
            sys.exit(
                "Wrong input name for ye-k dependence\n"
                + "Please use:\n"
                + '"TH" for Tanaka & Hotokezaka 2020\n'
            )

    def __call__(self, mode, value):
        if mode == "opacity":
            return self.f_k2ye(value)  # calculate from value of opacity
        elif mode == "ye":
            return self.f_ye2k(value)  # calculate from value of ye
        else:
            sys.exit("Please choose 'opacity' or 'ye' to convert from one to the other")


def func_interp_tanaka():
    ye_tan = np.asarray([0.01, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
    k_tan = np.asarray([30.1, 30.0, 29.9, 22.30, 5.60, 5.36, 3.30, 0.96, 0.1])
    return interp1d(ye_tan, k_tan, kind="linear", fill_value="extrapolate")


def func_interp_tanaka_invert():
    ye_tan = np.asarray([0.01, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])[::-1]
    k_tan = np.asarray([30.1, 30.0, 29.9, 22.30, 5.60, 5.36, 3.30, 0.96, 0.1])[::-1]
    return interp1d(k_tan, ye_tan, kind="linear", fill_value="extrapolate")
