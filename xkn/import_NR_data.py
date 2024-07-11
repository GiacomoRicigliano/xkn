import numpy as np
from scipy.interpolate import interp1d


def importNRprofiles(filename, angles):

    th, mass, ye, vel = np.loadtxt(filename, unpack=True, usecols=(0, 1, 2, 3))

    mask = mass < 1.0e-9
    mass[mask] = 1.0e-9
    ye[mask] = 0.1
    vel[mask] = 0.1

    # find the central angles and bin sizes
    th_central = [0.5 * (a[1] + a[0]) for a in angles]
    dth = th[0]
    if dth == 0:
        sys.exit("NR data unsupported format: input central values for angular bins")
    th_size = [np.cos(a - dth) - np.cos(a + dth) for a in th]
    ang_size = [np.cos(a[0]) - np.cos(a[1]) for a in angles]
    # smooth the mass
    mass_smooth = smooth_array(mass)
    mass_smooth = np.sum(mass) / np.sum(mass_smooth) * mass_smooth

    # smooth the velocity
    vel_smooth = smooth_array(mass * vel)
    vel_smooth = (np.sum(vel * mass) / np.sum(vel_smooth) * vel_smooth) / mass_smooth

    # smooth the ye
    ye_smooth = smooth_array(ye)
    ye_smooth = np.sum(ye * mass) / np.sum(ye_smooth * mass) * ye_smooth

    # generating smooth, interpolated profiles for mass, vel, ye
    func_m = interp1d(th, mass_smooth / th_size)
    func_vel = interp1d(th, vel_smooth)
    func_ye = interp1d(th, ye_smooth)

    profile_m = func_m(th_central) * ang_size
    profile_vel = func_vel(th_central)
    profile_ye = func_ye(th_central)

    profile_m *= np.sum(mass_smooth) / np.sum(2.0 * profile_m)

    return profile_m, profile_vel, profile_ye


def smooth_array(arr):
    larr = np.log10(arr)
    arr_smooth = np.zeros_like(larr)
    arr_smooth[0] = larr[0]
    arr_smooth[1] = (larr[0] + larr[1] + larr[2]) / 3.0
    arr_smooth[2] = (larr[0] + larr[1] + larr[2] + larr[3] + larr[4]) / 5.0
    arr_smooth[3:-3] = (
        larr[:-6]
        + larr[1:-5]
        + larr[2:-4]
        + larr[3:-3]
        + larr[4:-2]
        + larr[5:-1]
        + larr[6:]
    ) / 7.0
    arr_smooth[-3] = (larr[-5] + larr[-4] + larr[-3] + larr[-2] + larr[-1]) / 5.0
    arr_smooth[-2] = (larr[-3] + larr[-2] + larr[-1]) / 3.0
    arr_smooth[-1] = larr[-1]
    return 10.0**arr_smooth
