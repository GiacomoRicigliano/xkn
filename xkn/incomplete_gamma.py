from math import gamma
from mpmath import mp
from numba import njit
import matplotlib.pyplot as plt
import mpmath
import numpy as np


# Compute nodes and weights (without h) for the double exponential (or
# tanh-sinh) rule with mpmath, coonvert them to Python floats, and scale the
# nodes for the interval (0, 1).
with mp.workdps(30):
    m = 5
    h = float(mp.fadd(2 ** -mp.mpf(m) * 2, 0, prec=53, rounding="n"))
    tmp = mpmath.calculus.quadrature.TanhSinh(mp).calc_nodes(m, 53)
    DE_nodes = np.array(
        [
            float(mp.fadd(mp.mpf(0.5) * (e + mp.mpf(1)), 0, prec=53, rounding="n"))
            for e in np.array(tmp)[:, 0]
        ]
    )
    DE_weights = np.array(
        [float(mp.fadd(e, 0, prec=53, rounding="n")) for e in np.array(tmp)[:, 1]]
    )


# Note this is HEAVILY optimized for handling s in (0, 1) and z in (-inf, 0).
# For any other arguments, it will give wrong results.
@njit(fastmath=True)
def scaled_upper_gamma(s, z):

    if s == 0.5:
        return np.pi**0.5 * np.exp(z)

    # Close to the origin, use a sort of MacLaurin series
    if z > -0.97:

        r = 0
        g = gamma(s)
        G = g * s
        Z = 1.0
        for k in range(0, 20):
            r += Z / G
            Z *= z
            G *= s + k + 1
        r = g * (np.exp(z) - np.cos(np.pi * s) * (-z) ** s * r)

    # For large arguments, use an asymptotic series
    elif z < -50:

        r = 0.0
        S = s - 1
        u = 1.0
        iz = 1.0 / z
        Z = 1.0
        for k in range(0, 20):
            r += u * Z
            Z *= iz
            u *= S - k
        r *= np.cos(np.pi * S) * (-z) ** S

    # In between, need to numerically integrate one of its integral
    # representation. Moreover we use the recursive formula of the incomplete
    # gamma function to get rid of the (integrable) singularity at the left
    # endpoint of the integration interval.
    else:

        # This is what we call the "gamma star formulation"
        S = s + 1.0
        r = (
            gamma(S)
            - np.cos(np.pi * S)
            * (-z) ** S
            * np.sum(DE_nodes**s * np.exp(-z * DE_nodes) * DE_weights)
            * h
            * 0.5
        )
        r = (r * np.exp(z) - np.cos(np.pi * s) * (-z) ** s) / s

    return r


if __name__ == "__main__":

    import sys
    import timeit

    mp.dps = 100

    s = float(sys.argv[1])

    scaled_upper_gamma = np.vectorize(scaled_upper_gamma)

    @np.vectorize
    def scaled_upper_gamma_mpmath(s, z):
        if s == 0.5:
            r = mp.sqrt(mp.pi) * mp.exp(z)
        else:
            s, z = mp.mpf(s), mp.mpf(z)
            r = mp.re(mp.gammainc(s, z) * mp.exp(z))
        return float(mp.fadd(r, 0, prec=53, rounding="n"))

    def execution_time(fname, ts):
        t = (
            timeit.timeit(
                stmt=f"{fname:s}(s, ts)", number=10, globals=(locals() | globals())
            )
            / 10
            / len(ts)
        )
        return t

    ts = np.geomspace(1e-8, 0.97, 1000)
    ts = np.append(ts, np.linspace(0.97 + 1e-8, 50, 1000))
    ts = np.append(ts, np.geomspace(50 + 1e-8, 1e11, 1000))
    ts = -ts[::-1]

    exact = scaled_upper_gamma_mpmath(s, ts)
    approximate = scaled_upper_gamma(s, ts)
    abs_error = np.abs(exact - approximate)
    rel_error = np.abs(abs_error / exact)

    exact_time = execution_time("scaled_upper_gamma_mpmath", ts)
    approximate_time = execution_time("scaled_upper_gamma", ts)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.loglog(-ts, np.abs(exact), label="'Exact' (mpmath)")
    ax1.loglog(-ts, np.abs(approximate), label="Custom")
    ax2.loglog(-ts, abs_error, label="Absolute error")
    ax2.loglog(-ts, rel_error, label="Relative error")
    ax1.set_xlabel("-t")
    ax2.set_xlabel("-t")
    ax2.axvline(
        0.97,
        color="r",
        linestyle="--",
        linewidth=1,
        label="Boundary between MacLaurin series and numerical integral",
    )
    ax2.axvline(
        50,
        color="g",
        linestyle="--",
        linewidth=1,
        label="Boundary between numerical integral and asymptotic series",
    )
    ax1.set_ylim([np.abs(exact).min() * 0.9, np.abs(exact).max() * 1.1])
    ax2.set_ylim([1e-17, 1e-9])
    ax1.legend(loc="best")
    ax2.legend(loc="best")
    fig.suptitle(
        f""" Scaled Upper Gamma function
s=1 - alpha/2={s:f}
Execution time for mpmath (average over all ts, 1 call with numpy broadcasting) = {exact_time:.2e} seconds
Execution time for custom (average over all ts, 1 call with numpy broadcasting) = {approximate_time:.2e} seconds
Speedup = {exact_time/approximate_time:.1f}x
        """
    )
    plt.show()
