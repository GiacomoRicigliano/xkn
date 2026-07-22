"""
incomplete_gamma.py
===================
Vectorized scaled upper incomplete gamma function and precomputed
tanh-sinh quadrature nodes/weights used by :mod:`diffusion_luminosity`.

The main public symbol is :func:`scaled_upper_gamma_vec`, a fully
numpy-native implementation that avoids the per-element Python dispatch
overhead of the former ``@njit + np.vectorize`` approach (~2× faster for
the 500×60 arrays used in :class:`~diffusion_luminosity.DiffusionLum`;
no numba dependency; no JIT warm-up delay on first import).
"""

from math import gamma as _gamma, pi as _pi, cos as _cos

import numpy as np
from mpmath import mp
from mpmath.calculus.quadrature import TanhSinh

# ---------------------------------------------------------------------------
# Tanh-sinh (double-exponential) quadrature nodes and weights
# ---------------------------------------------------------------------------
# Nodes and weights are computed once at module load time with 30 decimal
# digits of working precision and stored as ordinary float64 arrays. They
# are shared with :mod:`diffusion_luminosity` and consumed by the
# intermediate-regime branch of :func:`scaled_upper_gamma_vec`.

with mp.workdps(30):
    _m = 5
    h = float(mp.fadd(2 ** -mp.mpf(_m) * 2, 0, prec=53, rounding="n"))
    _tmp = TanhSinh(mp).calc_nodes(_m, 53)
    DE_nodes = np.array(
        [
            float(mp.fadd(mp.mpf(0.5) * (e + mp.mpf(1)), 0, prec=53, rounding="n"))
            for e in np.array(_tmp)[:, 0]
        ]
    )
    DE_weights = np.array(
        [float(mp.fadd(e, 0, prec=53, rounding="n")) for e in np.array(_tmp)[:, 1]]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scaled_upper_gamma_vec(s: float, z: np.ndarray) -> np.ndarray:
    """Vectorized scaled upper incomplete gamma function.

    Computes ``exp(z) * Re[Γ(s, −z)]`` for a scalar order *s* and an
    array of (negative) arguments *z*, switching between three
    numerically appropriate regimes:

    * **Near-origin** (``z > -0.97``): MacLaurin series expansion.
    * **Asymptotic** (``z < -50``): asymptotic series in ``1/z``.
    * **Intermediate** (``-50 ≤ z ≤ -0.97``): tanh-sinh quadrature
      using the precomputed nodes :data:`DE_nodes` / :data:`DE_weights`.

    Parameters
    ----------
    s : float
        Order of the incomplete gamma function.  Must satisfy
        ``0 < s < 1`` for the application in
        :class:`~diffusion_luminosity.DiffusionLum` (typically
        ``s = 1 - α/2`` with ``α ≈ 1.3``).
    z : array_like
        Argument array.  All elements must be **negative**.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as *z* containing
        ``exp(z) * Re[Γ(s, −z)]``.

    Notes
    -----
    This replaces the former ``@njit + np.vectorize`` implementation.
    Operating on whole masked sub-arrays rather than element-by-element
    eliminates the per-element Python→C dispatch overhead, giving roughly
    2× speed improvement for the 30 000-element arrays used inside
    :class:`~diffusion_luminosity.DiffusionLum`.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)

    S         = s - 1
    cos_pi_s  = _cos(_pi * s)
    cos_pi_S1 = _cos(_pi * (s + 1))
    g_s       = _gamma(s)
    g_s1      = _gamma(s + 1)

    m_near = z > -0.97          # MacLaurin series regime
    m_far  = z < -50.0          # asymptotic series regime
    m_mid  = ~m_near & ~m_far   # tanh-sinh quadrature regime

    # -- near-origin: MacLaurin series ----------------------------------
    if m_near.any():
        zn = z[m_near]
        r  = np.zeros(zn.shape)
        G  = g_s * s
        Z  = np.ones(zn.shape)
        for k in range(20):
            r += Z / G
            Z *= zn
            G *= s + k + 1
        out[m_near] = g_s * (np.exp(zn) - cos_pi_s * (-zn) ** s * r)

    # -- asymptotic: series in 1/z --------------------------------------
    if m_far.any():
        zf = z[m_far]
        r  = np.zeros(zf.shape)
        u  = 1.0
        iz = 1.0 / zf
        Z  = np.ones(zf.shape)
        for k in range(20):
            r += u * Z
            Z *= iz
            u *= S - k
        out[m_far] = r * _cos(_pi * S) * (-zf) ** S

    # -- intermediate: tanh-sinh quadrature ----------------------------
    if m_mid.any():
        zm     = z[m_mid]
        nodes  = DE_nodes[:, None]    # (n_nodes, 1)  for broadcast
        wts    = DE_weights[:, None]
        integ  = (
            np.sum(nodes ** s * np.exp(-zm[None, :] * nodes) * wts, axis=0)
            * h * 0.5
        )
        r = g_s1 - cos_pi_S1 * (-zm) ** (s + 1) * integ
        out[m_mid] = (r * np.exp(zm) - cos_pi_s * (-zm) ** s) / s

    return out


# ---------------------------------------------------------------------------
# Stand-alone test / benchmark (run as ``python incomplete_gamma.py <s>``)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import timeit
    import matplotlib.pyplot as plt

    mp.dps = 100

    s = float(sys.argv[1])

    # Vectorize a high-precision mpmath reference for comparison
    def _mp_ref(s, z):
        import mpmath as _mp
        return float(_mp.exp(z) * _mp.gammainc(s, -z, regularized=False))

    ref = np.vectorize(lambda z: _mp_ref(s, z))

    x = np.linspace(-10, -1e-4, 500)
    y_vec = scaled_upper_gamma_vec(s, x)
    y_ref = ref(x)

    max_err = np.max(np.abs(y_vec - y_ref) / (np.abs(y_ref) + 1e-300))
    print(f"s={s}  max relative error vs mpmath: {max_err:.2e}")

    t = timeit.timeit(lambda: scaled_upper_gamma_vec(s, x), number=500) / 500
    print(f"mean time per call ({len(x)} pts): {t*1e6:.1f} µs")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(x, y_vec, label="numpy_vec")
    ax1.plot(x, y_ref, "--", label="mpmath ref")
    ax1.set_ylabel("scaled_upper_gamma")
    ax1.legend()
    ax2.semilogy(x, np.abs(y_vec - y_ref) / (np.abs(y_ref) + 1e-300))
    ax2.set_ylabel("relative error")
    ax2.set_xlabel("z")
    plt.tight_layout()
    plt.show()
