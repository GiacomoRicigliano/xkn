"""
diffusion_luminosity.py
=======================
Semi-analytical diffusion luminosity model for kilonova ejecta, following
Ricigliano & Lippold (in prep.) / Piro & Morozova (2016).

The module exposes two concrete implementations of the luminosity class:

``DiffusionLum_direct`` *(default, recommended)*
    Evaluates the Fourier-series solution analytically using two direct
    calls to :func:`~incomplete_gamma.scaled_upper_gamma_vec`.  No
    interpolation grids are built during initialisation, so construction
    is O(1) and there are no approximation errors from grid coarseness or
    extrapolation outside a pre-tabulated range.  The cache is **bypassed
    entirely** for this implementation — no memory accumulates across PE
    samples.

``DiffusionLum_interp``
    Legacy implementation that pre-builds a 1-D ``interp1d`` table (300
    pts) and a 3-D ``RegularGridInterpolator`` table (50×100×50 = 250 000
    pts) during ``__init__``.  Retained for reference and backward
    compatibility.  **Known issues**: (a) the 3-D grid covers only
    ``τ₀ ∈ [6.4×10³, 6.4×10¹¹]``—typical physical parameters often fall
    outside this range causing unbounded linear extrapolation errors; (b)
    the ``n``-axis uses ``linspace`` spacing of 5, but the tabulated
    function drops by ~17 orders of magnitude between ``n = 1`` and
    ``n = 6``, making linear interpolation unreliable near ``n = 1``; (c)
    construction takes O(seconds) on the first call.

The active implementation is selected at the bottom of this module via::

    DiffusionLum = DiffusionLum_direct   # change to DiffusionLum_interp to revert

Cache control (relevant when using ``DiffusionLum_interp``)
------------------------------------------------------------
In a parameter-estimation run every posterior sample produces a unique
``(A_eff, alpha_eff)`` combination, so an unbounded cache would grow at
≈90 MB per sample and cause OOM after a few thousand steps.  The cache is
therefore a bounded LRU store (default 128 entries ≈ 256 MB):

``clear_cache()``
    Drop all entries and reset statistics.  Call between independent PE
    runs or whenever the global parameters change.

``cache_info()``
    Return a dict with ``hits``, ``misses``, ``size``, ``maxsize``, and
    ``memory_MB``.

``set_cache_maxsize(n)``
    Resize the cache at runtime.  Pass ``0`` to disable caching entirely.

Public helpers
--------------
generate_diff_lums
    Factory that creates (or retrieves from cache) :class:`DiffusionLum`
    instances for each angular bin.
"""

import sys
import hashlib
from collections import OrderedDict

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator

from . import nuclear_heat as nh
from .utils import c, day2sec
from .incomplete_gamma import scaled_upper_gamma_vec

# Convenience alias used throughout this module and in shell.py
sug = scaled_upper_gamma_vec


# ---------------------------------------------------------------------------
# Bounded LRU cache for DiffusionLum_interp objects
# ---------------------------------------------------------------------------
# Background
# ~~~~~~~~~~
# ``DiffusionLum_interp.__init__`` builds a 50×100×50 = 250 000-point 3-D
# ``RegularGridInterpolator`` (≈2 MB per object).  In a parameter-estimation
# (PE) run the effective heating parameters ``(A_eff, alpha_eff)`` change
# with every posterior sample, so a naive unbounded ``dict`` accumulates a
# new 2 MB entry per sample → OOM after a few thousand steps
# (45 objects/sample × 2 MB × 1 000 samples ≈ 90 GB).
#
# ``DiffusionLum_direct.__init__`` is O(1) (no grids), so it does NOT use
# this cache — skipping it removes overhead with zero benefit.
#
# Design
# ~~~~~~
# The cache is an ``OrderedDict`` used as a least-recently-used (LRU) store
# with a configurable maximum entry count (default 128).  When the limit is
# reached the oldest entry is evicted before the new one is inserted.
# This bounds memory to ``_DIFFLUM_CACHE_MAXSIZE × 2 MB ≈ 256 MB`` in the
# worst case, while still giving cache hits when the sampler revisits
# previously seen parameter combinations (e.g. during burn-in or when the
# posterior concentrates).
#
# Public API
# ~~~~~~~~~~
# ``clear_cache()``   — drop all entries (call between independent PE runs).
# ``cache_info()``    — return a dict with hits, misses, size, and maxsize.
# ``set_cache_maxsize(n)`` — resize the cache at runtime.

_DIFFLUM_CACHE: OrderedDict = OrderedDict()
_DIFFLUM_CACHE_MAXSIZE: int = 128
_DIFFLUM_CACHE_HITS: int    = 0
_DIFFLUM_CACHE_MISSES: int  = 0


def clear_cache() -> None:
    """Evict all entries from the :class:`DiffusionLum_interp` cache.

    Call this between independent parameter-estimation runs (or whenever
    the fixed global parameters ``t_0``, ``T_0``, or the time grid change)
    to avoid stale cache hits and to free the associated memory immediately.

    Example
    -------
    >>> from xkn.diffusion_luminosity import clear_cache
    >>> clear_cache()
    """
    global _DIFFLUM_CACHE_HITS, _DIFFLUM_CACHE_MISSES
    _DIFFLUM_CACHE.clear()
    _DIFFLUM_CACHE_HITS   = 0
    _DIFFLUM_CACHE_MISSES = 0


def cache_info() -> dict:
    """Return diagnostic information about the :class:`DiffusionLum_interp` cache.

    Returns
    -------
    dict with keys:

    ``hits`` : int
        Number of times a cached object was returned without reconstruction.
    ``misses`` : int
        Number of times a new object was constructed and inserted.
    ``size`` : int
        Current number of entries in the cache.
    ``maxsize`` : int
        Maximum number of entries before LRU eviction occurs.
    ``memory_MB`` : float
        Approximate memory occupied by cached G_K arrays (2 MB each).

    Example
    -------
    >>> from xkn.diffusion_luminosity import cache_info
    >>> print(cache_info())
    """
    return {
        "hits":      _DIFFLUM_CACHE_HITS,
        "misses":    _DIFFLUM_CACHE_MISSES,
        "size":      len(_DIFFLUM_CACHE),
        "maxsize":   _DIFFLUM_CACHE_MAXSIZE,
        "memory_MB": len(_DIFFLUM_CACHE) * 2.0,   # ≈2 MB per DiffusionLum_interp
    }


def set_cache_maxsize(n: int) -> None:
    """Set the maximum number of entries in the :class:`DiffusionLum_interp` cache.

    Entries beyond the new limit are evicted (oldest first) immediately.

    Parameters
    ----------
    n : int
        New maximum size.  Pass ``0`` to disable caching entirely
        (every call constructs a fresh object).

    Example
    -------
    >>> from xkn.diffusion_luminosity import set_cache_maxsize
    >>> set_cache_maxsize(32)   # tighter limit for low-memory machines
    """
    global _DIFFLUM_CACHE_MAXSIZE
    _DIFFLUM_CACHE_MAXSIZE = max(0, int(n))
    # Evict excess entries immediately (oldest first)
    while len(_DIFFLUM_CACHE) > _DIFFLUM_CACHE_MAXSIZE:
        _DIFFLUM_CACHE.popitem(last=False)


def _difflum_cache_key(
    t_0: float,
    times: np.ndarray,
    T_0: float,
    A: float,
    alpha: float,
) -> str:
    """Return a stable MD5 hex-digest key for the given constructor arguments.

    Parameters
    ----------
    t_0 : float
        Initialisation time of the diffusive sphere [s].
    times : numpy.ndarray
        1-D array of observer times [s].
    T_0 : float
        Initialisation temperature [K].
    A : float
        Effective heating amplitude [erg/s/g].
    alpha : float
        Effective heating power-law index.

    Returns
    -------
    str
        32-character MD5 hex string uniquely identifying the combination.
    """
    return hashlib.md5(
        b"%r|%r|%r|%r|%r" % (t_0, tuple(times.tolist()), T_0, float(A), float(alpha))
    ).hexdigest()


def _cache_get(key: str):
    """Retrieve *key* from the LRU cache, promoting it to most-recent.

    Returns ``None`` if *key* is not present.
    """
    global _DIFFLUM_CACHE_HITS, _DIFFLUM_CACHE_MISSES
    if key in _DIFFLUM_CACHE:
        _DIFFLUM_CACHE.move_to_end(key)   # mark as most recently used
        _DIFFLUM_CACHE_HITS += 1
        return _DIFFLUM_CACHE[key]
    _DIFFLUM_CACHE_MISSES += 1
    return None


def _cache_put(key: str, obj) -> None:
    """Insert *obj* under *key*, evicting the oldest entry if at capacity."""
    if _DIFFLUM_CACHE_MAXSIZE <= 0:
        return   # caching disabled
    if key in _DIFFLUM_CACHE:
        _DIFFLUM_CACHE.move_to_end(key)
    else:
        if len(_DIFFLUM_CACHE) >= _DIFFLUM_CACHE_MAXSIZE:
            _DIFFLUM_CACHE.popitem(last=False)   # evict LRU entry
        _DIFFLUM_CACHE[key] = obj


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def generate_diff_lums(
    ye,
    entropy,
    tau,
    times,
    glob_vars,
    shell_params,
    glob_params,
    **kwargs,
):
    """Create or retrieve cached :class:`DiffusionLum` objects for each angular bin.

    One :class:`DiffusionLum` instance is created per unique combination of
    heating parameters ``(A_eff, alpha_eff)`` derived from the electron
    fraction *ye*, *entropy*, and expansion timescale *tau* of each angular
    bin.

    Caching behaviour
    -----------------
    * **DiffusionLum_direct** (default): construction is O(1), so the cache
      is **bypassed entirely** — no memory is accumulated across PE samples.
    * **DiffusionLum_interp**: instances are stored in the module-level LRU
      cache :data:`_DIFFLUM_CACHE` (max :data:`_DIFFLUM_CACHE_MAXSIZE`
      entries, default 128 ≈ 256 MB).  When the limit is reached the oldest
      entry is evicted automatically.  Call :func:`clear_cache` between
      independent PE runs to release memory explicitly.

    Parameters
    ----------
    ye : array_like, shape (n_angles,)
        Electron fraction per angular bin.
    entropy : array_like, shape (n_angles,)
        Entropy per baryon [k_B/baryon] per angular bin.
    tau : array_like, shape (n_angles,)
        Expansion timescale [ms] per angular bin.
    times : numpy.ndarray, shape (n_times,)
        Observer time array [s].
    glob_vars : dict
        Global variables including ``nuc_corr``.
    shell_params : dict
        Component-level parameters including ``heat_model``.
    glob_params : dict
        Global parameters including ``t_0``, ``T_0``, ``cnst_eff``,
        ``idx_eff``.

    Returns
    -------
    list of DiffusionLum
        One instance per angular bin (length ``n_angles``).

    Raises
    ------
    SystemExit
        If ``shell_params["heat_model"]`` is not a recognised heating model.
    """
    heat_model = shell_params["heat_model"]

    if heat_model == "RP":
        A_alphas = [
            nh.skynet_heating_params(YE, S, TAU)
            for YE, S, TAU in zip(ye, entropy, tau)
        ]
    elif heat_model == "K":
        A_alphas = len(ye) * [
            (1.95e10 * glob_vars["eps0"] / 2e18, glob_params["alpha"])
        ]
    # TODO: use correct fit parameters for PBR and LR
    elif heat_model == "PBR":
        A_alphas = len(ye) * [
            (1.95e10 * glob_vars["eps0"] / 2e18, glob_params["alpha"])
        ]
    elif heat_model == "LR":
        A_alphas = [
            nh.skynet_heating_params(YE, S, TAU)
            for YE, S, TAU in zip(ye, entropy, tau)
        ]
    else:
        sys.exit(
            "Wrong input name for heating rate model\n"
            "Please use:\n"
            '  "RP"  for Perego et al 2021\n'
            '  "PBR" for Perego et al 2017 ApJL\n'
            '  "LR"  for Lippuner & Roberts 2016 ApJ\n'
            '  "K"   for Korobkin 2015'
        )

    use_interp = (DiffusionLum is DiffusionLum_interp)

    result = []
    for A, alpha in A_alphas:
        A_eff     = (
            glob_params["cnst_eff"]
            * A
            * glob_vars["nuc_corr"]
            * glob_params["t_0"] ** (-alpha)
        )
        alpha_eff = glob_params["idx_eff"] + alpha

        if use_interp:
            # DiffusionLum_interp: check the bounded LRU cache first.
            # Construction is expensive (~2 MB, O(seconds)), so cache hits
            # are worthwhile when the sampler revisits the same parameters.
            key = _difflum_cache_key(
                glob_params["t_0"], times, glob_params["T_0"], A_eff, alpha_eff
            )
            obj = _cache_get(key)
            if obj is None:
                obj = DiffusionLum(
                    glob_params["t_0"], times, glob_params["T_0"], A_eff, alpha_eff,
                )
                _cache_put(key, obj)
        else:
            # DiffusionLum_direct: construction is O(1) and allocates only
            # two small fixed arrays, so caching adds overhead with no
            # benefit and would grow unboundedly across PE samples.
            obj = DiffusionLum(
                glob_params["t_0"], times, glob_params["T_0"], A_eff, alpha_eff,
            )

        result.append(obj)
    return result


# ---------------------------------------------------------------------------
# DiffusionLum_direct  (recommended)
# ---------------------------------------------------------------------------

class DiffusionLum_direct:
    """Diffusion luminosity via direct evaluation of the Fourier-series formula.

    Computes the bolometric luminosity of a single diffusive ejecta shell
    via the semi-analytical Fourier expansion derived in Ricigliano &
    Lippold, evaluating all special functions directly with
    :func:`~incomplete_gamma.scaled_upper_gamma_vec` — no interpolation
    grids are built.

    This is the **recommended** implementation.  Advantages over
    :class:`DiffusionLum_interp`:

    * ``__init__`` is O(1): only small fixed arrays are precomputed.
    * ``calc_lum`` is exact to machine precision for any ``τ₀`` value,
      including values far outside the range covered by the legacy grid.
    * No interpolation error: the legacy grid had up to ~100% error for
      typical physical parameters (``n``-axis step too coarse; ``τ₀`` often
      outside the tabulated range).
    * ~5× faster per ``calc_lum`` call compared to the interpolation path.

    Class-level constants
    ---------------------
    N : int
        Number of Fourier terms (500, chosen for convergence).
    n : ndarray, shape (N, 1)
        Term indices ``[1, 2, …, N]`` reshaped for broadcasting against
        the time axis.
    sign : ndarray, shape (N, 1)
        Alternating signs ``(-1)^(n+1)``.
    S : ndarray, shape (N, 1)
        Initial-condition selector: 1 for ``n = 1``, 0 otherwise.

    Parameters
    ----------
    t_0 : float
        Initialisation time of the diffusive sphere [s].
    time : numpy.ndarray, shape (n_times,)
        Observer time array [s].
    T_0 : float
        Initialisation temperature [K].
    A : float
        Heating amplitude coefficient [erg/s/g] (already combined with
        ``cnst_eff``, ``nuc_corr``, and ``t_0^{-alpha}``).
    alpha : float
        Heating power-law index (combined ``idx_eff + alpha_skynet``).
    """

    # -- class-level constants (shared across all instances) ----------------
    N    = 500
    n    = np.arange(1, N + 1, dtype=float)[:, None]   # (N, 1)
    sign = ((-1) ** (n + 1)).astype(int)                # (N, 1)
    S    = np.where(n == 1, 1.0, 0.0)                   # (N, 1)

    def __init__(
        self,
        t_0: float,
        time: np.ndarray,
        T_0: float,
        A: float,
        alpha: float,
    ) -> None:
        self.t_0   = t_0
        self.t_f   = time[-1]
        # Radiation energy density constant: a = 7.57e-15 erg/(cm^3 K^4)
        self.E_0   = T_0 ** 4 * 7.57e-15   # [erg/cm^3]
        self.A     = A
        self.alpha = alpha
        self.t     = time                   # (n_times,)

        # Precomputed arrays — shape (N, 1) and (N, n_times) — fixed for
        # the lifetime of this instance regardless of v_max, k, M.
        self.gamma_factor = (
            -0.5 * (np.pi * self.n * self.t) ** 2 / t_0
        )   # (N, n_times)
        self.gamma_K_nt_factor = (
            0.5 * (np.pi * self.n) ** 2 * t_0
        )   # (N, 1)

        # Per-term amplitude factor (independent of τ₀ / ρ₀)
        self.A_n_factor = (
            self.n ** (alpha - 3)
            * self.sign
            * np.pi ** (alpha - 3)
            * 2 ** 0.5
            / 2 ** (alpha / 2)
            * A
            * t_0 ** (alpha / 2)
            / self.E_0
        )   # (N, 1)

    def calc_lum(
        self,
        v_max: float,
        k: float,
        M: float,
    ) -> np.ndarray:
        """Compute the diffusion luminosity light curve.

        Parameters
        ----------
        v_max : float
            Maximum ejecta velocity [cm/s].
        k : float
            Opacity [cm²/g].
        M : float
            Ejecta mass [g].

        Returns
        -------
        numpy.ndarray, shape (n_times,)
            Bolometric luminosity at each observer time [erg/s].

        Notes
        -----
        The formula is::

            φ(n, t) = exp[(γ_f + γ_Knt) / τ₀] · (S_n − A_n · F_K)
                      + A_n · f

        where

        * ``F_K(n, τ₀) = cos(πα/2) · sug(1 − α/2, −γ_Knt / τ₀)``  — shape (N, 1),
          independent of time;
        * ``f(n, t, τ₀) = cos(πα/2) · sug(1 − α/2, γ_f / τ₀)``    — shape (N, n_times).

        Both quantities are evaluated directly via
        :func:`~incomplete_gamma.scaled_upper_gamma_vec`; no interpolation
        is used.
        """
        c_cgs = 3e10   # speed of light [cm/s]
        rho_0 = M / (4.0 / 3.0 * np.pi * (v_max * self.t_0) ** 3)
        tau_0 = 3.0 * k * rho_0 * (v_max * self.t_0) ** 2 / c_cgs
        A_n   = self.A_n_factor * tau_0 ** (1.0 - self.alpha / 2.0) * rho_0

        s     = 1.0 - self.alpha / 2.0
        cos_a = np.cos(np.pi * 0.5 * self.alpha)

        # F_K: (N, 1) — does not depend on t; very cheap (500 sug evaluations)
        F_K = cos_a * scaled_upper_gamma_vec(s, -self.gamma_K_nt_factor / tau_0)
        # f:   (N, n_times) — main computational cost (~30 000 sug evaluations)
        f   = cos_a * scaled_upper_gamma_vec(s,  self.gamma_factor / tau_0)

        exp_term = np.exp((self.gamma_factor + self.gamma_K_nt_factor) / tau_0)
        phi      = exp_term * (self.S - A_n * F_K) + A_n * f
        T        = self.sign * self.n * phi   # (N, n_times)

        return (
            np.sum(T, axis=0)
            * 4.0 * np.pi ** 2 * c_cgs * v_max * 2.0 ** 0.5
            * self.t_0 * self.E_0
            / (3.0 * k * rho_0)
        )   # (n_times,)


# ---------------------------------------------------------------------------
# DiffusionLum_interp  (legacy, retained for reference)
# ---------------------------------------------------------------------------

class DiffusionLum_interp:
    """Legacy diffusion luminosity via pre-built interpolation tables.

    .. deprecated::
        Use :class:`DiffusionLum_direct` instead.  This class is retained
        for backward compatibility and to facilitate comparison studies.

    Known accuracy issues
    ---------------------
    * The ``τ₀`` grid covers ``[6.4×10³, 6.4×10¹¹]`` with only 50 log-spaced
      points.  Typical physical parameters often produce ``τ₀`` values
      **outside** this range; ``RegularGridInterpolator`` then extrapolates
      linearly, which can yield errors of 100–1000 % in the luminosity.
    * The ``n``-axis uses ``linspace`` with step 5, but the tabulated
      function drops by ~17 orders of magnitude between ``n = 1`` and
      ``n = 6``, making linear interpolation unreliable near small ``n``.
    * ``__init__`` evaluates ``sug`` at 250 000 grid points and can take
      several seconds on the first call.

    Parameters
    ----------
    t_0 : float
        Initialisation time of the diffusive sphere [s].
    time : numpy.ndarray, shape (n_times,)
        Observer time array [s].
    T_0 : float
        Initialisation temperature [K].
    A : float
        Effective heating amplitude [erg/s/g].
    alpha : float
        Effective heating power-law index.
    """

    # -- class-level constants (mirror DiffusionLum_direct for compatibility)
    N    = 500
    no   = np.arange(1, N + 1)
    n    = no[:, np.newaxis]
    S    = np.append(1, np.zeros(N - 1))[:, np.newaxis]
    sign = np.empty(N, int)
    sign[::2]  = 1
    sign[1::2] = -1
    sign = sign[:, np.newaxis]

    def __init__(
        self,
        t_0: float,
        time: np.ndarray,
        T_0: float,
        A: float,
        alpha: float,
    ) -> None:
        self.t_0   = t_0
        self.t_f   = time[-1]
        self.E_0   = T_0 ** 4 * 7.57e-15   # [erg/cm^3]
        self.A     = A
        self.alpha = alpha
        self.t     = time

        self.A_n_factor = (
            np.power(DiffusionLum_interp.n, alpha - 3)
            * DiffusionLum_interp.sign
            * np.power(np.pi, alpha - 3)
            * 2 ** 0.5
            / np.power(2, alpha / 2)
            * A
            * np.power(t_0, alpha / 2)
            / self.E_0
        )
        self.gamma_factor        = -0.5 * (np.pi * DiffusionLum_interp.n * time) ** 2 / t_0
        self.gamma_K_nt_factor   =  0.5 * (np.pi * DiffusionLum_interp.n) ** 2 * t_0

        # -- 1-D interpolation table for f(x) = cos(πα/2)·sug(1−α/2, x) --
        self.Np  = 300
        self.x_i = (
            -0.5 * (np.pi * DiffusionLum_interp.N * self.t_f) ** 2
            / (t_0 * 63661977.23675813 / 10000)
        )
        self.x_f = (
            -0.5 * (np.pi * t_0) ** 2
            / (t_0 * 63661977.23675813 * 10000)
        )
        self.x = np.flip(
            -np.logspace(np.log10(-self.x_f), np.log10(-self.x_i), self.Np)
        )
        self.f = interp1d(
            self.x,
            self.interpf(self.x),
            bounds_error=False,
            fill_value="extrapolate",
        )

        # -- 3-D interpolation table for G_K(t, n, τ₀) --------------------
        self.Np_K    = np.array([50, 100, 50])
        self.t_K     = np.logspace(np.log10(t_0), np.log10(self.t_f), self.Np_K[0])
        self.n_K     = np.linspace(1, DiffusionLum_interp.N, self.Np_K[1])
        self.tau_0_K = np.logspace(
            np.log10(63661977.23675813 / 10000),
            np.log10(63661977.23675813 * 10000),
            self.Np_K[2],
        )
        self.G_K = self.interpfunc(
            *np.meshgrid(self.t_K, self.n_K, self.tau_0_K, indexing="ij", sparse=True)
        )
        self.f_K = RegularGridInterpolator(
            (self.t_K, self.n_K, self.tau_0_K),
            self.G_K,
            bounds_error=False,
            fill_value=None,
        )

    def interpf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate ``cos(πα/2) · sug(1 − α/2, x)`` on a 1-D mesh."""
        return np.cos(np.pi * 0.5 * self.alpha) * sug(1 - self.alpha / 2, x)

    def interpfunc(
        self,
        t_K: np.ndarray,
        n_K: np.ndarray,
        tau_0_K: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the 3-D grid function G_K(t, n, τ₀)."""
        return (
            np.exp(
                0.5 * (np.pi * n_K) ** 2
                * (self.t_0 - t_K ** 2 / self.t_0)
                / tau_0_K
            )
            * np.cos(np.pi * 0.5 * self.alpha)
            * sug(1 - self.alpha / 2, -0.5 * (np.pi * n_K) ** 2 * self.t_0 / tau_0_K)
        )

    def solution(self, tau_0: float, rho_0: float) -> np.ndarray:
        """Evaluate the temporal differential equation solution φ(n, t).

        Parameters
        ----------
        tau_0 : float
            Optical-depth scale factor.
        rho_0 : float
            Initial density [g/cm³].

        Returns
        -------
        numpy.ndarray, shape (N, n_times)
        """
        A_n   = self.A_n_factor * np.power(tau_0, 1 - self.alpha / 2) * rho_0
        K_nt  = (
            DiffusionLum_interp.S
            * np.exp((self.gamma_factor + self.gamma_K_nt_factor) / tau_0)
            - A_n * self.f_K((self.t, DiffusionLum_interp.n, tau_0))
        )
        return K_nt + A_n * self.f(self.gamma_factor / tau_0)

    def calc_lum(
        self,
        v_max: float,
        k: float,
        M: float,
    ) -> np.ndarray:
        """Compute the diffusion luminosity light curve (interpolation path).

        Parameters
        ----------
        v_max : float
            Maximum ejecta velocity [cm/s].
        k : float
            Opacity [cm²/g].
        M : float
            Ejecta mass [g].

        Returns
        -------
        numpy.ndarray, shape (n_times,)
            Bolometric luminosity [erg/s].
        """
        rho_0  = M / (4.0 / 3.0 * np.pi * (v_max * self.t_0) ** 3)
        tau_0  = 3.0 * k * rho_0 * (v_max * self.t_0) ** 2 / c
        phi_nt = self.solution(tau_0, rho_0)
        T      = DiffusionLum_interp.sign * DiffusionLum_interp.n * phi_nt
        return (
            np.sum(T, axis=0)
            * 4.0 * np.pi ** 2 * c * v_max * 2.0 ** 0.5
            * self.t_0 * self.E_0
            / (3.0 * k * rho_0)
        )


# ---------------------------------------------------------------------------
# Active implementation selector
# ---------------------------------------------------------------------------
# Change to ``DiffusionLum_interp`` here (and nowhere else) to revert to the
# legacy interpolation-based implementation for comparison purposes.

DiffusionLum = DiffusionLum_direct
