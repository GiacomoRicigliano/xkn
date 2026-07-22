# xkn — Kilonova Light-Curve Framework

xkn is a Python package for generating bolometric and broadband kilonova
light curves, comparing them to photometric data, and computing
log-likelihoods for use in Bayesian inference pipelines.

---

## Installation

**Dependencies** (all pip-installable):

```
pip install astropy matplotlib mpmath numpy scipy
```

**From source:**

```bash
git clone https://github.com/GiacomoRicigliano/xkn
cd xkn
pip install .
```

Python ≥ 3.10 and SciPy ≥ 1.14 are required (the legacy `interp2d`
function was removed in SciPy 1.14 and is no longer used by xkn).

---

## Quick start

```python
from xkn import MKN, MKNConfig

# 1. Load the configuration file
config = MKNConfig("examples/kn_config.ini")
mkn    = MKN(*config.get_params())

# 2. Set the free variables
inputs = {
    "view_angle":          0.524,   # rad
    "distance":            40,      # Mpc
    "m_ej_dynamics":       0.03,    # Msun
    "vel_dynamics":        0.13,    # c
    "high_lat_op_dynamics": 5,      # cm^2/g
    "low_lat_op_dynamics": 20,
    "m_ej_secular":        0.08,
    "vel_secular":         0.06,
    "op_secular":          5,
    "m_ej_wind":           0.02,
    "vel_wind":            0.1,
    "high_lat_op_wind":    1,
    "low_lat_op_wind":     5,
}
mkn_vars = config.get_vars(inputs)

# 3. Compute bolometric light curves
lum_bol, lum_photo, radius_photo, T_photo, \
    lum_shells, T_shells, lum_bol_raw = mkn.calc_lightcurve_vars(mkn_vars)

# 4. Compute broadband magnitudes
mags = mkn.calc_magnitudes(mkn_vars)   # dict: {wavelength_nm: {"time": ..., "mag": ...}}

# 5. Compute log-likelihood against loaded data
log_like = mkn.calc_log_like(mkn_vars)
```

See `examples/example.py` for a complete runnable script.

---

## Repository layout

```
xkn/
├── xkn/                      # Source package
│   ├── __init__.py           # Exports MKN, MKNConfig
│   ├── mkn.py                # MKN top-level class
│   ├── config.py             # MKNConfig: parameter/variable I/O
│   ├── ejecta.py             # Multi-component ejecta orchestration
│   ├── shell.py              # Single-component expansion and photosphere
│   ├── diffusion_luminosity.py   # Ricigliano/Lippold diffusion model
│   ├── incomplete_gamma.py   # Vectorized scaled upper incomplete gamma
│   ├── thermalization.py     # BKWM and other ε_th models
│   ├── nuclear_heat.py       # RP/PBR/LR/K heating-rate models
│   ├── filters.py            # Filter I/O, flux→magnitude conversion
│   ├── angular_distribution.py   # Angular discretisation laws
│   ├── utils.py              # Physical constants and helper functions
│   ├── plotting.py           # Magnitude plotting
│   ├── kappa_2_ye.py         # Opacity ↔ electron fraction conversion
│   ├── heating_function.py   # Tabulated LR heating interpolation
│   ├── import_NR_data.py     # Numerical-relativity ejecta profile loader
│   ├── extrapolation_2d.py   # 2-D spline with smooth extrapolation
│   └── interp_tables/        # Binary/ASCII look-up tables
├── examples/
│   ├── example.py            # End-to-end usage example
│   └── kn_config.ini         # Example configuration file
├── filter_data/              # AT2017gfo photometric data
├── flux_factor_data/         # Observer-projection tables
└── README.md
```

---

## Configuration file

The model is steered through an INI-format configuration file divided into
sections. Each ejecta component (e.g. `[dynamical]`, `[secular]`, `[wind]`)
has a matching `[<name>_vars]` section for the free variables.

A full listing of supported parameters and variables is printed by:

```python
config.get_info()
```

### Key global parameters

| Parameter | Description |
|---|---|
| `lc_model` | Light-curve model: `ricigliano_lippold`, `grossman`, `villar` |
| `slices_num` | Number of polar angular bins (12 / 18 / 24 / 30) |
| `slices_dist` | Angular discretisation law: `uniform`, `cos_uniform` |
| `t_scale` | Time grid: `lin`, `log`, or `measures` (use data epochs) |
| `t_0` | Diffusion initialisation time [s] (Ricigliano/Lippold only) |
| `T_0` | Diffusion initialisation temperature [K] |
| `filter_usage` | `measures` (compare to data) or `properties` (model only) |
| `filter_dictionary` | Filter set: `telescopes`, `AT2017gfo`, `lsst`, … |

### Key component parameters

| Parameter | Description |
|---|---|
| `mass_dist` | Mass angular distribution: `uniform`, `sin`, `sin2`, `cos2`, `step` |
| `vel_dist` | Velocity angular distribution (same options + `abscos`) |
| `op_dist` | Opacity angular distribution |
| `therm_model` | Thermalization model: `BKWM`, `BKWM_1d`, `power_law`, `cnst` |
| `heat_model` | Heating model: `RP`, `PBR`, `LR`, `K` |

---

## Output quantities

`calc_lightcurve_vars` returns a 7-tuple, all arrays indexed
`(n_angles, n_times)` unless noted:

| Name | Shape | Unit | Description |
|---|---|---|---|
| `lum_bol` | `(n_angles, n_times)` | erg/s | Total bolometric luminosity |
| `lum_photo` | `(n_angles, n_times)` | erg/s | Photospheric contribution |
| `radius_photo` | `(n_angles, n_times)` | cm | Photospheric radius |
| `T_photo` | `(n_angles, n_times)` | K | Photospheric temperature |
| `lum_shells` | `(…, n_shells)` or `None` | erg/s | Thin-shell luminosity |
| `T_shells` | `(…, n_shells)` or `None` | K | Thin-shell temperature |
| `lum_bol_raw` | `(n_angles, n_times)` | erg/s | Total lum. (no thin-shell correction) |

> **Note:** the total bolometric luminosity accounting for both hemispheres
> is `2 * np.sum(lum_bol, axis=0)`.

---

## Adding ejecta components

Any number of ejecta components can be added by repeating the component
parameter and variable sections in the config file under a unique name:

```ini
[dynamical]
mass_dist = step
...

[dynamical_vars]
m_ej = 0.03
...

[wind]
mass_dist = uniform
...

[wind_vars]
m_ej = 0.02
...
```

---

## Performance notes

* The `ricigliano_lippold` model uses `DiffusionLum_direct` by default,
  which evaluates the Fourier-series solution analytically via a vectorized
  NumPy implementation of the scaled upper incomplete gamma function.  No
  interpolation grids are built and there is no JIT warm-up delay.
* For MCMC applications a module-level cache avoids reconstructing
  `DiffusionLum` objects when parameters are unchanged between calls.
* The legacy interpolation-based `DiffusionLum_interp` is retained in
  `diffusion_luminosity.py` for comparison; activate it by changing the
  one-line selector at the bottom of that file.

---

## Development status

The package is under active development. Some features are incomplete
(notably the `LR` heating model integration).
Please check the repository regularly for updates and open issues for bug
reports or feature requests.

---

## Citation

If you use **xkn** in your research, please cite:

> Ricigliano G., Perego A., Borhanian S., Loffredo E., Kawaguchi K., Bernuzzi S., Lippold L. C.,
> *xkn: a semi-analytic framework for the modelling of kilonovae*,
> Monthly Notices of the Royal Astronomical Society **529**, 647–663 (2024).
> https://doi.org/10.1093/mnras/stae572

```bibtex
@article{Ricigliano2024xkn,
  author = {Ricigliano, Giacomo and Perego, Albino and Borhanian, Ssohrab and
            Loffredo, Eleonora and Kawaguchi, Kyohei and Bernuzzi, Sebastiano
            and Lippold, Lukas Chris},
  title = {xkn: a semi-analytic framework for the modelling of kilonovae},
  journal = {Monthly Notices of the Royal Astronomical Society},
  volume = {529},
  number = {1},
  pages = {647--663},
  year = {2024},
  doi = {10.1093/mnras/stae572}
}
```
