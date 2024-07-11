import sys

import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize

from . import import_NR_data as nrd
from . import angular_distribution as ad
from . import diffusion_luminosity as dl
from . import heating_function as hf
from . import kappa_2_ye as k2y
from . import nuclear_heat as nh
from . import thermalization as therm
from . import utils


class Shell(object):
    """
    Ejecta shell class
    Required arguments are:
    name,          # name of the shell     (dynamical, wind, secular)
    mass_dist,     # mass distribution     (step, uniform, continuos)
    vel_dist,      # velocity distribution (step, uniform, continuos)
    op_dist,       # opacity distribution  (step, uniform, continuos)
    therm_model,   # thermal model         (BKWM_dens, BKWM, cnst)
    heat_model,    # nuclear heating model (Ye depedence True, False)
    """

    def __init__(self, name, shell_params, **kwargs):
        self.name = name
        self.params = shell_params
        self.mass_dist = ad.MassAngularDistribution(shell_params["mass_dist"])
        self.vel_dist = ad.VelocityAngularDistribution(shell_params["vel_dist"])
        self.op_dist = ad.OpacityAngularDistribution(shell_params["op_dist"])
        self.expansion_model = utils.ExpansionModelSingleSpherical("GK")
        self.thermalization = therm.Thermalization(shell_params["therm_model"])
        self.nuclear_heat = nh.NuclearHeat(shell_params["heat_model"])
        self.kappa_2_ye = k2y.Kappa2Ye(shell_params["ye_k_dep"])
        self.heating_function = hf.HeatingFunction(
            shell_params["heat_model"], shell_params["entropy"], shell_params["tau"]
        )

    def expansion_angular_distribution(
        self, angles, omegas, times, shell_vars, glob_vars, glob_params, **kwargs
    ):
        if glob_params["lc_model"] == "grossman":
            func = self.expansion_angular_distribution_grossman
        elif glob_params["lc_model"] == "ricigliano_lippold":
            func = self.expansion_angular_distribution_ricigliano_lippold
        elif glob_params["lc_model"] == "villar":
            func = self.expansion_angular_distribution_villar
        else:
            sys.exit(
                "Specified lightcurve model is not known! Please choose an available method: diff_lum, grossman, or villar."
            )

        return func(angles, omegas, times, shell_vars, glob_vars, glob_params, **kwargs)

    ###-------------------------------------------------------------------------------------------------
    ###-------------------------------------------------------------------------------------------------
    # ------Utility functions-------------------------------------------------------------------------------
    ###-------------------------------------------------------------------------------------------------
    ###-------------------------------------------------------------------------------------------------

    def set_mass_vel_opacity_ye_entropy_tau_profiles(
        self, angles, shell_vars, glob_vars, glob_params, **kwargs
    ):
        if self.params["NR_data"]:
            self.mass_ej, self.vel_rms, self.ye = nrd.importNRprofiles(
                self.params["NR_data_filename"], angles
            )
            if (
                "low_lat_op" in shell_vars and shell_vars["low_lat_op"] is not None
            ) or ("max_op" in shell_vars and shell_vars["max_op"] is not None):
                self.opacity = self.op_dist(angles, **shell_vars)
            else:
                self.opacity = self.kappa_2_ye("ye", self.ye)
        else:
            if shell_vars["m_ej"] is not None:
                m_tot = float(shell_vars["m_ej"])
            elif shell_vars["xi_disk"] is not None:
                m_tot = float(shell_vars["xi_disk"]) * float(glob_vars["m_disk"])
            else:
                raise NameError(
                    f"Please specify either m_ej or xi_disk for shell: {self.name}"
                )
            self.mass_ej = self.mass_dist(angles, m_tot=m_tot, **shell_vars)
            self.vel_rms = self.vel_dist(angles, **shell_vars)
            self.opacity = self.op_dist(angles, **shell_vars)
            self.ye = self.kappa_2_ye("opacity", self.opacity)
        self.entropy = self.params["entropy"] * np.ones(len(self.opacity))
        if self.params["tau"] is not None:
            self.tau = self.params["tau"] * np.ones(len(self.opacity))
        else:
            self.tau = 1 / self.vel_rms

    def generate_diff_lums(
        self, angles, times, shell_vars, glob_vars, glob_params, **kwargs
    ):
        self.set_mass_vel_opacity_ye_entropy_tau_profiles(
            angles, shell_vars, glob_vars, glob_params
        )
        self.diff_lums = dl.generate_diff_lums(
            self.ye,
            self.entropy,
            self.tau,
            times,
            glob_vars,
            self.params,
            glob_params,
            **kwargs,
        )

    ###-------------------------------------------------------------------------------------------------
    ###-------------------------------------------------------------------------------------------------
    # ------Optically thin shells------------------------------------------------------------------------
    ###-------------------------------------------------------------------------------------------------
    ###-------------------------------------------------------------------------------------------------

    def expansion_angular_distribution_thin_layers(
        self, angles, omegas, times, shell_vars, glob_vars, glob_params, **kwargs
    ):

        if "diff_lums" in self.__dict__:
            alphas = np.array([diff_lum.alpha for diff_lum in self.diff_lums])
        else:
            alphas = glob_params["alpha"] * np.ones_like(omegas)

        # velocity discretization
        if glob_params["shell_const"] == "vel":
            self.v_shells = np.linspace(0, self.vel_woll, glob_params["n_thin"])
        elif glob_params["shell_const"] == "mass":
            self.v_shells = np.array(
                [
                    np.interp(
                        np.linspace(mass_scaled, 0, glob_params["n_thin"]),
                        utils.Mv_Woll(mass_scaled, np.linspace(1.0, 0.0, 50)),
                        vel_woll * np.linspace(1.0, 0.0, 50),
                    )
                    for vel_woll, mass_scaled in zip(self.vel_woll, self.mass_scaled)
                ]
            ).T
        else:
            sys.exit('Choose "vel" or "mass" as constant shell mode.')

        # thin regime luminosity computation #
        if self.params["therm_model"] == "BKWM_dens":
            self.lum_shells = np.zeros(
                (len(angles), len(times), glob_params["n_thin"] - 1)
            )
            for i, v_shells in enumerate(self.v_shells.T):

                e_nuc = np.array(
                    [
                        self.nuclear_heat(
                            time / (1 - (v_shells[:-1] / self.vel_woll[i]) ** 2),
                            omegas[i],
                            self.mass_ej[i],
                            self.vel_rms[i],
                            alphas[i],
                            glob_params["t0eps"],
                            glob_params["sigma0"],
                            glob_vars["eps0"],
                            glob_params["cnst_eff"],
                            glob_params["idx_eff"],
                            self.thermalization,
                            self.kappa_2_ye,
                            self.heating_function,
                            opacity=self.opacity[i],
                            ye=self.ye[i],
                            s=self.entropy[i],
                            tau=self.tau[i],
                            cnst_a_eps_nuc=glob_params["a_eps_nuc"],
                            cnst_b_eps_nuc=glob_params["b_eps_nuc"],
                            cnst_t_eps_nuc=glob_params["t_eps_nuc"],
                            shell=self.name,
                        )[0]
                        for time in times
                    ]
                )

                self.lum_shells[i] = (
                    np.array(
                        [
                            (
                                utils.Mv_Woll(
                                    self.mass_scaled[i],
                                    v_shells[:-1] / self.vel_woll[i],
                                )
                                - utils.Mv_Woll(
                                    self.mass_scaled[i], v_shells[1:] / self.vel_woll[i]
                                )
                            )
                            * enuc
                            / (1 - (v_shells[:-1] / self.vel_woll[i]) ** 2) ** alphas[i]
                            * omegas[i]
                            / utils.fourpi
                            for enuc in e_nuc
                        ]
                    )
                    * glob_vars["nuc_fac"]
                )

        elif self.params["therm_model"] in ["cnst", "power_law", "BKWM"]:
            e_nuc = self.nuclear_heat(
                times,
                omegas,
                self.mass_ej,
                self.vel_rms,
                alphas,
                glob_params["t0eps"],
                glob_params["sigma0"],
                glob_vars["eps0"],
                glob_params["cnst_eff"],
                glob_params["idx_eff"],
                self.thermalization,
                self.kappa_2_ye,
                self.heating_function,
                opacity=self.opacity,
                ye=self.ye,
                s=self.entropy,
                tau=self.tau,
                cnst_a_eps_nuc=glob_params["a_eps_nuc"],
                cnst_b_eps_nuc=glob_params["b_eps_nuc"],
                cnst_t_eps_nuc=glob_params["t_eps_nuc"],
                shell=self.name,
            ).T

            self.lum_shells = np.einsum(
                "ijk->kij",
                np.array(
                    (
                        utils.Mv_Woll(
                            self.mass_scaled, self.v_shells[:-1] / self.vel_woll
                        )
                        - utils.Mv_Woll(
                            self.mass_scaled, self.v_shells[1:] / self.vel_woll
                        )
                    )
                    * e_nuc[:, None, :]
                    * omegas
                    / utils.fourpi
                )
                * glob_vars["nuc_fac"],
            )

        else:
            sys.exit("Unknown thermalization efficiency model used for thin layers\n")

        return (
            self.v_shells.T,
            self.lum_shells,
            self.vel_woll,
            self.mass_scaled,
            self.ye,
        )

    ###-------------------------------------------------------------------------------------------------
    ###-------------------------------------------------------------------------------------------------
    # ------Ricigliano-Lippold model---------------------------------------------------------------------
    ###-------------------------------------------------------------------------------------------------
    ###-------------------------------------------------------------------------------------------------

    def expansion_angular_distribution_ricigliano_lippold(
        self, angles, omegas, times, shell_vars, glob_vars, glob_params, **kwargs
    ):

        if "diff_lums" in self.__dict__:
            pass
        elif "diff_lums" in kwargs:
            self.diff_lums = kwargs["diff_lums"]
        else:
            sys.exit("Please generate or provide list of diff_lum objects.")

        self.set_mass_vel_opacity_ye_entropy_tau_profiles(
            angles, shell_vars, glob_vars, glob_params
        )

        self.mass_scaled = self.mass_ej * utils.fourpi / omegas * 2e33
        vel_max = self.vel_rms * np.sqrt(5 / 3) * 3e10  # Ricigliano velocity profile
        self.lum_bol = np.array(
            [
                glob_vars["nuc_fac"]
                * diff_lum.calc_lum(vel_max[i], self.opacity[i], self.mass_scaled[i])
                * omegas[i]
                / utils.fourpi
                for i, diff_lum in enumerate(self.diff_lums)
            ]
        )  # adjusting luminosity to bin size; *glob_vars['eps0']/2e18
        self.lum_bol[self.lum_bol < 0] = (
            0  # thin regime model break down (setting it to negligable value but big enough so there are no errors)
        )

        ### Wollaeger velocity profile approximate photosphere calculation###
        t0 = 1.0  # arbitrary value
        self.vel_woll = 3 * self.vel_rms * 3e10  # Wollaeger vlaw
        rho0 = (
            105
            / 32
            * self.mass_scaled
            / (np.pi * (self.vel_woll * t0) ** 3)
            / glob_params["tau_photo"]
        )
        t3 = np.sqrt(
            27 * self.mass_scaled * self.opacity / (8 * np.pi * self.vel_woll**2)
        )
        t2 = 0.1623 * t3
        points_t = np.array([np.zeros_like(t2), t2, t3]).T
        points_R = np.array(
            [
                np.zeros_like(t2),
                utils.R_early(t2, self.vel_woll, self.opacity, rho0, t0),
                np.zeros_like(t2),
            ]
        ).T
        p0 = [-1e3, 2e9, 0]
        sols = np.array(
            [
                optimize.leastsq(utils.error, p0, args=(p_t, p_R))[0]
                for p_t, p_R in zip(points_t, points_R)
            ]
        ).T

        # joining early approximant and parabola fit in R_ph calculation
        _, times_mesh = np.meshgrid(self.vel_woll, times)
        rtmp1 = utils.R_early(times_mesh, self.vel_woll, self.opacity, rho0, t0)
        rtmp2 = utils.parab(sols, times_mesh)
        self.radius_photo = np.where(times_mesh < t2, rtmp1, rtmp2)
        self.radius_photo[times_mesh > t3] = 0  # avoid negative radii

        if "thin_shells" in glob_params and glob_params["thin_shells"]:
            self.radius_photo = self.radius_photo.T
            return self.radius_photo, self.lum_bol

        if shell_vars["T_floor"] is None:
            T_f = utils.calc_Tfloor(
                "ye", self.ye, glob_vars["T_floor_LA"], glob_vars["T_floor_Ni"]
            )
        else:
            T_f = shell_vars["T_floor"]

        self.radius_photo = np.minimum(
            self.radius_photo,
            np.sqrt(
                utils.fourpi / omegas * self.lum_bol.T / (utils.fourpisigma_SB * T_f**4)
            ),
        ).T

        return self.radius_photo, self.lum_bol

    ###-------------------------------------------------------------------------------------------------
    ###-------------------------------------------------------------------------------------------------
    # ------Grossman model-------------------------------------------------------------------------------
    ###-------------------------------------------------------------------------------------------------
    ###-------------------------------------------------------------------------------------------------

    def expansion_angular_distribution_grossman(
        self, angles, omegas, times, shell_vars, glob_vars, glob_params, **kwargs
    ):

        self.set_mass_vel_opacity_ye_entropy_tau_profiles(
            angles, shell_vars, glob_vars, glob_params
        )

        vel, m_vel, t_diff, t_fs = self.expansion_model(
            omegas,
            self.mass_ej,
            self.vel_rms,
            self.opacity,
            glob_params["vel_min"],
            glob_params["vel_num"],
            glob_params["vel_scale"],
            glob_params["vel_law"],
        )

        v_fs = np.array(
            [np.interp(times, tmp1[::-1], tmp2[::-1]) for tmp1, tmp2 in zip(t_fs, vel)]
        )
        m_rad = np.array(
            [
                np.interp(times, tmp1[::-1], tmp2[::-1])
                for tmp1, tmp2 in zip(t_diff, m_vel)
            ]
        )
        if glob_params["rad_shell"]:
            m_rad -= np.array(
                [
                    np.interp(times, tmp1[::-1], tmp2[::-1])
                    for tmp1, tmp2 in zip(t_fs, m_vel)
                ]
            )

        self.lum_bol = m_rad * self.nuclear_heat(
            times,
            omegas,
            self.mass_ej,
            self.vel_rms,
            glob_params["alpha"],
            glob_params["t0eps"],
            glob_params["sigma0"],
            glob_vars["eps0"],
            glob_params["cnst_eff"],
            glob_params["idx_eff"],
            self.thermalization,
            self.kappa_2_ye,
            self.heating_function,
            opacity=self.opacity,
            ye=self.ye,
            s=self.entropy,
            tau=self.tau,
            cnst_a_eps_nuc=glob_params["a_eps_nuc"],
            cnst_b_eps_nuc=glob_params["b_eps_nuc"],
            cnst_t_eps_nuc=glob_params["t_eps_nuc"],
            shell=self.name,
        )

        if shell_vars["T_floor"] is None:
            T_f = utils.calc_Tfloor(
                "opacity",
                self.opacity,
                glob_vars["T_floor_LA"],
                glob_vars["T_floor_Ni"],
            )
        else:
            T_f = shell_vars["T_floor"]

        self.radius_photo = np.minimum(
            ((v_fs * utils.c) * times),
            np.sqrt(
                utils.fourpi / omegas * self.lum_bol.T / (utils.fourpisigma_SB * T_f**4)
            ).T,
        )

        return self.radius_photo, self.lum_bol

    ###-------------------------------------------------------------------------------------------------
    ###-------------------------------------------------------------------------------------------------
    # ------Villar model---------------------------------------------------------------------------------
    ###-------------------------------------------------------------------------------------------------
    ###-------------------------------------------------------------------------------------------------

    # NUCLEAR HEATING RATE
    # luminosity integrand
    def L_in(self, omegas, times, glob_vars, glob_params):
        return (
            self.mass_ej
            * utils.Msun
            * nh.heat_rate_K(
                times,
                omegas,
                self.mass_ej,
                self.vel_rms,
                glob_params["alpha"],
                glob_params["t0eps"],
                glob_params["sigma0"],
                glob_vars["eps0"],
                glob_params["cnst_eff"],
                glob_params["idx_eff"],
                self.thermalization,
                self.kappa_2_ye,
                self.heating_function,
                opacity=self.opacity,
                cnst_a_eps_nuc=glob_params["a_eps_nuc"],
                cnst_b_eps_nuc=glob_params["b_eps_nuc"],
                cnst_t_eps_nuc=glob_params["t_eps_nuc"],
            ).T
        ).T

    # BOLOMETRIC LUMINOSITY
    # Note: to avoid overflow in e^((t/td)^2) a 'cut' is introduced. If (t/td)^2 > 500, it is fixed to 500.
    # Now L is computed also at later times.
    def L_villar(self, times, omegas, glob_vars, glob_params, NN=100):
        td = np.sqrt(
            (8 * self.opacity * np.pi * self.mass_ej * utils.Msun / omegas)
            / (13.4 * self.vel_rms * utils.c**2)
        )
        _, td_NN = np.meshgrid(np.zeros(NN), td)
        _, td = np.meshgrid(times, td)
        init_x = np.logspace(-3, np.log10(times[0]), NN)
        init_int = integrate.trapz(
            self.L_in(omegas, init_x, glob_vars, glob_params)
            * np.exp((init_x**2) / (td_NN**2))
            * (init_x / td_NN),
            init_x,
        )
        _, init_int = np.meshgrid(times, init_int)
        t_cut = (times / td) ** 2
        t_cut[t_cut > 500] = 500
        tmp = (
            self.L_in(omegas, times, glob_vars, glob_params)
            * np.exp(t_cut)
            * (times / td)
        )
        return (
            integrate.cumtrapz(tmp, times, initial=0)
            + init_int * np.exp(-((times / td) ** 2.0)) / td
        )

    def expansion_angular_distribution_villar(
        self, angles, omegas, times, shell_vars, glob_vars, glob_params, **kwargs
    ):
        self.set_mass_vel_opacity_ye_entropy_tau_profiles(
            angles, shell_vars, glob_vars, glob_params
        )

        self.lum_bol = self.L_villar(times, omegas, glob_vars, glob_params, NN=100)
        if shell_vars["T_floor"] is None:
            T_floor = utils.calc_Tfloor(
                "opacity",
                self.opacity,
                glob_vars["T_floor_LA"],
                glob_vars["T_floor_Ni"],
            )
        else:
            T_floor = shell_vars["T_floor"]
        _, self.radius_photo = self.vel_rms * np.meshgrid(self.vel_rms, times)
        self.radius_photo = np.minimum(
            self.radius_photo,
            np.sqrt(
                utils.fourpi
                / omegas
                * self.lum_bol.T
                / (utils.fourpisigma_SB * T_floor**4)
            ),
        ).T

        return self.radius_photo, self.lum_bol
