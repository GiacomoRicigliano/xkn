import os

import numpy as np

##############################################################################################
# class to handle parameters reading and variables assignment from config file
##############################################################################################


class MKNConfig:

    def __init__(self, config_path):
        from configparser import ConfigParser
        from copy import deepcopy

        # check if the config file exists
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found!")

        self.path = config_path
        self.config = ConfigParser()
        self.config.optionxform = str
        self.config.read(self.path)

        # shell_params and glob_params
        self.read_params()

        # mkn_vars
        self.init_vars()

    ### read the shell_params and glob_params from the config file
    def read_params(self):
        from copy import deepcopy

        params_info_dict = {**glob_params_info_dict, **comp_params_info_dict}

        self.shell_params = {}
        self.glob_params = {}

        for section in self.config.sections():
            if "vars" in section:
                continue
            tmp = {}
            for key in self.config[section]:
                if self.config[section][key] == "None":
                    tmp[key] = None
                elif params_info_dict[key][0] == "bool":
                    tmp[key] = self.config[section].getboolean(key)
                elif params_info_dict[key][0] == "float":
                    tmp[key] = float(self.config[section][key])
                elif params_info_dict[key][0] == "int":
                    tmp[key] = int(self.config[section][key])
                elif params_info_dict[key][0] == "str":
                    tmp[key] = self.config[section][key]
                elif params_info_dict[key][0] == "int_list":
                    tmp[key] = [
                        int(val) for val in self.config[section][key].split(" ")
                    ]
                elif params_info_dict[key][0] == "float_list":
                    tmp[key] = [
                        float(val) for val in self.config[section][key].split(" ")
                    ]

            if section == "glob":
                self.glob_params = deepcopy(tmp)
            else:
                self.shell_params[section] = deepcopy(tmp)

    ### initialize the necessary tools for get_vars from the config file
    def init_vars(self):
        self.comps = [key for key in self.config.sections() if "vars" not in key]
        self.vars_secs = {
            key[:-5]: self.config[key]
            for key in self.config.sections()
            if "vars" in key
        }
        assert list(self.comps) == list(self.vars_secs)

        self.vars_free = {comp: {} for comp in self.comps}
        self.vars_fixed = {comp: {} for comp in self.comps}

        for comp, vars_sec in self.vars_secs.items():
            for key, val in vars_sec.items():
                if val == "None":
                    self.vars_fixed[comp][key] = None
                else:
                    try:
                        self.vars_fixed[comp][key] = float(val)
                    except ValueError:
                        try:
                            self.vars_fixed[comp][key] = self.config[
                                f"{comp}_vars"
                            ].getboolean(key)
                        except ValueError:
                            self.vars_free[comp][key] = self._correct_ext_vars_key(
                                val, key
                            )

    ### auxiallary functions to properly assign variable values in get_vars
    def _ident(self, x):
        return x

    def _correct_ext_vars_key(self, var, comp):
        import numpy as np

        if var in ["cos_iota", "cosi"]:
            return (var, np.arccos)
        else:
            return (var, self._ident)

    ### print information about the config file options for parameters and variables
    def get_params_info(self):
        params_info()

    def get_vars_info(self):
        vars_info()

    def get_info(self):
        info()

    ### getters for the shell_params and glob_params
    def get_shell_params(self):
        from copy import deepcopy

        return deepcopy(self.shell_params)

    def get_glob_params(self):
        from copy import deepcopy

        return deepcopy(self.glob_params)

    def get_params(self):
        return self.get_shell_params(), self.get_glob_params()

    ### getter for correctly assigned variables dictionary
    def get_vars(self, params):
        return {
            comp: {
                **{
                    var: val[1](params[val[0]])
                    for var, val in self.vars_free[comp].items()
                },
                **{var: val for var, val in self.vars_fixed[comp].items()},
            }
            for comp in self.comps
        }


############################################################
# configuration info functions
############################################################


def info():
    params_info()
    print()
    print()
    vars_info()


def params_info():
    print("PARAMETERS")
    print(
        "---------------------------------------------------------------------------------------------------------------------------"
    )
    print()
    max_len = np.amax(
        np.array([len(key) for key in {**glob_params_info_dict, **comp_vars_info_dict}])
    )

    print("global parameters")
    print("-------------------------\n")
    for key, val in glob_params_info_dict.items():
        tmp1 = key.ljust(max_len, " ")
        tmp2 = f"({val[0]})".rjust(7, " ")
        print(f"{tmp1} {tmp2} : {val[1]}")

    print()
    print("component parameters")
    print("-------------------------")
    for key, val in comp_params_info_dict.items():
        tmp1 = key.ljust(max_len, " ")
        tmp2 = f"({val[0]})".rjust(7, " ")
        print(f"{tmp1} {tmp2} : {val[1]}")


def vars_info():
    print("VARIABLES")
    print(
        "---------------------------------------------------------------------------------------------------------------------------"
    )
    print()
    max_len1 = np.amax(
        np.array([len(key) for key in {**glob_vars_info_dict, **comp_vars_info_dict}])
    )
    max_len2 = np.amax(
        np.array(
            [
                len(val[0])
                for val in {**glob_vars_info_dict, **comp_vars_info_dict}.values()
            ]
        )
    )

    print("global variables")
    print("-------------------------")
    for key, val in glob_vars_info_dict.items():
        tmp1 = key.ljust(max_len1, " ")
        tmp2 = f"{val[0]}".rjust(max_len2, " ")
        print(f"{tmp1} {tmp2} : {val[1]}")

    print()
    print("component variables")
    print("-------------------------")
    for key, val in comp_vars_info_dict.items():
        tmp1 = key.ljust(max_len1, " ")
        tmp2 = f"{val[0]}".rjust(max_len2, " ")
        print(f"{tmp1} {tmp2} : {val[1]}")


############################################################
# parameter configuration dictionaries
############################################################

glob_params_info_dict = {
    # lightcurve model
    "lc_model": ["str", "kilonova model [ricigliano_lippold, grossman, villar]"],
    # cosmology
    "cosmology": ["str", "cosmological model [None, Planck18]"],
    # filter data settings
    "filter_usage": [
        "str",
        "import data and correspondent filter dictionary (measures) or not (properties) [properties, measures]",
    ],
    "filter_dictionary": [
        "str",
        "dictionary used for the filters [telescopes, iso_calc, AT2017gfo, lsst]",
    ],
    "filter_dictionary_path": [
        "str",
        "path to the filter dictionary: if None, uses the default one to telescopes",
    ],
    "filter_data_path": ["str", "path to the data"],
    "lam_list": ["int_list", "values need to be single-space separated"],
    "lam_min": ["int", "minimum wavelenght in nm considered from data"],
    "lam_max": ["int", "maximum wavelenght in nm considered from data"],
    "mag_min": ["float", "minimum magnitude considered from data"],
    "mag_max": ["float", "maximum magnitude considered from data"],
    # data type
    "upper_limits": ["bool", "include upper limits from data"],
    # dered correction
    "dered_correction": [
        "bool",
        "correction for reddening: if True uses filters.dered_CCM to correct data for reddening.",
    ],
    "R_V": ["float", "parameters for filters.dered_CCM"],
    "EBV": ["float", "parameters for filters.dered_CCM"],
    "A_V": ["float", "parameters for filters.dered_CCM"],
    # angular distribution
    "slices_num": ["int", "number of slices along the polar angle [12, 18, 24, 30]"],
    "slices_dist": [
        "str",
        "discretization law for the polar angle [uniform, cos_uniform]",
    ],
    "omega_frac": ["float", "auxiliary parameter [1]"],
    # times handling
    "t_scale": [
        "str",
        "scale for the velocity [lin, log, measures] - measures will use data times",
    ],
    "t_num": ["int", "integer number of bins in time"],
    "t_min": [
        "float",
        "minimum time in s (has to be bigger than 1e4 s (0.12 d) for Ricigliano and Composed model)",
    ],
    "t_max": ["float", "maximum time in s"],
    "t_start_filter": [
        "float",
        "start time of filter data in respective units (e.g. Julian days for AT2017gfo)",
    ],
    "t_toll": [
        "float",
        "relative tolerance between time steps when extracting from measures",
    ],
    # vel handling
    "vel_min": ["float", "minimum velocity for the grossman model"],
    "vel_num": ["int", "number of velocity points for the grossman model"],
    "vel_scale": ["str", "scale for the velocity in the grossman model [lin, log]"],
    "vel_law": ["str", "relationship between vel_max and vel_rms [poly, uniform]"],
    # other parameters
    "alpha": ["float", "parameter for nuclear heating rate [1.3]"],
    "sigma0": ["float", "parameter for nuclear heating rate [0.11]"],
    "t0eps": ["float", "parameter for nuclear heating rate [1.3]"],
    "a_eps_nuc": ["float", "parameter for nuclear heating rate [0.5]"],
    "b_eps_nuc": ["float", "parameter for nuclear heating rate [2.5]"],
    "t_eps_nuc": ["float", "parameter for nuclear heating rate [1.0]"],
    "cnst_eff": ["float", "parameter for constant heating efficiency [2.958]"],
    "idx_eff": ["float", "parameter for power law heating efficiency [0.176]"],
    # grossman parameters
    "rad_shell": ["bool", "auxiliary switch [False]"],
    # ricigliano_lippold parameters
    "t_0": ["float", "initialization time of diffusive sphere in s"],
    "T_0": ["float", "initialization temperature of diffusive sphere in K"],
    "tau_photo": ["float", "optical depth at photosphere [0.66]"],
    # thin_shells parameters
    "thin_shells": [
        "bool",
        "if True applies the opcitally thin shells correction (only for ricigliano_lippold model)",
    ],
    "n_thin": ["int", "number of optically thin shells"],
    "shell_const": [
        "str",
        "uniform discretization in mass (mass) or velocity (vel) for optically thin shells",
    ],
    "n_heat": ["int", "auxiliary parameter [30]"],
}

comp_params_info_dict = {
    "mass_dist": [
        "str",
        "mass angular distribution law [uniform, sin, sin2, cos2, step]",
    ],
    "vel_dist": [
        "str",
        "velocity angular distribution law [uniform, sin, sin2, cos2, abscos, step]",
    ],
    "op_dist": [
        "str",
        "opacity angular distribution law [uniform, sin, sin2, cos2, abscos, step]",
    ],
    "therm_model": [
        "str",
        "thermalization model [BKWM, BKWM_dens, BKWM_1d, power_law, cnst]",
    ],
    "heat_model": ["str", "heating rate model [RP, PBR, LR, K]"],
    "ye_k_dep": ["str", "relation between opacity and electron fraction [TH]"],
    "entropy": ["float", "entropy value in k_B/baryon for heating rate"],
    "tau": ["float", "expansion timescale in ms for heating rate"],
    "NR_data": ["bool", "if True imports ejecta profile from NR_data_filename"],
    "NR_data_filename": ["str", "name of file containing ejecta profile"],
}

############################################################
# variable configuration dictionaries
############################################################

glob_vars_info_dict = {
    "sigma_sys": ["float", "error absorbing any source of systematics in the model"],
    "view_angle": ["float", "source viewing angle in radians"],
    "distance": ["float", "source distance in Mpc"],
    "m_disk": ["float", "mass of remnant disk in Msun"],
    "T_floor_Ni": ["float", "floor temperature for Ni composition in K"],
    "T_floor_LA": ["float", "floor temperature for LA composition in K"],
    "nuc_fac": ["float", "auxiliary variable [1]"],
    "eps0": ["float", "auxiliary variable [2e18]"],
}

comp_vars_info_dict = {
    "m_ej": ["float", "ejecta mass in Msun"],
    "xi_disk": ["float", "fraction of disk mass expelled"],
    "high_lat_flag": ["bool", "if True the mass is distributed more at high latitudes"],
    "step_angle_mass": [
        "float",
        "angle of the step in the mass distribution in radians",
    ],
    "central_vel": ["float", "central ejecta velocity in units of c"],
    "high_lat_vel": ["float", "high latitude ejecta velocity in units of c"],
    "low_lat_vel": ["float", "low latitude ejecta velocity in units of c"],
    "step_angle_vel": [
        "float",
        "angle of the step in the velocity distribution in radians",
    ],
    "central_op": ["float", "central ejecta opacity in cm^2/g"],
    "high_lat_op": ["float", "high latitude ejecta opacity in cm^2/g"],
    "low_lat_op": ["float", "low latitude ejecta opacity in cm^2/g"],
    "step_angle_op": [
        "float",
        "angle of the step in the opacity distribution in radians",
    ],
    "T_floor": ["float", "floor temperature in K"],
}
