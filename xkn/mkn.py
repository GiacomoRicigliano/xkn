import sys
import logging
from copy import deepcopy
from warnings import filterwarnings
filterwarnings('ignore')

import numpy as np

from . import filters as flt
from .angular_distribution import AngularDistribution
from .ejecta import Ejecta
from .plotting import plot_magnitudes
from .utils import Mpc2cm, sec2day, day2sec, fourpi, ObserverProjection, init_times, time_safe, check_dict_variables, Redshift

#####
# MKN class to facilitate the Macro-KiloNova model functionality
#####
class MKN:

    #####
    # initialization of the class object
    #####
    def __init__(self, shell_params, glob_params, inj_dict=None, log_name='MKN', log_level='INFO'):
        self.set_logger(name=log_name, level=log_level)
        self.set_ejecta(list(shell_params.keys()), shell_params)
        self.set_glob_params(glob_params)
        self.gen_inj_data(inj_dict)
        self.logger.info('--- MKN object fully initialized. ---')

    #####
    # setters used during the initialization
    #####
    def set_logger(self, name='MKN', level='INFO', stdout=True, logfile=None):
        logging.basicConfig(format = '%(asctime)s - %(name)s - %(levelname)s : %(message)s')
        if stdout: logging.basicConfig(stream = sys.stdout)
        if logfile is not None: logging.basicConfig(filename = logfile, filemode = 'w')
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

    def set_ejecta(self, shell_names, shell_params):
        self.shell_names  = shell_names
        self.shell_params = shell_params
        self.ejecta       = Ejecta(shell_names, shell_params)
        self.logger.info('Initialized ejecta.')

    def set_glob_params(self, glob_params):
        self.glob_params = glob_params
        self.set_flux_factor_func()
        self.set_angles_omegas()
        self.set_filter_data()
        self.set_redshift()
        self.set_times()

    def set_flux_factor_func(self):
        check_dict_variables(dic=(self.glob_params, ['slices_dist', 'slices_num']), logger=self.logger, strict=True, label='set_flux_factor_func')
        self.flux_factor_func = ObserverProjection(self.glob_params['slices_num'], self.glob_params['slices_dist'])
        self.logger.info('Initialized flux_factor_func.')
        self.logger.debug(f"   settings: slices_dist={self.glob_params['slices_dist']}, slices_num={self.glob_params['slices_num']}.")

    def set_angles_omegas(self):
        check_dict_variables(dic=(self.glob_params, ['slices_dist', 'slices_num', 'omega_frac']), logger=self.logger, strict=True, label='set_angles_omegas')
        self.angles, self.omegas = AngularDistribution(self.glob_params['slices_dist'])(self.glob_params['slices_num'] / 2, self.glob_params['omega_frac'])
        self.logger.info('Initialized angular distribution.')
        self.logger.debug(f"   settings: slices_dist={self.glob_params['slices_dist']}, slices_num={self.glob_params['slices_num']}, omega_frac={self.glob_params['omega_frac']}.")

    def set_filter_data(self):
        if check_dict_variables(dic=(self.glob_params, ['filter_usage', 't_min', 't_max', 't_start_filter']), logger=self.logger, strict=False, label='set_filter_data'):
            self.dic_filt_full, self.lams_full, self.mag_full = flt.read_filters(self.glob_params['filter_usage'], self.glob_params['filter_data_path'],
                                                                                 self.glob_params['t_min'] * sec2day + self.glob_params['t_start_filter'],
                                                                                 self.glob_params['t_max'] * sec2day + self.glob_params['t_start_filter'],
                                                                                 filter_dict=self.glob_params['filter_dictionary'],
                                                                                 filter_dict_path=self.glob_params['filter_dictionary_path'],
                                                                                 dered_correction=self.glob_params['dered_correction'],
                                                                                 R_V=self.glob_params['R_V'], EBV=self.glob_params['EBV'], A_V=self.glob_params['A_V'],
                                                                                 upper_limits=self.glob_params['upper_limits'])

            self.dic_filt, self.lams, self.mag =  flt.limit_mags(*flt.limit_lams(self.dic_filt_full, self.lams_full, self.mag_full,
                                                                                 lam_list=self.glob_params['lam_list'], lam_min=self.glob_params['lam_min'], lam_max=self.glob_params['lam_max']),
                                                                 mag_min=self.glob_params['mag_min'], mag_max=self.glob_params['mag_max'])
            self.logger.info(f'Initialized filter data from local data.')
            self.logger.debug(f"   settings: path={self.glob_params['filter_data_path']}.")
        else:
            self.dic_filt_full, self.lams_full, self.mag_full, self.dic_filt, self.lams, self.mag = None, None, None, None, None, None
            self.logger.info('Did not initialize filter data from local data.')

    def set_redshift(self):
        check_dict_variables(dic=(self.glob_params, ['cosmology']), logger=self.logger, strict=True, label='set_redshift', allow_none=True)
        self.redshift = Redshift(self.glob_params['cosmology'])
        self.logger.info('Initialized redshift')
        self.logger.debug(f"   settings: cosmology={self.glob_params['cosmology']}.")

    def set_times(self):
        if check_dict_variables(dic=(self.glob_params, ['t_scale', 't_min', 't_max', 't_num', 't_start_filter']), logger=None) \
        or check_dict_variables(dic=(self.glob_params, ['t_scale', 't_start_filter']), var=([self.mag], ['mag']), logger=None):
            if check_dict_variables(dic=(self.glob_params, ['t_toll']), logger=None): toll = self.glob_params['t_toll']
            else:                                                                     toll = 0.1
            self.times = init_times(self.glob_params['t_scale'], self.glob_params['t_min'], self.glob_params['t_max'],
                                    self.glob_params['t_num'], self.glob_params['t_start_filter'], self.mag, toll)
            self.logger.info('Initialized times.')
            self.logger.debug(f"   settings: t_scale={self.glob_params['t_scale']}, t_min={self.glob_params['t_min']}, t_max={self.glob_params['t_max']}, t_num={self.glob_params['t_num']}, t_start_filter={self.glob_params['t_start_filter']}, t_toll={toll}")
        else:
            check_dict_variables(dic=(self.glob_params, ['t_scale', 't_min', 't_max', 't_num', 't_start_filter']), var=([self.mag], ['mag']),
                logger=self.logger, strict=True, label='set_times')

    #####
    # injection mag generation
    #####
    def prep_inj_mag(self, new_mags, measures=False, seed=None, sigma_min=None, sigma_max=None):
        mag = flt.prep_inj_mag(new_mags, mag=self.mag, measures=measures, t_start_filter=self.glob_params['t_start_filter'], seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
        if not self.mag:
            for lam in mag:
                mag[lam]['name'] = self.dic_filt[lam]['name']
        return mag

    def gen_inj_data(self, inj_dict):
        self.inj_dict = inj_dict
        if inj_dict is None: return
        else:
            mkn      = MKN(inj_dict['shell_params'], inj_dict['glob_params'], log_name='INJ-MKN', log_level='WARNING')
            measures = inj_dict['glob_params']['t_scale'] == 'measures'
            mag      = mkn.prep_inj_mag(mkn.calc_magnitudes(inj_dict['mkn_vars'], measures=measures),
                                        measures=measures, seed=inj_dict['seed'],
                                        sigma_min=inj_dict['sigma_min'],
                                        sigma_max=inj_dict['sigma_max'])
            self.dic_filt, self.lams, self.mag = flt.limit_mags(*flt.limit_lams(mkn.dic_filt, mkn.lams, mag,
                                                                                lam_list=inj_dict['glob_params']['lam_list'],
                                                                                lam_min=inj_dict['glob_params']['lam_min'],
                                                                                lam_max=inj_dict['glob_params']['lam_max']),
                                                                mag_min=inj_dict['glob_params']['mag_min'], mag_max=inj_dict['glob_params']['mag_max'])
        self.logger.info('Initialized injection.')


    #####
    # from obs frame to source frame time
    #####
    # Compute source time
    def time_source(self, mkn_vars):
        return time_safe(self.times / (1. + self.redshift(mkn_vars['glob']['distance'])), self.glob_params['t_0'])

    # Truncate observer time for consistency with source time array length
    def time_observer(self, mkn_vars):
        return time_safe(self.times, self.glob_params['t_0'] * (1. + self.redshift(mkn_vars['glob']['distance'])))

    #####
    # model quantities calculation: flux_factors, lightcurve variables, and magnitudes
    #####
    def calc_flux_factors(self, mkn_vars):
        if mkn_vars['glob']['view_angle'] > np.pi / 2: return self.flux_factor_func(np.degrees(np.pi - mkn_vars['glob']['view_angle']))
        else:                                          return self.flux_factor_func(np.degrees(mkn_vars['glob']['view_angle']))

    def calc_lightcurve_vars(self, mkn_vars):
        return self.ejecta.calc_lightcurve_vars(self.angles, self.omegas, self.time_source(mkn_vars), mkn_vars, mkn_vars['glob'], self.glob_params, logger=self.logger)

    def calc_magnitudes(self, mkn_vars, measures=False):
        self.calc_lightcurve_vars(mkn_vars)
        return flt.calc_magnitudes(self.calc_flux_factors(mkn_vars), self.time_observer(mkn_vars), self.lams, self.dic_filt, mkn_vars['glob']['distance'] * Mpc2cm,
                                   self.redshift(mkn_vars['glob']['distance']), self.ejecta.radius_photo, T_photo=self.ejecta.T_photo, lum_shells=self.ejecta.lum_shells,
                                   T_shells=self.ejecta.T_shells, omegas=self.omegas, measures=measures, mag=self.mag,
                                   t_start_filter=self.glob_params['t_start_filter'])

    #####
    # residuals and log_like calculation
    #####
    def calc_residuals(self, mkn_vars):
        self.calc_lightcurve_vars(mkn_vars)
        return flt.calc_residuals(self.calc_flux_factors(mkn_vars), self.time_observer(mkn_vars), self.lams, self.dic_filt, mkn_vars['glob']['distance'] * Mpc2cm,
                                  self.redshift(mkn_vars['glob']['distance']), self.mag, self.glob_params['t_start_filter'], self.ejecta.radius_photo, T_photo=self.ejecta.T_photo,
                                  lum_shells=self.ejecta.lum_shells, T_shells=self.ejecta.T_shells, omegas=self.omegas, sigma_sys=mkn_vars['glob']['sigma_sys'])

    def calc_log_like(self, mkn_vars):
        return -0.5 * sum([sum(residual**2) for residual in self.calc_residuals(mkn_vars).values()]) + self.calc_log_like_normalization(mkn_vars)

    def calc_log_like_normalization(self, mkn_vars):
        return -0.5 * len(self.lams) * sum([sum(np.log(2 * np.pi * (self.mag[lam]['sigma']**2 + mkn_vars['glob']['sigma_sys']**2))) for lam in self.lams ])

    #####
    # isotropized luminosity calculation for model consistency check
    #####
    def calc_lum_iso(self, mkn_vars, t_scale=None, t_num=None, t_min=None, t_max=None):
        return calc_lum_iso_fake_filters(self.shell_params, self.glob_params, mkn_vars,
                                         t_scale=t_scale, t_num=t_num, t_min=t_min, t_max=t_max)

    #####
    # plotting
    #####
    def plot_magnitudes(self, mkn_vars, ax=None, filename=None, title=None, titlesize=30,
                         hsize=16, wsize=9, labelsize=30, ticksize=26, legendsize=12, legend_geom=[0,0,3,4]):
        plot_magnitudes(self, mkn_vars, ax=ax, filename=filename, title=title, titlesize=titlesize,
                              hsize=hsize, wsize=wsize, labelsize=labelsize, ticksize=ticksize, legendsize=legendsize, legend_geom=legend_geom)


#####
# Auxiliary functions for injection generation
#####
def gen_inj_dict(shell_params, glob_params, mkn_vars, seed=None, sigma_min=None, sigma_max=None):
 return {
    'shell_params'  : deepcopy(shell_params),
    'glob_params'   : deepcopy(glob_params),
    'mkn_vars'      : deepcopy(mkn_vars),
    'seed'          : seed,
    'sigma_min'     : sigma_min,
    'sigma_max'     : sigma_max,
    }

#####
# Auxiliary functions for lum_iso calculation
#####
def calc_lum_iso_fake_filters(shell_params, glob_params, mkn_vars, t_scale=None, t_num=None, t_min=None, t_max=None):
    glob_params_lum_iso                      = deepcopy(glob_params)
    glob_params_lum_iso['filter_dictionary'] = 'iso_calc'

    if t_scale is not None: glob_params_lum_iso['t_scale'] = t_scale
    if t_num   is not None: glob_params_lum_iso['t_num']   = t_num
    if t_min   is not None: glob_params_lum_iso['t_min']   = t_min
    if t_max   is not None: glob_params_lum_iso['t_max']   = t_max

    mkn = MKN(shell_params, glob_params_lum_iso, log_name='LUM-ISO-MKN', log_level='WARNING')
    if glob_params_lum_iso['t_scale'] not in ['lin', 'log']:
        mkn.logger.error('lum_iso calculation has to be done with t_usage = lin or log! ... Exiting.')
        sys.exit()
    mkn.calc_lightcurve_vars(mkn_vars)
    return mkn.time_observer(mkn_vars), mkn.time_source(mkn_vars), \
           flt.calc_lum_iso_from_bol(mkn.ejecta.lum_bol, mkn.calc_flux_factors(mkn_vars), mkn.omegas), \
           flt.calc_lum_iso_from_mags(mkn.calc_magnitudes(mkn_vars, measures=False), mkn.dic_filt, mkn_vars['glob']['distance'])
