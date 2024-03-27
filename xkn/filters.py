import json
import os
import sys
from copy import deepcopy

import numpy as np
from scipy import interpolate, integrate

from . import utils

###-------------------------------------------------------------------------------------------------
#------Reading data---------------------------------------------------------------------------------
###-------------------------------------------------------------------------------------------------

# UPPER LIMITS: if 'True' it includes every data in the files inside 'filter_data'
#               filter_data_path (also upper limits and uncertain data).
# Note: Upper limits are flagged with '-1' errors, uncertain data
#       (that are presented in Villar table without errors but are not upper limits) with '123456789' errors.

def read_filters(filter_usage, filter_data_path, t_min, t_max, filter_dict='telescopes', filter_dict_path=None, dered_correction=True, R_V=3.1, EBV=0.105, A_V=None, upper_limits=True):
    if   filter_usage == 'measures':   return read_filter_measures(filter_data_path, t_min, t_max, filter_dict, filter_dict_path, dered_correction, R_V=R_V, EBV=EBV, A_V=A_V, upper_limits=upper_limits)
    elif filter_usage == 'properties': return read_filter_properties(filter_dict, filter_dict_path)
    else: sys.exit("Wrong usage for filters. Choose from 'measures' or 'properties'.")

def read_filter_properties(filter_dict='telescopes', filter_dict_path=None):
    # load the filter informations
    if filter_dict_path is None: filter_dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'filter_dictionary', filter_dict + '.json')
    with open(filter_dict_path, 'r') as fi:
        dic_filt = json.load(fi)
    lams = np.sort(np.asarray(list(dic_filt.keys()), dtype=int))
    return {lam : dic_filt[str(lam)] for lam in lams}, lams, {}

def read_filter_measures(filter_data_path, t_min, t_max, filter_dict='telescopes', filter_dict_path=None, dered_correction=True, R_V=3.1, EBV=0.105, A_V=None, upper_limits=True):
    # check if filter_data_path exists
    if not os.path.exists(filter_data_path): raise FileNotFoundError(f"The path to the filter data {filter_data_path} does not exist!")

    # load the filter information
    dic_filt, lams, _ = read_filter_properties(filter_dict=filter_dict, filter_dict_path=filter_dict_path)

    # load the measured magnitudes
    measures = {}
    for lam in lams:
        times_tot  = np.asarray([])
        mags_tot   = np.asarray([])
        sigmas_tot = np.asarray([])

        for fname, mag_type in zip(dic_filt[lam]['filename'], dic_filt[lam]['type']):
            # time, mag, sigma
            times, mags, sigmas = np.loadtxt(os.path.join(filter_data_path, fname), unpack = True, ndmin=1)
            t_mask  = np.logical_and(times > t_min, times < t_max)
            times   = np.atleast_1d(times[t_mask])
            mags    = np.atleast_1d(mags[t_mask])
            sigmas  = np.atleast_1d(sigmas[t_mask])

            # if the magnitude type is vega then convert the magnitudes to AB
            if mag_type == 'vega': mags += get_corr_vega_to_AB(lam)

            if upper_limits:
                times_tot  = np.append(times_tot,  times)
                mags_tot   = np.append(mags_tot,   mags)
                sigmas_tot = np.append(sigmas_tot, sigmas)
            else:
                mask_upper = sigmas > 0
                times_tot  = np.append(times_tot,  times[mask_upper])
                mags_tot   = np.append(mags_tot,   mags[mask_upper])
                sigmas_tot = np.append(sigmas_tot, sigmas[mask_upper])

        # if dered_correction is 'True' it corrects the magnitudes [M_dered = M - correction]
        if dered_correction: mags_tot -= dered_CCM(np.asarray([lam]), R_V=R_V, EBV=EBV, A_V=A_V)

        if len(times_tot):
            measures[lam] = {
                'time'  : times_tot,
                'mag'   : mags_tot,
                'sigma' : sigmas_tot,
                'name'  : '_'.join(dic_filt[lam]["name"]),
                }
        else: del dic_filt[lam]

    assert list(dic_filt.keys()) == list(measures.keys())

    return dic_filt, np.sort(np.asarray(list(dic_filt.keys()), dtype=int)), measures

# MAGNITUDE CORRECTION FOR VEGA TO AB
def get_corr_vega_to_AB(lam):

    # table taken from https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
    vega_to_AB_corr = {
        # band  mAB - mVega  lam_eff (nm)
        'u'     : { 'corr' :  0.91,   'lam' :  354.6, }, 
        'U'     : { 'corr' :  0.79,   'lam' :  357.1, },
        'B'     : { 'corr' : -0.09,   'lam' :  434.4, },
        'g'     : { 'corr' : -0.08,   'lam' :  467.0, },
        'V'     : { 'corr' :  0.02,   'lam' :  545.6, },
        'r'     : { 'corr' :  0.16,   'lam' :  615.6, },
        'R'     : { 'corr' :  0.21,   'lam' :  644.2, },
        'i'     : { 'corr' :  0.37,   'lam' :  747.2, },
        'I'     : { 'corr' :  0.45,   'lam' :  799.4, },
        'z'     : { 'corr' :  0.54,   'lam' :  891.7, },
        'Y'     : { 'corr' :  0.634,  'lam' : 1030.5, },
        'J'     : { 'corr' :  0.91,   'lam' : 1235.5, },
        'H'     : { 'corr' :  1.39,   'lam' : 1645.8, },
        'Ks'    : { 'corr' :  1.85,   'lam' : 2160.3, },
        }
    
    corrs = np.array([ vega_to_AB_corr[band]['corr'] for band in vega_to_AB_corr ])
    lams  = np.array([ vega_to_AB_corr[band]['lam'] for band in vega_to_AB_corr ])

    return corrs[np.digitize(lam, lams[:-1] + 0.5 * (lams[1:] - lams[:-1]))]

# MAGNITUDE CORRECTION FOR DEREDDENING
# Input wavelength in nanometers!!!
def dered_CCM(wave, R_V=3.1, EBV=0.105, A_V=None):
    '''
    - Input:
                wave 1D array (Nanometers)
                EBV: E(B-V) (default 0.105)
                R_V: Reddening coefficient to use (default 3.1)
    - Output:
                A_lambda correction according to the CCM89 Law.
    '''
    x = 1000./ wave  #Convert to inverse microns
    a = np.zeros_like(x)
    b = np.zeros_like(x)

    ## Infrared ##
    mask = (x > 0.3) & (x < 1.1)
    if np.any(mask):
        a[mask] =  0.574 * x[mask]**(1.61)
        b[mask] = -0.527 * x[mask]**(1.61)

    ## Optical/NIR ##
    mask = (x >= 1.1) & (x < 3.3)
    if np.any(mask):
        xxx = x[mask] - 1.82
        # c1 = [ 1. , 0.17699, -0.50447, -0.02427,  0.72085, #Original
        #        0.01979, -0.77530,  0.32999 ]               #coefficients
        # c2 = [ 0.,  1.41338,  2.28305,  1.07233, -5.38434, #from CCM89
        #       -0.62251,  5.30260, -2.09002 ]
        c1 = [ 1. , 0.104,   -0.609,    0.701,  1.137,     #New coefficients
              -1.718,   -0.827,    1.647, -0.505 ]         #from O'Donnell
        c2 = [ 0.,  1.952,    2.908,   -3.989, -7.985,     #(1994)
               11.102,    5.491,  -10.805,  3.347 ]
        a[mask] = np.poly1d(c1[::-1])(xxx)
        b[mask] = np.poly1d(c2[::-1])(xxx)

    ## Mid-UV ##
    mask = (x >= 3.3) & (x < 8.0)
    if np.any(mask):
        F_a = np.zeros_like(x[mask])
        F_b = np.zeros_like(x[mask])
        mask1 = x[mask] > 5.9
        if np.any(mask1):
            xxx = x[mask][mask1] - 5.9
            F_a[mask1] = -0.04473 * xxx**2 - 0.009779 * xxx**3
        a[mask] = 1.752 - 0.316*x[mask] - (0.104 / ( (x[mask]-4.67)**2 + 0.341 )) + F_a
        b[mask] = -3.090 + 1.825*x[mask] + (1.206 / ( (x[mask]-4.62)**2 + 0.263 )) + F_b

    ## Far-UV ##
    mask = (x >= 8.0) & (x < 11.0)
    if np.any(mask):
        xxx = x[mask] - 8.0
        c1 = [ -1.073, -0.628,  0.137, -0.070 ]
        c2 = [ 13.670,  4.257, -0.420,  0.374 ]
        a[mask] = np.poly1d(c1[::-1])(xxx)
        b[mask] = np.poly1d(c2[::-1])(xxx)

    #Now compute extinction correction
    if A_V is None: A_V = R_V * EBV
    A_lambda = A_V * (a + b/R_V)
    return A_lambda

###-------------------------------------------------------------------------------------------------
#------isotropized luminosity-----------------------------------------------------------------------
###-------------------------------------------------------------------------------------------------

def calc_lum_iso_from_bol(lum_bol, flux_factors, omegas):
    lumbol = lum_bol * utils.fourpi / omegas[:,None]
    return np.sum(
            np.array([lum * f for lum,f in zip(lumbol,       flux_factors[:len(flux_factors)//2])]) +
            np.array([lum * f for lum,f in zip(lumbol[::-1], flux_factors[len(flux_factors)//2:])]), axis=0
            ) / np.pi

def calc_lum_iso_from_mags(mags, dic_filt, distance):
    lambdas = [100. * dic_filt[lam]['lambda'] for lam in mags]
    lum_lams = np.array([ 10**( (mags[lam]['mag'] + 48.6) / -2.5 ) * utils.fourpi * (distance * utils.Mpc2cm)**2 / (100. * dic_filt[lam]['lambda'])**2 * utils.c for lam in mags ])
    return integrate.simps(lum_lams, np.array(lambdas), axis=0)

###-------------------------------------------------------------------------------------------------
#------magnitude filter calculation-----------------------------------------------------------------
###-------------------------------------------------------------------------------------------------

def planckian_help(val):
    return (val > 10e-15) * (np.exp(val) - 1) + (val <= 10e-15) * val

def planckian(nu, T_plk):
    return 2. * utils.h * nu**3 / utils.c2 / planckian_help(utils.h * nu / utils.kB / T_plk)

def calc_fnu(flux_factors, lambda_meters, distance, redshift, radius_photo, T_photo=None, lum_shells=None, T_shells=None, omegas=None):
    fnu_cont = np.zeros_like(radius_photo)
    if T_photo is not None:
        fnu_cont += np.where(radius_photo > 0, radius_photo**2 * planckian(utils.c / (100. * lambda_meters / (1. + redshift)), T_photo), 0)
    if lum_shells is not None and T_shells is not None and omegas is not None:
        mask_zeroes = T_shells != 0
        fnu_cont_tmp = np.zeros_like(T_shells)
        fnu_cont_tmp[mask_zeroes] = (lum_shells / (omegas[None,:,None,None] * utils.sigma_SB * T_shells**4) * planckian(utils.c / (100. * lambda_meters / (1. + redshift)), T_shells))[mask_zeroes]
        fnu_cont += fnu_cont_tmp.sum(axis=-1).sum(axis=0)
    return (fnu_cont.T * flux_factors[:len(flux_factors)//2][None,:] + fnu_cont[::-1].T * flux_factors[len(flux_factors)//2:][None,:]).sum(axis=-1) * (1. + redshift) / distance**2

def m_filter(flux_factors, lambda_meters, distance, redshift, radius_photo, T_photo=None, lum_shells=None, T_shells=None, omegas=None):
    return -2.5 * np.log10(calc_fnu(flux_factors, lambda_meters, distance, redshift, radius_photo, T_photo=T_photo, lum_shells=lum_shells, T_shells=T_shells, omegas=omegas)) - 48.6

###-------------------------------------------------------------------------------------------------
#------magnitudes-----------------------------------------------------------------------------------
###-------------------------------------------------------------------------------------------------

def calc_magnitudes(flux_factors, times, lams, dic_filt, distance, redshift, radius_photo, T_photo=None, lum_shells=None, T_shells=None, omegas=None, measures=False, mag=None, t_start_filter=None, **kwargs):

    if measures:
        # calculate the magnitudes at the times specified in mag
        return { lam : { 'time' : (mag[lam]['time'] - t_start_filter) * utils.day2sec,
                         'mag'  : np.interp((mag[lam]['time'] - t_start_filter) * utils.day2sec, times,
                                             m_filter(flux_factors, dic_filt[lam]['lambda'], distance, redshift, radius_photo,
                                                      T_photo=T_photo, lum_shells=lum_shells, T_shells=T_shells, omegas=omegas)) } for lam in lams }

    else:
        # calculate the magnitudes at the specified times array
        return { lam :  { 'time' : times,
                          'mag'  : m_filter(flux_factors, dic_filt[lam]['lambda'], distance, redshift, radius_photo,
                                            T_photo=T_photo, lum_shells=lum_shells, T_shells=T_shells, omegas=omegas) } for lam in lams }

###-------------------------------------------------------------------------------------------------
#------residuals------------------------------------------------------------------------------------
###-------------------------------------------------------------------------------------------------

def prep_sigma(sigma, mag_diff=-1):
    # sigma >  0 corresponds to magnitude data below threshold: the sigma are good and not touched
    # sigma <= 0 corresponds to magnitude data above threshold:
    #       with mag_diff = mag_model - mag_data, thus
    #       large_sigma_mask: mag_diff >= 0 corresponds to the model being above threshold as well,
    #                         sigma cannot discriminate the model and a large sigma value is returned
    #       abs_sigma_mask:   mag_diff <  0 corresponds to the model being below threshold,
    #                         sigma can discriminate the model and the absolute value of sigma is returned
    large_sigma_mask        = np.logical_and(sigma <= 0, mag_diff >= 0)
    abs_sigma_mask          = np.logical_and(sigma <= 0, mag_diff <  0)
    Sigma                   = deepcopy(sigma)
    Sigma[large_sigma_mask] = 1e6
    Sigma[abs_sigma_mask]   = np.abs(sigma[abs_sigma_mask])
    return Sigma

def calc_residuals(flux_factors, times, lams, dic_filt, distance, redshift, mag, t_start_filter, radius_photo, T_photo=None, lum_shells=None, T_shells=None, omegas=None, sigma_sys=0, **kwargs):

    # calculate the difference in magnitudes at the times specified in mag
    mag_diffs = { lam : np.interp((mag[lam]['time'] - t_start_filter) * utils.day2sec, times,
                                  m_filter(flux_factors, dic_filt[lam]['lambda'], distance, redshift, radius_photo,
                                           T_photo=T_photo, lum_shells=lum_shells, T_shells=T_shells, omegas=omegas))
                        - mag[lam]['mag'] for lam in lams }

    return { lam : mag_diffs[lam] / np.sqrt( prep_sigma(mag[lam]['sigma'], mag_diff=mag_diffs[lam])**2 + sigma_sys**2 ) for lam in lams }

###-------------------------------------------------------------------------------------------------
#------Filter clean-up routines---------------------------------------------------------------------------
###-------------------------------------------------------------------------------------------------

def limit_lams(dic_filt, lams, mag, lam_list=None, lam_min=None, lam_max=None):
    dic_filt = deepcopy(dic_filt)
    lams     = deepcopy(lams)
    mag      = deepcopy(mag)

    if lam_list is None and lam_min is None and lam_max is None: return dic_filt, lams, mag
    else:
        if lam_min is None:  lam_min  = 0
        if lam_max is None:  lam_max  = np.inf
        if lam_list is None: lam_list = lams

        for lam in lams:
            if lam not in lam_list or lam < lam_min or lam > lam_max:
                del dic_filt[lam]
                if mag: del mag[lam]

        return dic_filt, np.sort(np.asarray(list(dic_filt.keys()), dtype=int)), mag

def limit_mags(dic_filt, lams, mag, mag_min=None, mag_max=None):
    dic_filt = deepcopy(dic_filt)
    lams     = deepcopy(lams)
    mag      = deepcopy(mag)

    if not mag or (mag_min is None and mag_max is None): return dic_filt, lams, mag
    elif mag_min is None: mag_min = -np.inf
    elif mag_max is None: mag_max =  np.inf

    for lam in lams:
        ids = np.nonzero(np.logical_and(mag[lam]['mag'] >= mag_min, mag[lam]['mag'] <= mag_max))[0]
        if not len(ids):
            del dic_filt[lam]
            del mag[lam]
        else:
            mag[lam]['mag']   = mag[lam]['mag'][ids]
            mag[lam]['time']  = mag[lam]['time'][ids]
            mag[lam]['sigma'] = mag[lam]['sigma'][ids]

    return dic_filt, np.sort(np.asarray(list(dic_filt.keys()), dtype=int)), mag

###-------------------------------------------------------------------------------------------------
#------injection preparation routines---------------------------------------------------------------
###-------------------------------------------------------------------------------------------------

def prep_inj_mag(new_mags, mag=None, measures=False, t_start_filter=None, seed=None, sigma_min=None, sigma_max=None):
    if mag is None: injection_mag = {}
    else:           injection_mag = deepcopy(mag)

    if mag is None or not mag:
        assert t_start_filter is not None and sigma_min is not None and sigma_max is not None
    elif not measures:
        assert t_start_filter is not None

    if not injection_mag:
        rng = np.random.default_rng(seed)
        for lam in new_mags:
            injection_mag[lam] = {
                'mag'   : new_mags[lam]['mag'],
                'time'  : new_mags[lam]['time'] * utils.sec2day + t_start_filter,
                'sigma' : rng.uniform(sigma_min, sigma_max, len(new_mags[lam]['time']))
                }

    elif measures:
        for lam in injection_mag:
            if len(injection_mag[lam]['mag']):
                injection_mag[lam]['mag']   = new_mags[lam]['mag']

    else:
        rng = np.random.default_rng(seed)
        for lam in injection_mag:
            if len(injection_mag[lam]['mag']):
                if sigma_min is None: sigma_min = np.amin(injection_mag[lam]['sigma'])
                if sigma_max is None: sigma_max = np.amax(injection_mag[lam]['sigma'])
                injection_mag[lam]['mag']   = new_mags[lam]['mag']
                injection_mag[lam]['time']  = new_mags[lam]['time'] / utils.day2sec + t_start_filter
                injection_mag[lam]['sigma'] = rng.uniform(sigma_min, sigma_max, len(new_mags[lam]['time']))

    return injection_mag
