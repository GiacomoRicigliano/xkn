import sys

import numpy as np
import scipy.optimize

from .shell import Shell
from .utils import T_eff_calc, calc_Tfloor, fourpi, sigma_SB, fourpisigma_SB, check_dict_variables, Mv_Woll
from . import utils

class Ejecta(object):

    def __init__(self, shell_names, shell_params, *args, **kwargs):
        self.ncomponents = len(shell_names)
        self.components  = [Shell(n, shell_params[n], **kwargs) for n in shell_names]


    def generate_diff_lums(self, angles, times, shell_vars, glob_vars, glob_params, **kwargs):
        for c in self.components:
            c.generate_diff_lums(angles, times, shell_vars[c.name], glob_vars, glob_params, **kwargs)


    def calc_lightcurve_vars(self, angles, omegas, times, shell_vars, glob_vars, glob_params, **kwargs):
        if check_dict_variables(dic=(kwargs, ['diff_lums']), label='calc_lightcurve_vars'): diff_lums = kwargs['diff_lums']
        else:
            if glob_params['lc_model'] == 'ricigliano_lippold': self.generate_diff_lums(angles, times, shell_vars, glob_vars, glob_params, **kwargs)
            diff_lums = self.ncomponents * [None]

        self.radius_photo = np.zeros((self.ncomponents, len(angles), len(times)))
        self.lum_bol_raw  = np.zeros_like(self.radius_photo)

        for ic,c in enumerate(self.components):
            self.radius_photo[ic], self.lum_bol_raw[ic] = \
                c.expansion_angular_distribution(angles, omegas, times, shell_vars[c.name], glob_vars, glob_params, diff_lums=diff_lums[ic], **kwargs)

        # select the photospheric radius as the maximum between the different single photospheric radii
        self.radius_photo = np.amax(self.radius_photo, axis=0)

        if 'thin_shells' in glob_params and glob_params['thin_shells']:
            self.v_shells    = np.zeros((self.ncomponents, len(angles), glob_params['n_thin']))
            self.lum_shells  = np.zeros((self.ncomponents, len(angles), len(times), glob_params['n_thin'] - 1))
            self.vel_woll    = np.zeros((self.ncomponents, len(angles)))
            self.mass_scaled = np.zeros((self.ncomponents, len(angles)))
            self.ye          = np.zeros((self.ncomponents, len(angles)))

            for ic,c in enumerate(self.components):
                self.v_shells[ic], self.lum_shells[ic], self.vel_woll[ic], self.mass_scaled[ic], self.ye[ic] = \
                    c.expansion_angular_distribution_thin_layers(angles, omegas, times, shell_vars[c.name], glob_vars, glob_params, **kwargs)

            return self.calc_lightcurve_vars_thin(angles, omegas, times, shell_vars, glob_vars, glob_params, **kwargs)

        else:
            # define the total bolometric luminosity as the sum of the different single luminosities
            self.lum_bol_raw = np.sum(self.lum_bol_raw, axis=0)
            # define bolometric luminosity consistent with magnitudes in case radius_photo drops to zero
            self.lum_photo   = np.where(self.radius_photo > 0, self.lum_bol_raw, 0)
            self.lum_bol     = np.copy(self.lum_photo)
            # compute the effective BB temperature based on the photospheric radius and luminosity
            self.T_photo     = T_eff_calc(omegas, self.radius_photo, self.lum_photo)
            # no thin shell contribution
            self.lum_shells  = None
            self.T_shells    = None

            return self.lum_bol, self.lum_photo, self.radius_photo, self.T_photo, self.lum_shells, self.T_shells, self.lum_bol_raw


    def calc_lightcurve_vars_thin(self, angles, omegas, times, shell_vars, glob_vars, glob_params, **kwargs):

        #THICK REGIME

        # calculating diffusive luminosity and photosphere temperature
        mass_scaled_thin = utils.mass_scaled_thin(self.radius_photo[None], times[None,None], self.vel_woll[:,:,None], self.mass_scaled[:,:,None])
        self.lum_photo   = (self.lum_bol_raw * (1 - mass_scaled_thin / self.mass_scaled.sum(axis=0)[:,None])).sum(axis=0)
        self.T_photo     = T_eff_calc(omegas, self.radius_photo, self.lum_photo)

        # correcting photosphere radius, temperature and diffusive luminosity based on floor temperatures
        T_floors = np.array([ shell_vars[c.name]['T_floor'] for c in self.components ])
        if None in T_floors: T_floors = calc_Tfloor('ye',(self.ye * self.mass_scaled).sum(axis=0) / self.mass_scaled.sum(axis=0),
                                                     glob_vars['T_floor_LA'], glob_vars['T_floor_Ni'])
        else:                T_floors = (T_floors[:,None] * self.mass_scaled).sum(axis=0) / self.mass_scaled.sum(axis=0)

        mask_floors                   = ((self.T_photo < T_floors[:,None]) & (self.T_photo > 0))
        self.T_photo[mask_floors]     = (mask_floors * T_floors[:,None])[mask_floors]

        self.radius_photo[mask_floors] = scipy.optimize.root(utils.R_poly, ((self.lum_photo / omegas[:,None] / sigma_SB / self.T_photo**4)**.5)[mask_floors],
                                                             args = ((self.lum_bol_raw.sum(axis=0) * fourpi / omegas[:,None])[mask_floors],
                                                                     (mask_floors * times[None])[mask_floors],
                                                                     (mask_floors[None] * self.vel_woll[:,:,None])[:,mask_floors],
                                                                     (mask_floors[None] * self.mass_scaled[:,:,None])[:,mask_floors],
                                                                     (mask_floors * T_floors[:,None])[mask_floors]), method='hybr').x

        mass_scaled_thin = utils.mass_scaled_thin(self.radius_photo[None], times[None,None], self.vel_woll[:,:,None], self.mass_scaled[:,:,None])
        self.lum_photo      = (self.lum_bol_raw * (1 - mass_scaled_thin / self.mass_scaled.sum(axis=0)[:,None])).sum(axis=0)

        #THIN REGIME

        # calculating thin shells luminosity
        vel_photo        = self.radius_photo / times[None]
        mass_scaled_thin = Mv_Woll(self.mass_scaled[:,:,None], vel_photo[None] / self.vel_woll[:,:,None])
        v_shells         = self.v_shells[:,:,None] * np.ones((self.ncomponents, len(angles), len(times), glob_params['n_thin']))

        mask_photo       = v_shells < vel_photo[None,:,:,None]
        mask_photo_edge  = mask_photo[:,:,:,:-1] & np.logical_not(mask_photo[:,:,:,1:])
        mask_photo       = mask_photo[:,:,:,1:]

        self.lum_shells[mask_photo]      = 0
        self.lum_shells[mask_photo_edge] = (self.lum_shells * ( mass_scaled_thin[:,:,:,None] -
              Mv_Woll(self.mass_scaled[:,:,None,None], v_shells[:,:,:,1:]  / self.vel_woll[:,:,None,None]) ) /
            ( Mv_Woll(self.mass_scaled[:,:,None,None], v_shells[:,:,:,:-1] / self.vel_woll[:,:,None,None]) -
              Mv_Woll(self.mass_scaled[:,:,None,None], v_shells[:,:,:,1:]  / self.vel_woll[:,:,None,None]) ) )[mask_photo_edge]
            # subtracting the contribution of the part of matter inside the photosphere

        # bolometric luminosity obtained from the diffusive and thin shells luminosity
        self.lum_bol     = self.lum_photo + self.lum_shells.sum(axis=(0,3))

        # calculating thin shells temperature
        self.T_shells    = np.zeros_like(self.lum_shells)
        T_floors         = self.T_shells + T_floors[None,:,None,None]

        mask_photo       = np.logical_not(mask_photo)

        self.T_shells[mask_photo] = (self.T_photo[None,:,:,None] *
                                    (1 - (v_shells[:,:,:,:-1] / np.amax(self.vel_woll,axis=0)[None,:,None,None])**2) /
                                    (1 - (vel_photo[None,:,:,None] / np.amax(self.vel_woll,axis=0)[None,:,None,None])**2))[mask_photo]

        mask_floors                             = self.T_shells < T_floors
        self.T_shells[mask_photo & mask_floors] = T_floors[mask_photo & mask_floors]

        return self.lum_bol, self.lum_photo, self.radius_photo, self.T_photo, self.lum_shells, self.T_shells, self.lum_bol_raw
        