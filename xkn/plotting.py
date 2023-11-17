import os
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.sans-serif': ['Helvetica'],
    'xtick.major.size': 13.0,
    'ytick.major.size': 13.0,
    'xtick.minor.size': 7.0,
    'ytick.minor.size': 7.0,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

from .utils import Mpc2cm, sec2day, day2sec, fourpi, ObserverProjection, init_times, check_dict_variables

def plot_magnitudes(mkn, mkn_vars, ax=None, filename=None, title=None, titlesize=30,
                         hsize=16, wsize=9, labelsize=30, ticksize=26,
                         legendsize=12, legend_geom=[0,0,3,4]):

    mags_d = deepcopy(mkn.mag)
    mags_a = mkn.calc_magnitudes(mkn_vars, measures=False)

    colors = plt.get_cmap('Spectral')(np.linspace(0,1,len(mags_a.keys())))[::-1]

    # correct time axis of data from days to seconds
    for i,lam in enumerate(mags_d):
        mags_d[lam]['time'] = (mags_d[lam]['time'] - mkn.glob_params['t_start_filter']) * day2sec

    if ax is None:
        ax_none = True
        fig, ax = plt.subplots(1, 1, figsize=(hsize,wsize), sharex=True, sharey=True)

    ymin =  np.inf
    ymax = -np.inf
    for i,lam in enumerate(mags_a):
        if not i:
            xmin = mags_a[lam]['time'][0]*sec2day
            xmax = mags_a[lam]['time'][-1]*sec2day
        ymin = min(ymin, np.amin(mags_d[lam]['mag']))
        ymax = max(ymax, np.amax(mags_d[lam]['mag']))
        ax.plot(mags_a[lam]['time']*sec2day, mags_a[lam]['mag'],      c=colors[i])
    for i,lam in enumerate(mags_a):
        ax.plot(mags_d[lam]['time']*sec2day, mags_d[lam]['mag'], 'D', c=colors[i], markeredgecolor='k', label=f'{lam}')

    ymin *= 0.99
    ymax *= 1.01

    ax.grid(which='both',lw=1)
    ax.invert_yaxis()
    ax.set_xlim((xmin,xmax))
    ax.set_ylim((ymax,ymin))
    ax.set_xscale('log')
    ax.set_xlabel('time [days]', fontsize=labelsize)
    ax.set_ylabel('magnitude', fontsize=labelsize)
    ax.tick_params(which='both', labelsize=ticksize)
    ax.legend(bbox_to_anchor=(legend_geom[0],legend_geom[1]), loc=legend_geom[2], ncol=legend_geom[3], fontsize=legendsize)
    if title is not None: ax.set_title(title, fontsize=titlesize)
    if ax_none:
        if filename is not None:
            fig.savefig(filename, facecolor='1', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
