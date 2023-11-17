########################
xkn - Kilonova Framework
########################

# Motivations:

- Generate bolometric and broad-band light-curves 
- Compare to AT2017gfo or other data 

###############################################

# WARNING # This is an ALPHA version!!! #######

###############################################

This version is not optimized for MC, there is no documentation yet and some functionalities are incomplete.

It is only intended to be explored in order to acquire familiarity and start using the basic functionalities.

The development is in progress, here the predicted timeline:

- End of 2023:          Extensive documentation
- First months of 2024: First beta release

Please, check the repo https://github.com/GiacomoRicigliano/xkn regularly for updates.

###############################################

# Basic usage:

Installation:
    
    The python package can be installed, e.g., with 'pip install .' in the package directory.

Content:
    
    The source code is contained in the folder 'xkn'.

    The folder 'filter_data' contains the data of AT2017gfo (one can add other observed or synthetic data there to compare with).
    The script 'gen_json_from_data.py' can be used to generate from data the filter dictionary, which contains the list of used filters.
    Note that the data must be stored in a specific format and with specific names (see AT2017gfo as an example).

    The folder 'flux_factor_data' contains tables of projection factors used to compute fluxes in anisotropic setups.

    The folder 'examples' contains a simple example of usage.
    The script 'example.py' computes the kilonova for a given model setup, specified by the 'kn_config.ini' file and on the fly by the user. The script evaluates the log-likelyhood compared to AT2017gfo and saves a plot of the resulting magnitudes vs data points.

Usage:

    The main MKN class is initialized starting from a configuration file .ini, which contains the setup of the model (see example).
    
    Inputs are divided into parameters (reasonably fixed for a given model) and variables (which can be changed on the fly).
    
    Each of the two groups further distinguishes between global inputs and inputs related to a specific ejecta component.
    
    NOTE: One can add an arbitrary number of ejecta components by simply listing the related inputs in the config file for both parameters and variables, using a different name (e.g. 'dynamical', 'wind', 'secular').
    
    Variables not fixed in the config file can be changed externally (see example).

    A list of the possible inputs with explanation can be visualized by running the function get_info() of the MKNConfig class (NOT COMPLETE YET).
    
    The main functions to be used are contained in the module 'mkn.py':
    
    - calc_lightcurve_vars() compute the bolometric light-curves and all the other relevant quantities as function of time (for every angular bin):
        lum_bol      : total bolometric luminosity (theta, time)
        lum_photo    : contribution to the bolometric luminosity from the photosphere (theta, time)
        radius_photo : photospheric radius (theta, time)
        T_photo      : photospheric temperature (theta, time)
        lum_shells   : contribution to the bolometric luminosity from the optically thin shells outside the photosphere (component, theta, time, shell)
        T_shells     : temperature of the optically thin shells outside the photosphere (component, theta, time, shell)
        lum_bol_raw  : total bolometric luminosity without the thin shell correction (theta, time)
        
        NOTE: the total bolometric luminosity will be 2*np.sum(lum_bol, axis=0), to account for both hemispheres.
    
    - calc_magnitudes() compute the magnitudes in the specified filters as function of time:
        return dictionary of magnitudes with correspondent times for each filter
    
    - calc_log_like() compute the loglikelyhood of the model compared to the data



