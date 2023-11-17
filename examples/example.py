from xkn import MKN, MKNConfig

# initialize MKNConfig object from the config file
config_path = 'examples/kn_config.ini'
mkn_config  = MKNConfig(config_path)

# Uncomment to print a list of inputs and their explantions
# mkn_config.get_info()

# initialize MKN object from the read parameters
mkn = MKN(*mkn_config.get_params(),log_level='WARNING')

if __name__ == '__main__':

    inputs = {
        'view_angle'           : 0.524,
        'distance'             : 40,

        'm_ej_dynamics'        : 0.03,
        'vel_dynamics'         : 0.13,
        'high_lat_op_dynamics' : 5,
        'low_lat_op_dynamics'  : 20,

        'm_ej_secular'         : 0.08,
        'vel_secular'          : 0.06,
        'op_secular'           : 5,

        'm_ej_wind'            : 0.02,
        'vel_wind'             : 0.1,
        'high_lat_op_wind'     : 1,
        'low_lat_op_wind'      : 5,
        }


    print('\nExample 1: Computing light-curves...')
    lum_bol, lum_photo, radius_photo, T_photo, lum_shells, T_shells, lum_bol_raw = mkn.calc_lightcurve_vars(mkn_config.get_vars(inputs))

    print('\nExample 2: Computing log-likelyhood...')
    log_like = mkn.calc_log_like(mkn_config.get_vars(inputs))
    print(f'log_like = {log_like}')

    print('\nExample 3: Computing magnitudes...')
    mag = mkn.calc_magnitudes(mkn_config.get_vars(inputs))

    print('\nExample 4: Computing and plotting magnitudes...')
    mkn.plot_magnitudes(mkn_config.get_vars(inputs), filename='examples/magnitudes.pdf')
