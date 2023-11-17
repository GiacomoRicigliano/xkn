import numpy as np
import json
# This updated dictionary separates bands for different telescopes.
# The majority of wavelengths are taken from SVO (Spanish Virtual Observatory) website (http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse).
# The missing information from SVO is taken from different instruments' websites.
# Then bands from different telescopes with the same wavelength are joined together.
# Note: band 'KK' is the K-band.
# Central Wavelengths in nanometers.

filters = {}

filters[443] ={ 'name'  : 'B_CTIO',
                 'lambda': 443.3e-9,
                 'type'  : 'AB',
                 'filename': ['mag_CTIO_band_B.txt']}

filters[828] ={ 'name'  : 'i_CTIO',
                 'lambda': 827.7e-9,
                 'type'  : 'AB',
                 'filename': ['mag_CTIO_band_i.txt']}

filters[657] ={ 'name'  : 'r_CTIO',
                 'lambda': 657.5e-9,
                 'type'  : 'AB',
                 'filename': ['mag_CTIO_band_r.txt']}

filters[548] ={ 'name'  : 'V_CTIO',
                 'lambda': 547.6e-9,
                 'type'  : 'AB',
                 'filename': ['mag_CTIO_band_V.txt']}

filters[2143] ={ 'name'  : 'K_CTIO',
                 'lambda': 2142.6e-9,
                 'type'  : 'AB',
                 'filename': ['mag_CTIO_band_KK.txt']}

filters[1631] ={ 'name'  : 'H_Gemini',
                 'lambda': 1630.5e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Gemini_telescope_band_H.txt','mag_Gemini-S_band_H.txt']}

filters[1255] ={ 'name'  : 'J_Gemini',
                 'lambda': 1255.1e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Gemini_telescope_band_J.txt','mag_Gemini-S_band_J.txt']}

filters[2157] ={ 'name'  : 'Ks_Gemini',
                 'lambda': 2157.4e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Gemini_telescope_band_Ks.txt','mag_Gemini-S_band_Ks.txt']}

filters[475] ={ 'name'  : 'g_Gemini_LCO1m_Keck',
                 'lambda': 475.1e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Gemini_telescope_band_g.txt','mag_LCO_band_g.txt','mag_Keck_band_g.txt']}

filters[779] ={ 'name'  : 'i_Gemini',
                 'lambda': 778.9e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Gemini_telescope_band_i.txt','mag_Gemini-S_band_i.txt']}

filters[778] ={ 'name'  : 'i_skymapper',
                 'lambda': 777.8e-9,
                 'type'  : 'AB',
                 'filename': ['mag_skymapper_band_i.txt']}

filters[630] ={ 'name'  : 'r_Gemini',
                 'lambda': 630.5e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Gemini_telescope_band_r.txt']}

filters[972] ={ 'name'  : 'z_Gemini',
                 'lambda': 972.3e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Gemini_telescope_band_z.txt']}

filters[633] ={ 'name'  : 'r_Gemini-S',
                 'lambda': 633.0e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Gemini-S_band_r.txt']}

filters[1162] ={ 'name'  : 'F110W_HST',
                 'lambda': 1162.4e-9,
                 'type'  : 'AB',
                 'filename': ['mag_HST_band_F110W.txt']}

filters[1539] ={ 'name'  : 'F160W_HST',
                 'lambda': 1539.2e-9,
                 'type'  : 'AB',
                 'filename': ['mag_HST_band_F160W.txt']}

filters[336] ={ 'name'  : 'F336W_HST',
                 'lambda': 335.9e-9,
                 'type'  : 'AB',
                 'filename': ['mag_HST_band_F336W.txt']}

filters[592] ={ 'name'  : 'F606W_HST',
                 'lambda': 592.5e-9,
                 'type'  : 'AB',
                 'filename': ['mag_HST_band_F606W.txt']}

filters[479] ={ 'name'  : 'F475W_HST_g_Swope',
                 'lambda': 479.2e-9,
                 'type'  : 'AB',
                 'filename': ['mag_HST_band_F475W.txt','mag_Swope_band_g.txt']}

filters[626] ={ 'name'  : 'F625W_HST',
                 'lambda': 625.8e-9,
                 'type'  : 'AB',
                 'filename': ['mag_HST_band_F625W.txt']}

filters[766] ={ 'name'  : 'F775W_HST',
                 'lambda': 766.0e-9,
                 'type'  : 'AB',
                 'filename': ['mag_HST_band_F775W.txt']}

filters[908] ={ 'name'  : 'F850W_HST',
                 'lambda': 908.4e-9,
                 'type'  : 'AB',
                 'filename': ['mag_HST_band_F850W.txt']}

filters[806] ={ 'name'  : 'F814W_HST',
                 'lambda': 805.8e-9,
                 'type'  : 'AB',
                 'filename': ['mag_HST_band_F814W.txt']}

filters[798] ={ 'name'  : 'i_VLT',
                 'lambda': 798.1e-9,
                 'type'  : 'AB',
                 'filename': ['mag_VLT_band_i.txt']}

filters[238] ={ 'name'  : 'F225W_HST',
                 'lambda': 237.8e-9,
                 'type'  : 'AB',
                 'filename': ['mag_HST_band_F225W.txt']}

filters[271] ={ 'name'  : 'F275W_HST',
                 'lambda': 271.5e-9,
                 'type'  : 'AB',
                 'filename': ['mag_HST_band_F275W.txt']}

filters[761] ={ 'name'  : 'i_Keck',
                 'lambda': 760.5e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Keck_band_i.txt']}

filters[623] ={ 'name'  : 'w_LCO',
                 'lambda': 623.5e-9,
                 'type'  : 'AB',
                 'filename': ['mag_LCO_band_w.txt']}

filters[753] ={ 'name'  : 'i_LCO',
                 'lambda': 753.2e-9,
                 'type'  : 'AB',
                 'filename': ['mag_LCO_band_i.txt']}

filters[537] ={ 'name'  : 'V_LCO',
                 'lambda': 537.0e-9,
                 'type'  : 'AB',
                 'filename': ['mag_LCO_band_V.txt']}

filters[869] ={ 'name'  : 'z_LCO',
                 'lambda': 869.2e-9,
                 'type'  : 'AB',
                 'filename': ['mag_LCO_band_z.txt']}

filters[459] ={ 'name'  : 'g_LaSilla',
                 'lambda': 458.7e-9,
                 'type'  : 'AB',
                 'filename': ['mag_LaSilla_band_g.txt']}

filters[622] ={ 'name'  : 'r_LaSilla',
                 'lambda': 622.0e-9,
                 'type'  : 'AB',
                 'filename': ['mag_LaSilla_band_r.txt']}

filters[764] ={ 'name'  : 'i_LaSilla',
                 'lambda': 764.1e-9,
                 'type'  : 'AB',
                 'filename': ['mag_LaSilla_band_i.txt']}

filters[899] ={ 'name'  : 'z_LaSilla',
                 'lambda': 899.0e-9,
                 'type'  : 'AB',
                 'filename': ['mag_LaSilla_band_z.txt']}

filters[1240] ={ 'name'  : 'J_LaSilla',
                 'lambda': 1239.9e-9,
                 'type'  : 'AB',
                 'filename': ['mag_LaSilla_band_J.txt']}

filters[1647] ={ 'name'  : 'H_LaSilla',
                 'lambda': 1646.9e-9,
                 'type'  : 'AB',
                 'filename': ['mag_LaSilla_band_H.txt']}

filters[2171] ={ 'name'  : 'K_LaSilla',
                 'lambda': 2170.6e-9,
                 'type'  : 'AB',
                 'filename': ['mag_LaSilla_band_KK.txt']}

filters[1619] ={ 'name'  : 'H_Magellan',
                 'lambda': 1618.8e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Magellan_band_H.txt']}

filters[1242] ={ 'name'  : 'J_Magellan',
                 'lambda': 1241.5e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Magellan_band_J.txt']}

filters[1055] ={ 'name'  : 'J1_Magellan',
                 'lambda': 1055.1e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Magellan_band_J1.txt']}

filters[555] ={ 'name'  : 'V_NTT',
                 'lambda': 554.6e-9,
                 'type'  : 'AB',
                 'filename': ['mag_NTT_band_V.txt']}

filters[355] ={ 'name'  : 'U_NTT',
                 'lambda': 355.0e-9,
                 'type'  : 'AB',
                 'filename': ['mag_NTT_band_U.txt']}

filters[513] ={ 'name'  : 'g_NTT',
                 'lambda': 513.1e-9,
                 'type'  : 'AB',
                 'filename': ['mag_NTT_band_g.txt']}

filters[802] ={ 'name'  : 'i_NTT',
                 'lambda': 801.9e-9,
                 'type'  : 'AB',
                 'filename': ['mag_NTT_band_i.txt']}

filters[670] ={ 'name'  : 'r_NTT',
                 'lambda': 669.8e-9,
                 'type'  : 'AB',
                 'filename': ['mag_NTT_band_r.txt']}

filters[1652] ={ 'name'  : 'H_NTT',
                 'lambda': 1652.0e-9,
                 'type'  : 'AB',
                 'filename': ['mag_NTT_band_H.txt']}

filters[2164] ={ 'name'  : 'Ks_NTT',
                 'lambda': 2163.8e-9,
                 'type'  : 'AB',
                 'filename': ['mag_NTT_band_Ks.txt']}

filters[754] ={ 'name'  : 'i_PAN-STARRS',
                 'lambda': 754.5e-9,
                 'type'  : 'AB',
                 'filename': ['mag_PAN-STARRS_band_i.txt']}

filters[963] ={ 'name'  : 'y_PAN-STARRS',
                 'lambda': 963.3e-9,
                 'type'  : 'AB',
                 'filename': ['mag_PAN-STARRS_band_y.txt']}

filters[868] ={ 'name'  : 'z_PAN-STARRS',
                 'lambda': 868.0e-9,
                 'type'  : 'AB',
                 'filename': ['mag_PAN-STARRS_band_z.txt']}

filters[621] ={ 'name'  : 'r_PAN-STARRS',
                 'lambda': 621.5e-9,
                 'type'  : 'AB',
                 'filename': ['mag_PAN-STARRS_band_r.txt']}

filters[2178] ={ 'name'  : 'Ks_Palomar5m',
                 'lambda': 2177.7e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Palomar5m_band_Ks.txt']}

filters[892] ={ 'name'  : 'z_Subaru',
                 'lambda': 892.5e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Subaru_band_z.txt']}

filters[2145] ={ 'name'  : 'Ks_Subaru',
                 'lambda': 2145.3e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Subaru_band_Ks.txt']}

filters[225] ={ 'name'  : 'M2_Swift',
                 'lambda': 225.5e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Swift_band_M2.txt']}

filters[261] ={ 'name'  : 'W1_Swift',
                 'lambda': 261.4e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Swift_band_W1.txt']}

filters[348] ={ 'name'  : 'U_Swift',
                 'lambda': 347.6e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Swift_band_U.txt']}

filters[208] ={ 'name'  : 'W2_Swift',
                 'lambda': 208.0e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Swift_band_W2.txt']}

filters[436] ={ 'name'  : 'B_Swift',
                 'lambda': 435.9e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Swift_band_B.txt']}

filters[543] ={ 'name'  : 'V_Swift',
                 'lambda': 543.0e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Swift_band_V.txt']}

filters[763] ={ 'name'  : 'i_Swope',
                 'lambda': 763.0e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Swope_band_i.txt']}

filters[539] ={ 'name'  : 'V_Swope',
                 'lambda': 538.8e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Swope_band_V.txt']}

filters[440] ={ 'name'  : 'B_Swope',
                 'lambda': 439.7e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Swope_band_B.txt']}

filters[624] ={ 'name'  : 'r_Swope_LCO',
                 'lambda': 624.0e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Swope_band_r.txt','mag_LCO_band_r.txt']}

filters[480] ={ 'name'  : 'g_T80s',
                 'lambda': 480.3e-9,
                 'type'  : 'AB',
                 'filename': ['mag_T80s_band_g.txt']}

filters[625] ={ 'name'  : 'r_T80s',
                 'lambda': 625.4e-9,
                 'type'  : 'AB',
                 'filename': ['mag_T80s_band_r.txt']}

filters[767] ={ 'name'  : 'i_T80s',
                 'lambda': 766.8e-9,
                 'type'  : 'AB',
                 'filename': ['mag_T80s_band_i.txt']}

filters[2149] ={ 'name'  : 'Ks_VISTA',
                 'lambda': 2148.8e-9,
                 'type'  : 'AB',
                 'filename': ['mag_VISTA_band_Ks.txt']}

filters[1254] ={ 'name'  : 'J_VISTA_NTT',
                 'lambda': 1254.1e-9,
                 'type'  : 'AB',
                 'filename': ['mag_VISTA_band_J.txt','mag_NTT_band_J.txt']}

filters[1021] ={ 'name'  : 'Y_VISTA',
                 'lambda': 1021.1e-9,
                 'type'  : 'AB',
                 'filename': ['mag_VISTA_band_Y.txt']}

filters[652] ={ 'name'  : 'r_VLT',
                 'lambda': 652.1e-9,
                 'type'  : 'AB',
                 'filename': ['mag_VLT_band_r.txt']}

filters[918] ={ 'name'  : 'z_VLT_DECam',
                 'lambda': 917.6e-9,
                 'type'  : 'AB',
                 'filename': ['mag_VLT_band_z.txt','mag_DECam_band_z.txt']}

filters[509] ={ 'name'  : 'g_VLT',
                 'lambda': 509.3e-9,
                 'type'  : 'AB',
                 'filename': ['mag_VLT_band_g.txt']}

filters[425] ={ 'name'  : 'B_VLT',
                 'lambda': 424.7e-9,
                 'type'  : 'AB',
                 'filename': ['mag_VLT_band_B.txt']}

filters[550] ={ 'name'  : 'V_VLT',
                 'lambda': 550.4e-9,
                 'type'  : 'AB',
                 'filename': ['mag_VLT_band_V.txt']}

filters[8718] ={ 'name'  : 'J89_VLT',
                 'lambda': 8718.4e-9,
                 'type'  : 'AB',
                 'filename': ['mag_VLT_band_J8.9.txt']}

filters[2148] ={ 'name'  : 'Ks_VLT_Magellan',
                 'lambda': 2148.5e-9,
                 'type'  : 'AB',
                 'filename': ['mag_VLT_band_Ks.txt','mag_Magellan_band_Ks.txt']}

filters[616] ={ 'name'  : 'r_skymapper',
                 'lambda': 615.7e-9,
                 'type'  : 'AB',
                 'filename': ['mag_skymapper_band_r.txt']}

filters[510] ={ 'name'  : 'g_skymapper',
                 'lambda': 509.9e-9,
                 'type'  : 'AB',
                 'filename': ['mag_skymapper_band_g.txt']}

filters[783] ={ 'name'  : 'i_DECam',
                 'lambda': 782.6e-9,
                 'type'  : 'AB',
                 'filename': ['mag_DECam_band_i.txt']}

filters[989] ={ 'name'  : 'Y_DECam',
                 'lambda': 989.6e-9,
                 'type'  : 'AB',
                 'filename': ['mag_DECam_band_Y.txt']}

filters[644] ={ 'name'  : 'r_DECam',
                 'lambda': 643.5e-9,
                 'type'  : 'AB',
                 'filename': ['mag_DECam_band_r.txt']}

filters[483] ={ 'name'  : 'g_DECam',
                 'lambda': 482.7e-9,
                 'type'  : 'AB',
                 'filename': ['mag_DECam_band_g.txt']}

filters[382] ={ 'name'  : 'u_DECam',
                 'lambda': 381.7e-9,
                 'type'  : 'AB',
                 'filename': ['mag_DECam_band_u.txt']}

filters[1650] ={ 'name'  : 'H_IRSF',
                 'lambda': 1650.0e-9,
                 'type'  : 'AB',
                 'filename': ['mag_IRSF_band_H.txt']}

filters[1200] ={ 'name'  : 'J_IRSF',
                 'lambda': 1200.0e-9,
                 'type'  : 'AB',
                 'filename': ['mag_IRSF_band_J.txt']}

filters[2150] ={ 'name'  : 'Ks_IRSF',
                 'lambda': 2150.0e-9,
                 'type'  : 'AB',
                 'filename': ['mag_IRSF_band_Ks.txt']}

filters[2200] ={ 'name'  : 'K_Gemini',
                 'lambda': 2200.0e-9,
                 'type'  : 'AB',
                 'filename': ['mag_Gemini_telescope_band_KK.txt']}

filters[439] ={ 'name'  : 'B_MASTER',
                 'lambda': 439.2e-9,
                 'type'  : 'AB',
                 'filename': ['mag_MASTER_band_B.txt']}

filters[631] ={ 'name'  : 'R_MASTER',
                 'lambda': 631.3e-9,
                 'type'  : 'AB',
                 'filename': ['mag_MASTER_band_R.txt']}

with open('telescopes.json', 'w') as fi:
    json.dump(filters, fi, indent = '    ', sort_keys = True)
