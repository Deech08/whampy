import logging

from astropy import units as u
import numpy as np 
import matplotlib.pyplot as plt
from scipy import interpolate

from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle

try:
    from extinction import fm07 as extinction_law
except ModuleNotFoundError:
    # Error handling
    pass
try:
    from dustmaps.marshall import MarshallQuery
    from dustmaps.bayestar import BayestarQuery
    from dustmaps.bayestar import BayestarWebQuery
except ModuleNotFoundError:
    # Error handling
    pass

try:
    import statsmodels.api as sm
except ModuleNotFoundError:
    # Error handling
    pass

from scipy.interpolate import interp1d
from scipy import stats

import seaborn as sns
pal = sns.color_palette("colorblind")
from seaborn.algorithms import bootstrap

import pandas as pd


def kinematic_distance_flat(l, v, R_sun = 8.12 * u.kpc, v_sun = 220 * u.km/u.s):
    term1 = R_sun * np.cos(l*u.deg)
    r = R_sun * np.sin(l*u.deg) * v_sun / (v*u.km/u.s + v_sun * np.sin(l*u.deg))
    term2 = np.sqrt(r**2 - (R_sun *  np.sin(l*u.deg))**2)
    
    return term1 + term2, term1 - term2


def get_scale_height_data(data, track = None, deredden = False, 
                          return_pandas_dataframe = False, 
                          longitude_mask_width = None, 
                          step_size = None, R_sun = None, v_sun = None, 
                          closer = False, add_kinematic_distance = False, **kwargs):
    """
    Return data needed for scale height analysis
    
    Parameters
    ----------
    data: `skySurvey`
        WHAM skySurvey object of full sky (requires track keyword), or spiral arm section
    track: `str`, optional, must be keyword
        if provided, will apply skySurvey.get_spiral_slice for provided track
        if None, will check for track as an attribute of data
    deredden: `bool`, `dustmap`, optional, must be keyword
        if True, will apply dereddening using 3D dustmaps of Marshall et al. (2006)
        or can input a dustmap to query from using the dustmaps package
        default to `dustmaps.marshall.MarshallQuery`
        Warning: Currently only supports Marshall Dustmap
    return_pandas_dataframe: `bool`, optional, must be keyword
        if True, returns pandas dataframe with subset of data specific to scale height analysis
    longitude_mask_width: `number`, `u.Quantity`, optional, must be keyword
        if provided, returns list of masks splitting data into sky sections 
        of given width in degrees
    step_size: `number`, `u.Quantity`, optional, must be keyword
        if provided, sets step_size for longitude masks
        default to half width
    R_sun: `u.Quantity`, optional, must be keyword
        Sun Galactocentric Distance
    v_sun: `u.Quantity`, optional, must be keyword
        Sun rotation velocity
    add_kinematic_distance: `bool`, optional, must be keyword
        if True, adds in kinematic distances using a flat rotation curve 
        where no parallax ones available

    **kwargs: `dict`
        keywords passed to data.get_spiral_slice if track is provided
    """
    # Check Wrapping
    if "wrap_at_180" in kwargs:
        if kwargs["wrap_at_180"]:
            wrap_at = "180d"
        else:
            wrap_at = "360d"
    else:
        wrap_at = "360d"

    if add_kinematic_distance:
        if R_sun is None:
            R_sun = 8.127 * u.kpc
        if v_sun is None:
            v_sun = 220*u.km/u.s
        
    # Must have return_track
    try:
        test = np.all(kwargs["return_track"])
    except KeyError:
        kwargs["return_track"] = True
    finally: 
        if kwargs["return_track"] is False:
            kwargs["return_track"] = True
            logging.warning("keyword 'return_track' must be set to True!")
    
    # Get / ensure proper track is available
    if (track is None):
        if not hasattr(data, "lbv_RBD_track"):
            raise SyntaxError("No track provided - distance information is required")
    else:
        data, data.lbv_RBD_track = data.get_spiral_slice(track = track, **kwargs)
    
    if data.lbv_RBD_track.shape[1] < 3:
        raise ValueError("provided track does not have distance information!")

    if add_kinematic_distance:
        distance_is_none = data.lbv_RBD_track[:,-1] == 0.0
        if distance_is_none.sum() > 0:
            distances_1, distances_2  = kinematic_distance_flat(data.lbv_RBD_track[distance_is_none,0], 
                data.lbv_RBD_track[distance_is_none,1], R_sun = R_sun, v_sun = v_sun)

            if closer:
                neg_dist = distances_2 < 0.
                distances_2[neg_dist] = distances_1[neg_dist]
                data.lbv_RBD_track[distance_is_none,-1] = distances_2
            else:
                data.lbv_RBD_track[distance_is_none,-1] = distances_1



    
    # Setup dustmaps if needed
    if deredden.__class__ is bool:
        if deredden:
            deredden = MarshallQuery()
    elif not hasattr(deredden, "query"):
        raise TypeError("invaled dustmap provided - must provide a dustmap class that can query or set to \
                        True to set defualt dustmap to MarshallQuery.")
        
    data["tan(b)"] = np.tan(data["GAL-LAT"].data*u.deg)
    
    # Apply dereddening
    if not deredden.__class__ is bool:

        # Get all distances assuming plane parallel
        distance_interpolator = interpolate.interp1d(Angle(data.lbv_RBD_track[:,0]*u.deg).wrap_at(wrap_at), 
                                                     data.lbv_RBD_track[:,-1])
        distances = distance_interpolator(Angle(data["GAL-LON"].data*u.deg).wrap_at(wrap_at))
        coordinates = data.get_SkyCoord(distance = distances * u.kpc)


        if (deredden.__class__ is BayestarQuery) | (deredden.__class__ is BayestarWebQuery):
            Av_bayestar = 2.742 * deredden(coordinates)
            wave_ha = np.array([6562.8])
            A_V_to_A_ha = extinction_law(wave_ha, 1.)
            data["DISTANCE"] = distances * u.kpc
            data["Z"] = data["DISTANCE"] * data["tan(b)"]
            data["Av"] = Av_bayestar
            data["INTEN_DERED"] = data["INTEN"][:]
            data["INTEN_DERED"][~np.isnan(Av_bayestar)] = \
                    data["INTEN"][~np.isnan(Av_bayestar)] * 10**(0.4 * A_V_to_A_ha  * Av_bayestar[~np.isnan(Av_bayestar)])
        else:

            
            # Get A_Ks
            AKs = deredden(coordinates)
            wave_Ks = 2.17 *u.micron
            A_KS_to_A_v = 1. / extinction_law(np.array([wave_Ks.to(u.AA).value]), 1.)
            wave_ha = np.array([6562.8])
            A_V_to_A_ha = extinction_law(wave_ha, 1.)
            data["DISTANCE"] = distances * u.kpc
            data["Z"] = data["DISTANCE"] * data["tan(b)"]
            data["Av"] = A_KS_to_A_v * AKs
            data["INTEN_DERED"] = data["INTEN"][:]
            data["INTEN_DERED"][~np.isnan(AKs)] = \
                    data["INTEN"][~np.isnan(AKs)] * 10**(0.4 * A_V_to_A_ha * A_KS_to_A_v * AKs[~np.isnan(AKs)])
    
    if not longitude_mask_width is None:
        if not isinstance(longitude_mask_width, u.Quantity):
            longitude_mask_width *= u.deg
            logging.warning("No units provided for longitude_mask_width, assuming u.deg.")
        if step_size is None:
            step_size = longitude_mask_width / 2.
        elif not isinstance(step_size, u.Quantity):
            step_size *= u.deg
            logging.warning("No units provided for step_size, assuming u.deg.")

        # Construct masks
        wrapped_lon = Angle(data["GAL-LON"]).wrap_at(wrap_at)
        lon_range = np.min(wrapped_lon), np.max(wrapped_lon)
        n_steps = int(np.ceil(np.round(lon_range[1] - lon_range[0]) / step_size))
        lon_edge = np.linspace(lon_range[0], lon_range[1], n_steps)
        lon_edges = np.zeros((len(lon_edge)-1, 2)) * u.deg
        lon_edges[:,0] = lon_edge[:-1] 
        lon_edges[:,1] = lon_edge[:-1] + longitude_mask_width
        masks = [((wrapped_lon < lon_upper) & (wrapped_lon >= lon_lower)) \
                 for (lon_lower, lon_upper) in lon_edges]

    
    
    if return_pandas_dataframe:
        try:
            df = pd.DataFrame({
                "INTEN":data["INTEN"].byteswap().newbyteorder(), 
                "INTEN_DERED":data["INTEN_DERED"].byteswap().newbyteorder(),
                "tan(b)":data["tan(b)"].byteswap().newbyteorder(),
                "GAL-LON":data["GAL-LON"].byteswap().newbyteorder(),
                "GAL-LAT":data["GAL-LAT"].byteswap().newbyteorder(),
                "Av":data["Av"].byteswap().newbyteorder(),
                "DISTANCE":data["DISTANCE"],
                "Z":data["Z"]
            })
        except KeyError:
            df = pd.DataFrame({
                "INTEN":data["INTEN"].byteswap().newbyteorder(), 
                "tan(b)":data["tan(b)"].byteswap().newbyteorder(),
                "GAL-LON":data["GAL-LON"].byteswap().newbyteorder(),
                "GAL-LAT":data["GAL-LAT"].byteswap().newbyteorder(),
            })
        if longitude_mask_width is None:
            return data, df
        else:
            return data, df, masks
    else:
        return data


def fit_scale_heights(data, masks, min_lat = None, max_lat = None, 
    deredden = False, fig_names = None, return_smoothed = False, 
    smoothed_width = None, xlim = None, ylim = None, robust = True, 
    n_boot = 10000):
    """
    Fits scale height data and returns slopes

    Parameters
    ----------
    data: `skySurvey`
        WHAM skySurvey object of full sky (requires track keyword), or spiral arm section
    masks: `list like`
        longitude masks to use
    min_lat:   `u.Quantity`
        min latitude to fit
    max_lat:   `u.Quantity`
        max latitude to fit
    deredden: `bool`
        if True, also fits dereddened slopes
    fig_names: `str`
        if provided, saves figures following this name
    return_smoothed: `bool`
        if True, returns smoothed longitude and slope estimates
    smoothed_width: `u.Quantity`
        width to smooth data to in longitude
    robust: `bool`
        if True, uses stats.models.robust_linear_model
    n_boot: `int`
        only if robust = True
        number of bootstrap resamples
    """

    # Default values
    if min_lat is None:
        min_lat = 5*u.deg
    elif not hasattr(min_lat, "unit"):
        min_lat *= u.deg

    if max_lat is None:
        max_lat = 35*u.deg
    elif not hasattr(max_lat, "unit"):
        max_lat *= u.deg   

    if smoothed_width is None:
        smoothed_width = 5*u.deg
    elif not hasattr(smoothed_width, "unit"):
        smoothed_width *= u.deg

    #initialize data arrays

    slopes_pos = []
    slopes_neg = []
    slopes_pos_dr = []
    slopes_neg_dr = []
    intercept_pos = []
    intercept_neg = []
    intercept_pos_dr = []
    intercept_neg_dr = []
    slopes_pos_err = []
    slopes_neg_err = []
    slopes_pos_dr_err = []
    slopes_neg_dr_err = []
    intercept_pos_err = []
    intercept_neg_err = []
    intercept_pos_dr_err = []
    intercept_neg_dr_err = []
    median_longitude = []
    median_distance = []
    for ell2 in range(len(masks)):
        xx = data["tan(b)"][masks[ell2]]
        yy = np.log(data["INTEN"][masks[ell2]])
        nan_mask = np.isnan(yy)
        nan_mask |= np.isinf(yy)
        
        if deredden:
            zz = np.log(data["INTEN_DERED"][masks[ell2]])
            nan_mask_z = np.isnan(zz)
            nan_mask_z |= np.isinf(zz)

        median_longitude.append(np.median(data["GAL-LON"][masks[ell2]]))
        if deredden:
            median_distance.append(np.median(data["DISTANCE"][masks[ell2]]))

        y_min = np.tan(min_lat)
        y_max = np.tan(max_lat)

        if not robust:
        
            if hasattr(stats, "siegelslopes"):
                slope_estimator = stats.siegelslopes
            else:
                logging.warning("Installed version of scipy does not have the siegelslopes method in scipy.stats!")
                slope_estimator = stats.theilslopes

            siegel_result_pos = slope_estimator(yy[(xx > y_min) & (xx < y_max) & ~nan_mask],
                                                   xx[(xx > y_min) & (xx < y_max) & ~nan_mask])
            siegel_result_neg = slope_estimator(yy[(xx < -y_min) & (xx > -y_max) & ~nan_mask],
                                                   xx[(xx < -y_min) & (xx > -y_max) & ~nan_mask])
            
            if deredden:
                siegel_result_pos_dr = slope_estimator(zz[(xx > y_min) & (xx < y_max) & ~nan_mask_z],
                                                       xx[(xx > y_min) & (xx < y_max) & ~nan_mask_z])
                siegel_result_neg_dr = slope_estimator(zz[(xx < -y_min) & (xx > -y_max) & ~nan_mask_z],
                                                       xx[(xx < -y_min) & (xx > -y_max) & ~nan_mask_z])

            slopes_pos.append(siegel_result_pos[0])
            slopes_neg.append(siegel_result_neg[0])
            
            intercept_pos.append(siegel_result_pos[1])
            intercept_neg.append(siegel_result_neg[1])

            if deredden:
                slopes_pos_dr.append(siegel_result_pos_dr[0])
                slopes_neg_dr.append(siegel_result_neg_dr[0])
                intercept_pos_dr.append(siegel_result_pos_dr[1])
                intercept_neg_dr.append(siegel_result_neg_dr[1])

            if fig_names is not None:
                figure_name = "{0}_{1}.png".format(fig_names, ell2)

                if xlim is None:
                    xlim = np.array([-0.9, 0.9])
                if ylim is None:
                    ylim = np.array([-4.6, 3.2])

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax2 = ax.twiny()


                ax.scatter(xx, 
                           yy, 
                           color ="k", 
                           alpha = 0.8)
                if deredden:
                    ax.scatter(xx, 
                               zz, 
                               color ="grey", 
                               alpha = 0.8)


                ax.set_xlabel(r"$\tan$(b)", fontsize= 12)
                ax.set_ylabel(r"$\log$($H\alpha$ Intensity / R)", fontsize= 12)

                ax.set_title(r"${0:.1f} < l < {1:.1f}$".format(data["GAL-LON"][masks[ell2]].min(), 
                                                               data["GAL-LON"][masks[ell2]].max()), 
                             fontsize = 14)

                ax2.plot(np.degrees(np.arctan(xlim)), 
                         np.log([0.1,0.1]), ls = ":", lw = 1, 
                        color = "k", label = "0.1 R")
                ax2.fill_between([-min_lat.value, min_lat.value], [ylim[0], ylim[0]], [ylim[1], ylim[1]],
                                color = pal[1], 
                                alpha = 0.1, 
                                label = r"$|b| < 5\degree$")

                line_xx = np.linspace(y_min, y_max, 10)
                line_yy_pos = siegel_result_pos[0] * line_xx + siegel_result_pos[1]
                line_yy_neg = siegel_result_neg[0] * -line_xx + siegel_result_neg[1]
                ax.plot(line_xx, line_yy_pos, color = "r", lw = 3, alpha = 0.9, 
                        label = r"$H_{{n_e^2}} = {0:.2f} D$".format(1/-siegel_result_pos[0]))
                ax.plot(-line_xx, line_yy_neg, color = "b", lw = 3, alpha = 0.9, 
                        label = r"$H_{{n_e^2}} = {0:.2f} D$".format(1/siegel_result_neg[0]))

                if deredden:
                    line_yy_pos_dr = siegel_result_pos_dr[0] * line_xx + siegel_result_pos_dr[1]
                    line_yy_neg_dr = siegel_result_neg_dr[0] * -line_xx + siegel_result_neg_dr[1]
                    ax.plot(line_xx, line_yy_pos_dr, color = "r", lw = 3, alpha = 0.9, ls = "--", 
                            label = r"Dered: $H_{{n_e^2}} = {0:.2f} D$".format(1/-siegel_result_pos_dr[0]))
                    ax.plot(-line_xx, line_yy_neg_dr, color = "b", lw = 3, alpha = 0.9,  ls = "--", 
                            label = r"Dered: $H_{{n_e^2}} = {0:.2f} D$".format(1/siegel_result_neg_dr[0]))

                
                
                

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

                ax2.set_xlabel(r"$b$ (deg)", fontsize = 12)
                ax2.set_xlim(np.degrees(np.arctan(xlim)))

                ax.legend(fontsize = 12, loc = 1)
                ax2.legend(fontsize = 12, loc = 2)

                plt.tight_layout()

                plt.savefig(figure_name, dpi = 300)
                del(fig)
                plt.close()

            results = {
            "median_longitude":np.array(median_longitude),
            "slopes_pos":np.array(slopes_pos),
            "slopes_neg":np.array(slopes_neg),
            "intercept_pos":np.array(intercept_pos),
            "intercept_neg":np.array(intercept_neg)
            }

            if deredden:

                results["median_distance"] = np.array(median_distance),
                results["slopes_pos_dr"] = np.array(slopes_pos_dr)
                results["slopes_neg_dr"] = np.array(slopes_neg_dr)
                results["intercept_pos_dr"] = np.array(intercept_pos_dr)
                results["intercept_neg_dr"] = np.array(intercept_neg_dr)

        else:
            yy_pos = yy[(xx > y_min) & (xx < y_max) & ~nan_mask]
            xx_pos = xx[(xx > y_min) & (xx < y_max) & ~nan_mask]
            yy_neg = yy[(xx < -y_min) & (xx > -y_max) & ~nan_mask]
            xx_neg = xx[(xx < -y_min) & (xx > -y_max) & ~nan_mask]
            if ((len(yy_pos) < 5) | (len(yy_neg) < 5)):
                
                XX_pos = sm.add_constant(xx_pos)
                res_pos = sm.RLM(yy_pos, XX_pos, M=sm.robust.norms.HuberT()).fit()
                XX_neg = sm.add_constant(xx_neg)
                res_neg = sm.RLM(yy_neg, XX_neg, M=sm.robust.norms.HuberT()).fit()
                
                slopes_pos.append(res_pos.params[1])
                slopes_neg.append(res_neg.params[1])
                slopes_pos_err.append(res_pos.bse[1])
                slopes_neg_err.append(res_neg.bse[1])
                
                intercept_pos.append(res_pos.params[0])
                intercept_neg.append(res_neg.params[0])
                intercept_pos_err.append(res_pos.bse[0])
                intercept_neg_err.append(res_neg.bse[0])
            else:
                if deredden:
                    zz_dr_pos = zz[(xx > y_min) & (xx < y_max) & ~nan_mask_z]
                    xx_dr_pos = xx[(xx > y_min) & (xx < y_max) & ~nan_mask_z]
                    zz_dr_neg = zz[(xx < -y_min) & (xx > -y_max) & ~nan_mask_z]
                    xx_dr_neg = xx[(xx < -y_min) & (xx > -y_max) & ~nan_mask_z]
                    def slope_int_estimator_pos_dr(inds, 
                        YY = zz_dr_pos,
                        XX = xx_dr_pos):
                        """
                        estimate slope using sm.RLM
                        """
                        XX = XX[inds]
                        YY = YY[inds]
                        XX = sm.add_constant(XX)
                        res = sm.RLM(YY, XX, M=sm.robust.norms.HuberT()).fit()
                        return res.params

                    def slope_int_estimator_neg_dr(inds, 
                        YY = zz_dr_neg,
                        XX = xx_dr_neg):
                        """
                        estimate slope using sm.RLM
                        """
                        XX = XX[inds]
                        YY = YY[inds]
                        XX = sm.add_constant(XX)
                        res = sm.RLM(YY, XX, M=sm.robust.norms.HuberT()).fit()
                        return res.params

                def slope_int_estimator_pos(inds, 
                    YY = yy_pos,
                    XX = xx_pos):
                    """
                    estimate slope using sm.RLM
                    """
                    XX = XX[inds]
                    YY = YY[inds]
                    XX = sm.add_constant(XX)
                    res = sm.RLM(YY, XX, M=sm.robust.norms.HuberT()).fit()
                    return res.params

                def slope_int_estimator_neg(inds, 
                    YY = yy_neg,
                    XX = xx_neg):
                    """
                    estimate slope using sm.RLM
                    """
                    XX = XX[inds]
                    YY = YY[inds]
                    XX = sm.add_constant(XX)
                    res = sm.RLM(YY, XX, M=sm.robust.norms.HuberT()).fit()
                    return res.params
                

                boot_pos = bootstrap(np.arange(len(yy_pos)), func = slope_int_estimator_pos, n_boot = n_boot)
                boot_neg = bootstrap(np.arange(len(yy_neg)), func = slope_int_estimator_neg, n_boot = n_boot)

                slopes_pos.append(np.mean(boot_pos[:,1], axis = 0))
                slopes_neg.append(np.mean(boot_neg[:,1], axis = 0))
                slopes_pos_err.append(np.std(boot_pos[:,1], axis = 0))
                slopes_neg_err.append(np.std(boot_neg[:,1], axis = 0))
                
                intercept_pos.append(np.mean(boot_pos[:,0], axis = 0))
                intercept_neg.append(np.mean(boot_neg[:,0], axis = 0))
                intercept_pos_err.append(np.std(boot_pos[:,0], axis = 0))
                intercept_neg_err.append(np.std(boot_neg[:,0], axis = 0))

                if deredden:
                    boot_pos_dr = bootstrap(np.arange(len(zz_dr_pos)), func = slope_int_estimator_pos_dr, n_boot = n_boot)
                    boot_neg_dr = bootstrap(np.arange(len(zz_dr_neg)), func = slope_int_estimator_neg_dr, n_boot = n_boot)

                    slopes_pos_dr.append(np.mean(boot_pos_dr[:,1], axis = 0))
                    slopes_neg_dr.append(np.mean(boot_neg_dr[:,1], axis = 0))
                    slopes_pos_dr_err.append(np.std(boot_pos_dr[:,1], axis = 0))
                    slopes_neg_dr_err.append(np.std(boot_neg_dr[:,1], axis = 0))
                    
                    intercept_pos_dr.append(np.mean(boot_pos_dr[:,0], axis = 0))
                    intercept_neg_dr.append(np.mean(boot_neg_dr[:,0], axis = 0))
                    intercept_pos_dr_err.append(np.std(boot_pos_dr[:,0], axis = 0))
                    intercept_neg_dr_err.append(np.std(boot_neg_dr[:,0], axis = 0))


                if fig_names is not None:
                    figure_name = "{0}_{1}.png".format(fig_names, ell2)

                    if xlim is None:
                        xlim = np.array([-0.9, 0.9])
                    if ylim is None:
                        ylim = np.array([-4.6, 3.2])

                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax2 = ax.twiny()


                    ax.scatter(xx, 
                               yy, 
                               color ="k", 
                               alpha = 0.8)
                    if deredden:
                        ax.scatter(xx, 
                                   zz, 
                                   color ="grey", 
                                   alpha = 0.8)


                    ax.set_xlabel(r"$\tan$(b)", fontsize= 12)
                    ax.set_ylabel(r"$\log$($H\alpha$ Intensity / R)", fontsize= 12)

                    ax.set_title(r"${0:.1f} < l < {1:.1f}$".format(data["GAL-LON"][masks[ell2]].min(), 
                                                                   data["GAL-LON"][masks[ell2]].max()), 
                                 fontsize = 14)

                    ax2.plot(np.degrees(np.arctan(xlim)), 
                             np.log([0.1,0.1]), ls = ":", lw = 1, 
                            color = "k", label = "0.1 R")
                    ax2.fill_between([-min_lat.value, min_lat.value], [ylim[0], ylim[0]], [ylim[1], ylim[1]],
                                    color = pal[1], 
                                    alpha = 0.1, 
                                    label = r"$|b| < 5\degree$")

                    line_xx = np.linspace(y_min, y_max, 100)
                    def get_slope_conf_band(boot_res, X = line_xx):
                        yy = [[res[0] + res[1] * X] for res in boot_res]
                        yy = np.vstack(yy)
                        return np.percentile(yy, (5,95), axis = 0)


                    line_yy_pos = slopes_pos[-1] * line_xx + intercept_pos[-1]
                    line_yy_neg = slopes_neg[-1] * -line_xx + intercept_neg[-1]
                    line_yy_pos_range = get_slope_conf_band(boot_pos)
                    line_yy_neg_range = get_slope_conf_band(boot_neg, X = -line_xx)

                    ax.plot(line_xx, line_yy_pos, color = "r", lw = 3, alpha = 0.9, 
                            label = r"$H_{{n_e^2}} = ({0:.2f} \pm {1:.2f}) D$".format(1/-slopes_pos[-1], np.abs(1/slopes_pos[-1] * slopes_pos_err[-1] / slopes_pos[-1])))
                    ax.fill_between(line_xx, line_yy_pos_range[0], line_yy_pos_range[1], 
                        color = "r", alpha = 0.2)
                    ax.plot(-line_xx, line_yy_neg, color = "b", lw = 3, alpha = 0.9, 
                            label = r"$H_{{n_e^2}} = ({0:.2f} \pm {1:.2f}) D$".format(1/slopes_neg[-1],  np.abs(-1/slopes_pos[-1] * slopes_pos_err[-1] / slopes_pos[-1])))
                    ax.fill_between(-line_xx, line_yy_neg_range[0], line_yy_neg_range[1], 
                        color = "b", alpha = 0.2)

                    if deredden:
                        line_yy_pos_dr = slopes_pos_dr[-1] * line_xx + intercept_pos_dr[-1]
                        line_yy_neg_dr = slopes_neg_dr[-1] * -line_xx + intercept_neg_dr[-1]
                        line_yy_pos_range_dr = get_slope_conf_band(boot_pos_dr)
                        line_yy_neg_range_dr = get_slope_conf_band(boot_neg_dr, X = -line_xx)

                        ax.plot(line_xx, line_yy_pos_dr, color = "r", lw = 3, alpha = 0.9, ls = "--",
                                label = r"Dered: $H_{{n_e^2}} = ({0:.2f} \pm {1:.2f}) D$".format(1/-slopes_pos_dr[-1], np.abs(1/slopes_pos_dr[-1] * slopes_pos_dr_err[-1] / slopes_pos_dr[-1])))
                        ax.fill_between(line_xx, line_yy_pos_range_dr[0], line_yy_pos_range_dr[1], 
                            color = "r", alpha = 0.2)
                        ax.plot(-line_xx, line_yy_neg_dr, color = "b", lw = 3, alpha = 0.9, ls = "--",
                                label = r"Dered: $H_{{n_e^2}} = ({0:.2f} \pm {1:.2f}) D$".format(1/slopes_neg_dr[-1], np.abs(-1/slopes_pos_dr[-1] * slopes_pos_dr_err[-1] / slopes_pos_dr[-1])))
                        ax.fill_between(-line_xx, line_yy_neg_range_dr[0], line_yy_neg_range_dr[1], 
                            color = "b", alpha = 0.2)
                        

                    
                    
                    

                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)

                    ax2.set_xlabel(r"$b$ (deg)", fontsize = 12)
                    ax2.set_xlim(np.degrees(np.arctan(xlim)))

                    ax.legend(fontsize = 12, loc = 1)
                    ax2.legend(fontsize = 12, loc = 2)

                    plt.tight_layout()

                    plt.savefig(figure_name, dpi = 300)
                    del(fig)
                    plt.close()










            results = {
            "median_longitude":np.array(median_longitude),
            "slopes_pos":np.array(slopes_pos),
            "slopes_neg":np.array(slopes_neg),
            "intercept_pos":np.array(intercept_pos),
            "intercept_neg":np.array(intercept_neg),
            "slopes_pos_err":np.array(slopes_pos_err),
            "slopes_neg_err":np.array(slopes_neg_err),
            "intercept_pos_err":np.array(intercept_pos_err),
            "intercept_neg_err":np.array(intercept_neg_err)
            }

            if deredden:

                results["median_distance"] = np.array(median_distance),
                results["slopes_pos_dr"] = np.array(slopes_pos_dr)
                results["slopes_neg_dr"] = np.array(slopes_neg_dr)
                results["intercept_pos_dr"] = np.array(intercept_pos_dr)
                results["intercept_neg_dr"] = np.array(intercept_neg_dr)
                results["slopes_pos_dr_err"] = np.array(slopes_pos_dr_err)
                results["slopes_neg_dr_err"] = np.array(slopes_neg_dr_err)
                results["intercept_pos_dr_err"] = np.array(intercept_pos_dr_err)
                results["intercept_neg_dr_err"] = np.array(intercept_neg_dr_err)

    if return_smoothed:
        results["smoothed_longitude"] = np.arange(np.min(median_longitude), 
            np.max(median_longitude), 0.25)
        if deredden:
            distance_interp = interp1d(median_longitude, median_distance)
            results["smoothed_distance"] = distance_interp(results["smoothed_longitude"])
        smoothed_slope_pos_ha = np.zeros((3,len(results["smoothed_longitude"])))
        smoothed_slope_neg_ha = np.zeros((3,len(results["smoothed_longitude"])))
        smoothed_slope_pos_ha_dr = np.zeros((3,len(results["smoothed_longitude"])))
        smoothed_slope_neg_ha_dr = np.zeros((3,len(results["smoothed_longitude"])))
        for ell,lon in enumerate(results["smoothed_longitude"]):
            smoothed_slope_pos_ha[:,ell] = np.nanpercentile(np.array(slopes_pos)[(median_longitude <= lon + smoothed_width.value/2) & 
                                                             (median_longitude > lon - smoothed_width.value/2)], 
                                               (10, 50, 90))
            smoothed_slope_neg_ha[:,ell] = np.nanpercentile(np.array(slopes_neg)[(median_longitude <= lon + smoothed_width.value/2) & 
                                                             (median_longitude > lon - smoothed_width.value/2)], 
                                               (10, 50, 90))
            if deredden:
                smoothed_slope_pos_ha_dr[:,ell] = np.nanpercentile(np.array(slopes_pos_dr)[(median_longitude <= lon + smoothed_width.value/2) & 
                                                                 (median_longitude > lon - smoothed_width.value/2)], 
                                                   (10, 50, 90))
                smoothed_slope_neg_ha_dr[:,ell] = np.nanpercentile(np.array(slopes_neg_dr)[(median_longitude <= lon + smoothed_width.value/2) & 
                                                                 (median_longitude > lon - smoothed_width.value/2)], 
                                                   (10, 50, 90))

        results["smoothed_slopes_pos"] = smoothed_slope_pos_ha
        results["smoothed_slopes_neg"] = smoothed_slope_neg_ha
        if deredden:
            results["smoothed_slopes_pos_dr"] = smoothed_slope_pos_ha_dr
            results["smoothed_slopes_neg_dr"] = smoothed_slope_neg_ha_dr

        

    return results
                              
