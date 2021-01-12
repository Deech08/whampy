import logging

from astropy import units as u
import numpy as np 
import matplotlib.pyplot as plt

from.skySurvey import SkySurvey

from scipy import stats
from astropy.io import fits
import glob
import os.path

from scipy.interpolate import interp1d
from scipy.optimize import minimize
from unittest import mock

from lmfit import Model, Parameters

from lmfit.model import load_modelresult, ModelResult
from astropy.modeling.models import Gaussian1D

from unittest.mock import Mock


def get_calibration_files(directory, lines = None, calibrator = None, file_suffix = None):
    """
    Gets lists of all calibration files to use
    
    Parameters
    ----------
    
    directory: `str`
        directory with combo spectra
    lines: `str`, listlike, optional, must be keyword
        lines to load
        if None, searches for all default lines
    calibrator: `str`, listlike, optional, must be keyword
        calibrator to load
        if None, searches for all default calibrators
    file_suffix: `str`, optional, must be keyword
        Include a suffix to the file search criteria
    """
    
    # Check calibrators keyword
    if calibrator is None:
        calibrator = ["zeta_oph", "g300_0", "spica_hii", "l_ori", "g194_0"]
    elif isinstance(calibrator, str):
        calibrator = [calibrator]
    elif not hasattr(calibrator, "__iter__"):
        raise TypeError("Invalid calibrator Type, calibrator must be a string, list of strings, or None.")
        
    # Check lines keyword
    if lines is None:
        lines = ["ha", "hb", "nii", "sii", "oiii", "oi", "hei"]
    elif isinstance(lines, str):
        lines = [lines]
    elif not hasattr(lines, "__iter__"):
        raise TypeError("Invalid lines Type, lines must be a string, list of strings, or None.")
        
    
    # Get all filenames in the target directory
    filename_lines_dict = {}
    for line in lines:
        filename_dict = {}
        for calibrator_name in calibrator:
            if file_suffix is None:
                files = glob.glob(os.path.join(directory, 
                                           line, 
                                           "combo", 
                                           "{}-[0-9][0-9][0-9][0-9][0-9][0-9]*.fts".format(calibrator_name.lower())))
            else:
                files = glob.glob(os.path.join(directory, 
                                           line, 
                                           "combo", 
                                           "{}-[0-9][0-9][0-9][0-9][0-9][0-9]{}.fts".format(calibrator_name.lower(), 
                                            file_suffix)))
            
            if len(files) > 0:
                filename_dict[calibrator_name] = files
        if len(filename_dict) > 0:
            filename_lines_dict[line] = filename_dict
    
    
    return filename_lines_dict


def read_calibration_data(directory, lines = None, calibrator = None, extension = None, 
    file_suffix = None):
    """
    Reads in calibration data as a SkySurvey object
    
    Parameters
    ----------
    
    directory: `str`
        directory with combo spectra
    lines: `str`, listlike, optional, must be keyword
        lines to load
        if None, searches for all default lines
    calibrator: `str`, listlike, optional, must be keyword
        calibrator to load
        if None, searches for all default calibrators
    extension: 'str', optional, must be keyword
        extension to load
    file_suffix: `str`, optional, must be keyword
        Include a suffix to the file search criteria
    
    """
    file_lists = get_calibration_files(directory, lines = lines, calibrator = calibrator, 
        file_suffix = file_suffix)

    if extension is None:
        extension = "ATMSUB"
    
    data_dict = {}
    for line in file_lists.keys():
        data_dict[line] = {}
        for calibrator_name in file_lists[line].keys():
            try:
                data_dict[line][calibrator_name] = SkySurvey(unreduced_list = file_lists[line][calibrator_name], 
                    extension = extension)
                data_dict[line][calibrator_name]["INTEN"] = data_dict[line][calibrator_name].moment()
            except ValueError:
                logging.warning("No {0}/{1} calibrators have the right fits extension!".format(line, 
                                                                                               calibrator_name))
            
    return data_dict

def compute_transmissions(directory, lines = None, calibrator = None, plot = False, alpha = None, 
    extension = None, file_suffix = None):
    """
    Computes transmission curves (slopes) for a given directory
    
    Parameters
    ----------
    
    directory: `str`
        directory with combo spectra
    lines: `str`, listlike, optional, must be keyword
        lines to load
        if None, searches for all default lines
    calibrator: `str`, listlike, optional, must be keyword
        calibrator to load
        if None, searches for all default calibrators
    plot: `bool`, optional, must be keyword
        if True, plots all data and slopes
    alpha: float, optional
        Confidence degree between 0 and 1. 
        Default is 95% confidence. 
    extension: 'str', optional, must be keyword
        extension to load
    file_suffix: `str`, optional, must be keyword
        Include a suffix to the file search criteria
    """
    # Default confidence degree
    if alpha is None:
        alpha = 0.95
    
    # Read data
    data_dict = read_calibration_data(directory, lines = lines, calibrator = calibrator, 
        file_suffix = file_suffix, extension = extension)
    
    
    # Compute Slopees
    transmissions = {}
    for line in data_dict.keys():
        transmissions[line] = {}
        for calibrator_name in data_dict[line].keys():
            # Check that multiple data points exist
            if len(data_dict[line][calibrator_name]) > 2:
                transmissions[line][calibrator_name] = {}
                fit_result = stats.theilslopes(np.log(data_dict[line][calibrator_name]["INTEN"]), 
                                               data_dict[line][calibrator_name]["AIRMASS"], 
                                               alpha = alpha)

                transmissions[line][calibrator_name]["SLOPE_THEIL"] = fit_result[0]
                transmissions[line][calibrator_name]["INTERCEPT_THEIL"] = fit_result[1]
                transmissions[line][calibrator_name]["LO_SLOPE_THEIL"] = fit_result[2]
                transmissions[line][calibrator_name]["UP_SLOPE_THEIL"] = fit_result[3]
                
                if hasattr(stats, "siegelslopes"):
                    siegel_result = stats.siegelslopes(np.log(data_dict[line][calibrator_name]["INTEN"]), 
                                                   data_dict[line][calibrator_name]["AIRMASS"])
                else:
                    logging.warning("Installed version of scipy does not have the siegelslopes method in scipy.stats!")
                    siegel_result = np.array([np.nan, np.nan])

                
                transmissions[line][calibrator_name]["SLOPE_SIEGEL"] = siegel_result[0]
                transmissions[line][calibrator_name]["INTERCEPT_SIEGEL"] = siegel_result[1]

                if plot:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    from astropy.time import Time
                    dates = np.round(Time(data_dict[line][calibrator_name]["DATE-OBS"]).jd)
                    ax.scatter(data_dict[line][calibrator_name]["AIRMASS"], 
                               np.log(data_dict[line][calibrator_name]["INTEN"]), 
                               c = dates, label = "Observations")
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    slope_line_x = np.linspace(xlim[0], xlim[1], 10)
                    slope_line_y = fit_result[1] + fit_result[0] * slope_line_x
                    slope_line_y_siegel = siegel_result[1] + siegel_result[0] * slope_line_x
                    ax.plot(slope_line_x, 
                            slope_line_y, 
                            lw = 2, 
                            color = 'r', 
                            ls = "-", 
                            label = "Theil Slope = {:.3f}".format(fit_result[0]))
                    ax.plot(slope_line_x, 
                            slope_line_y_siegel, 
                            lw = 2, 
                            color = 'b', 
                            ls = "--", 
                            label = "Siegel Slope = {0:.3f}".format(siegel_result[0]))
                    
                    
                    ax.set_xlabel("Airmass", fontsize = 12)
                    ax.set_ylabel(r"$/ln$(Intensity/R)", fontsize =12)
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
        
                    ax.legend(fontsize = 12)
                    ax.set_title("{} {}".format(line, calibrator_name), fontsize = 12)
            
    return transmissions

slope = -2.85 * (u.km/u.s)/u.beam

def get_velocity_offset(pressure_a, pressure_a0 = 0 , slope = slope, intercept = 0):
    pressure_a *= u.beam # fake unit
    pressure_a0 *= u.beam
    offset = intercept * u.km/u.s + slope * (pressure_a - pressure_a0)
    return offset

ha_order_dict = {
    "delta_P":np.array([-20., -93., -75., 52., -113.]), 
    "p0":np.array([76.4, 0, 0, 0, 0]), 
    "intercept":np.array([99.314, 110, 100, 521, 521])
}

nii_order_dict = {
    "delta_P":np.array([29.5, -139.5, 99.5, -66, -42]), 
    "p0":np.array([131.8, 131.8, 0, 0, 0]), 
    "intercept":np.array([46.06, 46.06, 629.8, 629.8, 215.53])
}

hb_order_dict = {
    "delta_P":np.array([-13., 7., 65., 74.7]), 
    "p0":np.array([145.33, 0, 0, 0]), 
    "intercept":np.array([55.7, 165.962, 316.475, 514.377])
}

oiii_order_dict = {
    "delta_P":np.array([-60., -4.]), 
    "p0":np.array([108.785, 0.]), 
    "intercept":np.array([50.75, 455.591])
}

sii_order_dict = {
    "delta_P":np.array([-44.3, -115.7, 28.3]), 
    "p0":np.array([85.29, 0., 0.]), 
    "intercept":np.array([47.2, 78.446, 494.030])
}

# WHAM Instrument Profiles
IP_TH_AR = {
    "left":{#Blue
    "mean":[-0.1915317000, 0.0623242000, 0.1292077000], 
    "FWHM":[9.6666269000, 32.0365219000, 16.1042328000], 
    "amp":[0.0285350000, 0.0070548000, 0.0271723000]
    }, 
    "center":{
    "mean":[0.2976723000, -0.0952606000, -0.2024040000], 
    "FWHM":[18.5535583000, 37.1741791000, 9.9602461000], 
    "amp":[0.0201603000, 0.0045628000, 0.0397354000]
    }, 
    "right":{#Red
    "mean":[-0.5142975000, 0.3779373000, 0.1363449000], 
    "FWHM":[10.2697849000, 41.3955574000, 18.1468506000], 
    "amp":[0.0321637000, 0.0054176000, 0.0212081000]
    }
}

def cw(std_component, FWHM_IPs):
    FWHM_component = std_component * (2 * np.sqrt(2 * np.log(2)))
    return [np.sqrt(FWHM_component**2 + FWHM_IP**2) 
            for FWHM_IP in FWHM_IPs]

def ch(amp_component, std_component, amp_IPs, FWHM_IPs):
    convolved_widths = cw(std_component, FWHM_IPs)
    area_component = amp_component * np.sqrt(2 * np.pi) * std_component
    return [area_component * h_ip * w_ip/convoved_width 
            for (h_ip, w_ip, convoved_width) in zip(amp_IPs, FWHM_IPs, convolved_widths)]

def c_mean(mean_component, mean_IPs):
    return [mean_component + mean_IP for mean_IP in mean_IPs]

def c_component(amp_component, mean_component, std_component, 
    IP = "center"):
    amp_IPs = IP_TH_AR[IP]["amp"]
    mean_IPs = IP_TH_AR[IP]["mean"]
    FWHM_IPs = IP_TH_AR[IP]["FWHM"]

    std_convolved = cw(std_component, FWHM_IPs) / (2 * np.sqrt(2 * np.log(2)))
    amp_convolved = ch(amp_component, std_component, amp_IPs, FWHM_IPs)
    mean_convolved = c_mean(mean_component, mean_IPs)
    
    return Gaussian1D(amp_convolved[0], mean_convolved[0], std_convolved[0]) + \
            Gaussian1D(amp_convolved[1], mean_convolved[1], std_convolved[1]) + \
            Gaussian1D(amp_convolved[2], mean_convolved[2], std_convolved[2])
    





def raw_to_geo(row):
    """
    Shift from raw velocity to GEO frame using Pressures and Filter
    
    Parameters
    ----------
    row: `SkySurvey`
        row from SkySurvey
    """
    
    delta_P = row["PAMON"]/10. - row["PBMON"]/10.
    
    if row["FSHNAME"] == "ha":
        closest_order_arg = np.abs(ha_order_dict["delta_P"] - delta_P).argmin()
        #Check Distance
        close_enough = np.abs(delta_P - ha_order_dict["delta_P"][closest_order_arg]) < 10.
        
        if not close_enough:
            raise ValueError("Order cannot be solved manually!")
            
        # Get offset
        offset = get_velocity_offset(row["PAMON"]/10., 
                                     pressure_a0 = ha_order_dict["p0"][closest_order_arg], 
                                     intercept = ha_order_dict["intercept"][closest_order_arg])
        
        geo_velocity = row["VELOCITY"] - offset.value
            
        return geo_velocity

    elif row["FSHNAME"] == "nii_red":
        closest_order_arg = np.abs(nii_order_dict["delta_P"] - delta_P).argmin()

        #Check Distance
        close_enough = np.abs(delta_P - nii_order_dict["delta_P"][closest_order_arg]) < 10.
        
        if not close_enough:
            raise ValueError("Order cannot be solved manually!")
            
        # Get offset
        offset = get_velocity_offset(row["PAMON"]/10., 
                                     pressure_a0 = nii_order_dict["p0"][closest_order_arg], 
                                     intercept = nii_order_dict["intercept"][closest_order_arg])
        
        geo_velocity = row["VELOCITY"] - offset.value
            
        return geo_velocity

    elif row["FSHNAME"] == "hb":
        closest_order_arg = np.abs(hb_order_dict["delta_P"] - delta_P).argmin()
        #Check Distance
        close_enough = np.abs(delta_P - hb_order_dict["delta_P"][closest_order_arg]) < 10.
        
        if not close_enough:
            raise ValueError("Order cannot be solved manually!")
            
        # Get offset
        offset = get_velocity_offset(row["PAMON"]/10., 
                                     pressure_a0 = hb_order_dict["p0"][closest_order_arg], 
                                     intercept = hb_order_dict["intercept"][closest_order_arg], 
                                     slope = -2.78 * (u.km/u.s)/u.beam)
        
        geo_velocity = row["VELOCITY"] - offset.value
            
        return geo_velocity

    elif row["FSHNAME"] == "oiii":
        closest_order_arg = np.abs(oiii_order_dict["delta_P"] - delta_P).argmin()

        #Check Distance
        close_enough = np.abs(delta_P - oiii_order_dict["delta_P"][closest_order_arg]) < 10.
        
        if not close_enough:
            raise ValueError("Order cannot be solved manually!")
            
        # Get offset
        offset = get_velocity_offset(row["PAMON"]/10., 
                                     pressure_a0 = oiii_order_dict["p0"][closest_order_arg], 
                                     intercept = oiii_order_dict["intercept"][closest_order_arg], 
                                     slope = -2.74* (u.km/u.s)/u.beam)
        
        geo_velocity = row["VELOCITY"] - offset.value
            
        return geo_velocity

    elif row["FSHNAME"] == "sii":
        closest_order_arg = np.abs(sii_order_dict["delta_P"] - delta_P).argmin()

        #Check Distance
        close_enough = np.abs(delta_P - sii_order_dict["delta_P"][closest_order_arg]) < 10.
        
        if not close_enough:
            raise ValueError("Order cannot be solved manually!")
            
        # Get offset
        offset = get_velocity_offset(row["PAMON"]/10., 
                                     pressure_a0 = sii_order_dict["p0"][closest_order_arg], 
                                     intercept = sii_order_dict["intercept"][closest_order_arg])
        
        geo_velocity = row["VELOCITY"] - offset.value
            
        return geo_velocity


    else:
        return [np.nan] * len(row["VELOCITY"])
    
def all_raw_to_geo(self):
    """
    Shift from raw velocity to geocentric frame for all spectra
    
    Parameters
    ----------
    

    """
    
    geo_velocities = np.zeros_like(self["VELOCITY"])
    vel_len = len(geo_velocities[0,:])
    for ell,row in enumerate(self):
        try:
            geo = raw_to_geo(row)
        except ValueError:
            geo = [np.nan] * vel_len
        if len(geo) == vel_len:
            geo_velocities[ell,:] = geo
            
    self["VELOCITY_GEO"] = geo_velocities

def all_geo_to_lsr(self, velocity_column = None, vlsr = None):
    """
    Converts velocities from LSR frame to Geocentric Frame
    
    Parmaeters
    ----------
    velocity_column 'st'
        Column key containing GEO velocities
    self: 'whampy.SkySurvey', optional, must be keyword
        SkySurvey with relavent metadata
        must be same length as velocities
    vlsr: 'number' or 'list-like', optional, must be keyword
        LSR velocity offset
        must be same shape as velocities or single number
    """

    if velocity_column is None:
        velocity_column = "VELOCITY_GEO"

    velocities = self[velocity_column]
    
    # check length of velocities

    try:
        vlen = len(velocities)
    except TypeError:
        if isinstance(velocities, float):
            vlen = 1

    

    try:
        assert len(self) == vlen
    except AssertionError:
        try:
            assert hasattr(self, "index")
        except AssertionError:
            raise TypeError("survey must have same length as velocities or have length 1.")
    except TypeError:
        try:
            assert hasattr(self, "index")
        except AssertionError:
            raise TypeError("survey must have same length as velocities or have length 1.")
            
                
    finally:
        if vlsr is None:
            try:
                vlsr = self["VLSR"]
            except KeyError:
                raise ValueError("VLSR not in survey. Must provide vlsr manually!")


    # Check if vlsr provided
    if vlsr is None:
        raise ValueError("VLSR not provided!")

    else:
        if not isinstance(vlsr, float):
            try:
                assert len(vlsr) == vlen
                vlsr_not_list = False
            except AssertionError:
                raise TypeError("vlsr must have the same shape as velocities, or be a float number.")
            except TypeError:
                raise TypeError("vlsr must have the same shape as velocities, or be a float number.")
        else:
            vlsr_not_list = True
    
    if (len(np.shape(velocities)) > 1) & ~(vlsr_not_list):
        lsr_vel = velocities - vlsr[:,None]
    else:  
        lsr_vel = velocities - vlsr

    self["VELOCITY_LSR"] = lsr_vel
                
    

def find_0_point(row, max_ind = 10):
    """
    find vertical offset
    
    Parameters
    ----------
    row: `SkySurvey`
        row from SkySurvey
    max_ind: `int`
        max int to consier for velocity baseline
    """
    
    val_beginning = np.median(row["DATA"][:max_ind])
    return val_beginning

def all_shift_0_point(self, max_ind = 10):
    """
    Shift vertical offsets
    
    Parameters
    ----------
    
    max_ind: `int`
        max int to consier for velocity baseline
    """
    
    for row in self:
        row["DATA"] -= find_0_point(row, max_ind = max_ind)

def fit_geocoronal(target_row, 
                    velocity_column = None, 
                    data_column = None, 
                    IP = "center", **kwargs):
    """
    Fits geocoronal bright line to spectrum for future subtraction

    Parameters
    ----------
    target_row: `SkySurvey` row
        Row to match spectra to
    data_column: 'str', optional, must be keyword
        Name of data column, default of "DATA"
    velocity_column: 'str', optional, must be keyword
        Name of velocity column, default of "VELOCITY"
    IP: `str`, optional, must be keyword
        which Th-Ar instrument profile to use
    **kwargs: dict
        keywords passed to Model.fit()


    """

    if velocity_column is None:
        velocity_column = "VELOCITY_GEO"
    if data_column is None:
        data_column = "DATA"

    def bright_atm(x, baseline, amp, mean, std):
        g = c_component(amp, mean, std, IP = IP)
        
        y = np.zeros_like(x)
        y+= baseline
        y+= g(x)
        
        return y

    bright_atm_model = Model(bright_atm)

    params = Parameters()
    params.add("baseline", value = np.nanmin(target_row[data_column]))
    params.add("amp", value = np.nanmax(target_row[data_column]))
    params.add("mean", value = -2.33, min = -12, max = 8)
    params.add("std", value = 3)

    res = bright_atm_model.fit(target_row[data_column], 
                            x = target_row[velocity_column], 
                            params = params, 
                            nan_policy = "omit", 
                            **kwargs)

    return res

def fit_oxy_ha(target_row, 
                    velocity_column = None, 
                    data_column = None, 
                    IP = "center",
                    **kwargs):
    """
    Fits oxygen bright line to spectrum for future subtraction

    Parameters
    ----------
    target_row: `SkySurvey` row
        Row to match spectra to
    data_column: 'str', optional, must be keyword
        Name of data column, default of "DATA"
    velocity_column: 'str', optional, must be keyword
        Name of velocity column, default of "VELOCITY"
    **kwargs: dict
        keywords passed to Model.fit()


    """

    if velocity_column is None:
        velocity_column = "VELOCITY_GEO"
    if data_column is None:
        data_column = "DATA"

    def bright_atm(x, baseline, amp, mean, std):
        g = c_component(amp, mean, std, IP = IP)
        
        y = np.zeros_like(x)
        y+= baseline
        y+= g(x)
        
        return y

    bright_atm_model = Model(bright_atm)

    params = Parameters()
    params.add("baseline", value = np.nanmin(target_row[data_column]))
    params.add("amp", value = np.nanmax(target_row[data_column]))
    params.add("mean", value = 272.44)
    params.add("std", value = 3)

    res = bright_atm_model.fit(target_row[data_column], 
                            x = target_row[velocity_column], 
                            params = params, 
                            nan_policy = "omit", 
                            **kwargs)

    return res

def fit_oxy_nii(target_row, 
                    velocity_column = None, 
                    data_column = None, 
                    IP = "center",
                    **kwargs):
    """
    Fits oxygen bright line to spectrum for future subtraction

    Parameters
    ----------
    target_row: `SkySurvey` row
        Row to match spectra to
    data_column: 'str', optional, must be keyword
        Name of data column, default of "DATA"
    velocity_column: 'str', optional, must be keyword
        Name of velocity column, default of "VELOCITY"
    **kwargs: dict
        keywords passed to Model.fit()


    """

    if velocity_column is None:
        velocity_column = "VELOCITY_GEO"
    if data_column is None:
        data_column = "DATA"

    def bright_atm(x, baseline, amp, mean, std):
        g = c_component(amp, mean, std, IP = IP)
        
        y = np.zeros_like(x)
        y+= baseline
        y+= g(x)
        
        return y

    bright_atm_model = Model(bright_atm)

    params = Parameters()
    params.add("baseline", value = np.nanmin(target_row[data_column]))
    params.add("amp", value = np.nanmax(target_row[data_column]))
    params.add("mean", value = -281.3)
    params.add("std", value = 3)

    exclusion_mask = (target_row[velocity_column] < -315) | (target_row[velocity_column] > -215)

    res = bright_atm_model.fit(target_row[data_column][np.invert(exclusion_mask)], 
                            x = target_row[velocity_column][np.invert(exclusion_mask)], 
                            params = params, 
                            nan_policy = "omit", 
                            **kwargs)

    return res

def fit_double_oxy_nii(target_row, 
                    velocity_column = None, 
                    data_column = None, 
                    IP = "center",
                    **kwargs):
    """
    Fits oxygen bright lines to spectrum for future subtraction

    Parameters
    ----------
    target_row: `SkySurvey` row
        Row to match spectra to
    data_column: 'str', optional, must be keyword
        Name of data column, default of "DATA"
    velocity_column: 'str', optional, must be keyword
        Name of velocity column, default of "VELOCITY"
    **kwargs: dict
        keywords passed to Model.fit()


    """

    if velocity_column is None:
        velocity_column = "VELOCITY_GEO"
    if data_column is None:
        data_column = "DATA"

    def bright_atm(x, baseline, amp, mean, std):
        g = c_component(amp, mean, std, IP = IP)
        
        y = np.zeros_like(x)
        y+= baseline
        y+= g(x)
        
        return y

    def double_bright_atm(x, baseline, amp, mean, std, 
        fainter_factor, fainter_shift, fainter_std):
        g = c_component(amp, mean, std, IP = IP)
        g2 = c_component(amp/fainter_factor, 
            mean + fainter_shift, 
            fainter_std, IP = IP)
        y = np.zeros_like(x)
        y+= baseline
        y+= g(x)
        y+= g2(x)

        return y


    # bright_atm_model = Model(bright_atm) + Model(bright_atm, prefix = "fainter_")

    bright_atm_model = Model(double_bright_atm)
    params = Parameters()
    params.add("baseline", value = np.nanmin(target_row[data_column]))
    params.add("amp", value = np.nanmax(target_row[data_column]))
    params.add("mean", value = -281.3)
    params.add("std", value = 3)
    params.add("fainter_factor", value = 22, min = 7, max = 37)
    params.add("fainter_shift", value = 32, min = 28, max = 36)
    params.add("fainter_std", value = 3, max = 12, min = 2)
    # params.add("fainter_baseline", value = 0, vary = False)
    # params.add("fainter_amp", value = np.nanmax(target_row[data_column])/20, 
    #     min = np.nanmax(target_row[data_column])/40, 
    #     max = np.nanmax(target_row[data_column])/5)
    # params.add("fainter_mean", value = -281.3 + 32, 
    #         min = -281.3 + 29, max = -281.3 + 35)
    # params.add("fainter_std", value = 3, max = 12, min = 2)

    exclusion_mask = (target_row[velocity_column] < -315) | (target_row[velocity_column] > -215)

    res = bright_atm_model.fit(target_row[data_column][np.invert(exclusion_mask)], 
                            x = target_row[velocity_column][np.invert(exclusion_mask)], 
                            params = params, 
                            nan_policy = "omit", 
                            **kwargs)

    return res

def fit_all_geocoronals(target_data, 
                    velocity_column = None, 
                    data_column = None, 
                    **kwargs):
    """
    Fits geocoronal bright line to spectra for future subtraction

    Parameters
    ----------
    target_row: `SkySurvey` row
        Row to match spectra to
    data_column: 'str', optional, must be keyword
        Name of data column, default of "DATA"
    velocity_column: 'str', optional, must be keyword
        Name of velocity column, default of "VELOCITY"
    **kwargs: dict
        keywords passed to Model.fit()


    """
    if velocity_column is None:
        velocity_column = "VELOCITY_GEO"
    if data_column is None:
        data_column = "DATA"

    # Identify windows with Geocoronal possible
    to_fit_geo = [(np.nanmax(vels) > 5) & (np.nanmin(vels) < -10) 
                    for vels in target_data[velocity_column]]

    # Determine which IP to use:
    left_IP = [np.argmin(np.abs(vels - (-2.33))) < 34 
                for vels in target_data[velocity_column]]
    right_IP = [np.argmin(np.abs(vels - (-2.33))) > 65 
                for vels in target_data[velocity_column]]

    geo_results = []

    for ell,(mask,row) in enumerate(zip(to_fit_geo,target_data)):
        if mask:
            if left_IP[ell]:
                IP = "left"
            elif right_IP[ell]:
                IP = "right"
            else:
                IP = "center"

            geo_results.append(fit_geocoronal(row, 
                                            velocity_column = velocity_column, 
                                            data_column = data_column, 
                                            IP = IP,
                                            **kwargs))
        else:
            geo_results.append(Mock())

    return geo_results

def fit_all_oxy_ha(target_data, 
                    velocity_column = None, 
                    data_column = None, 
                    **kwargs):
    """
    Fits oxygen bright line to spectra for future subtraction

    Parameters
    ----------
    target_row: `SkySurvey` row
        Row to match spectra to
    data_column: 'str', optional, must be keyword
        Name of data column, default of "DATA"
    velocity_column: 'str', optional, must be keyword
        Name of velocity column, default of "VELOCITY"
    **kwargs: dict
        keywords passed to Model.fit()


    """
    if velocity_column is None:
        velocity_column = "VELOCITY_GEO"
    if data_column is None:
        data_column = "DATA"

    # Identify windows with Geocoronal possible
    to_fit_geo = [(np.nanmax(vels) > 275) & (np.nanmin(vels) < 265) 
                    for vels in target_data[velocity_column]]

    # Determine which IP to use:
    left_IP = [np.argmin(np.abs(vels - (272.44))) < 34 
                for vels in target_data[velocity_column]]
    right_IP = [np.argmin(np.abs(vels - (272.44))) > 65 
                for vels in target_data[velocity_column]]

    geo_results = []

    for ell,(mask,row) in enumerate(zip(to_fit_geo,target_data)):
        if mask:
            if left_IP[ell]:
                IP = "left"
            elif right_IP[ell]:
                IP = "right"
            else:
                IP = "center"
            geo_results.append(fit_oxy_ha(row, 
                                            velocity_column = velocity_column, 
                                            data_column = data_column, 
                                            IP = IP,
                                            **kwargs))
        else:
            geo_results.append(Mock())

    return geo_results

def fit_all_oxy_nii(target_data, 
                    velocity_column = None, 
                    data_column = None,
                    **kwargs):
    """
    Fits oxygen bright line to spectra for future subtraction

    Parameters
    ----------
    target_row: `SkySurvey` row
        Row to match spectra to
    data_column: 'str', optional, must be keyword
        Name of data column, default of "DATA"
    velocity_column: 'str', optional, must be keyword
        Name of velocity column, default of "VELOCITY"
    **kwargs: dict
        keywords passed to Model.fit()


    """
    if velocity_column is None:
        velocity_column = "VELOCITY_GEO"
    if data_column is None:
        data_column = "DATA"

    # Identify windows with Geocoronal possible
    to_fit_geo = [(np.nanmax(vels) > -260) & (np.nanmin(vels) < -290) 
                    for vels in target_data[velocity_column]]

    # Determine which IP to use:
    left_IP = [np.argmin(np.abs(vels - (-281.3))) < 34 
                for vels in target_data[velocity_column]]
    right_IP = [np.argmin(np.abs(vels - (-281.3))) > 65 
                for vels in target_data[velocity_column]]

    geo_results = []

    for ell,(mask,row) in enumerate(zip(to_fit_geo,target_data)):
        if mask:
            if left_IP[ell]:
                IP = "left"
            elif right_IP[ell]:
                IP = "right"
            else:
                IP = "center"
            geo_results.append(fit_oxy_nii(row, 
                                            velocity_column = velocity_column, 
                                            data_column = data_column, 
                                            IP  = IP,
                                            **kwargs))
        else:
            geo_results.append(Mock())

    return geo_results

def fit_all_double_oxy_nii(target_data, 
                    velocity_column = None, 
                    data_column = None,
                    **kwargs):
    """
    Fits oxygen bright line to spectra for future subtraction

    Parameters
    ----------
    target_row: `SkySurvey` row
        Row to match spectra to
    data_column: 'str', optional, must be keyword
        Name of data column, default of "DATA"
    velocity_column: 'str', optional, must be keyword
        Name of velocity column, default of "VELOCITY"
    **kwargs: dict
        keywords passed to Model.fit()


    """
    if velocity_column is None:
        velocity_column = "VELOCITY_GEO"
    if data_column is None:
        data_column = "DATA"

    # Identify windows with Geocoronal possible
    to_fit_geo = [(np.nanmax(vels) > -260) & (np.nanmin(vels) < -290) 
                    for vels in target_data[velocity_column]]

    fit_double = [(np.nanmax(vels) > -230) & (np.nanmin(vels) < -290) 
                    for vels in target_data[velocity_column]]

    to_fit_all = np.logical_or(fit_double, to_fit_geo)

    # Determine which IP to use:
    left_IP = [np.argmin(np.abs(vels - (-281.3))) < 34 
                for vels in target_data[velocity_column]]
    right_IP = [np.argmin(np.abs(vels - (-281.3))) > 65 
                for vels in target_data[velocity_column]]

    geo_results = []

    for ell,(mask,row) in enumerate(zip(to_fit_all,target_data)):
        if mask:
            if left_IP[ell]:
                IP = "left"
            elif right_IP[ell]:
                IP = "right"
            else:
                IP = "center"
            if fit_double[ell]:
                geo_results.append(fit_double_oxy_nii(row, 
                                                velocity_column = velocity_column, 
                                                data_column = data_column, 
                                                IP = IP,
                                                **kwargs))
            else:
                geo_results.append(fit_oxy_nii(row, 
                                                velocity_column = velocity_column, 
                                                data_column = data_column, 
                                                IP = IP,
                                                **kwargs))
        
        else:
            geo_results.append(Mock())


    return geo_results

def set_geocentric_zero_from_geocoronal(target_data, geo_results = None,
                                        velocity_column = None, 
                                        data_column = None,
                                        new_velocity_column = None, 
                                        subtract = True,
                                        **kwargs):

    """
    shifts GEO Velocity to be set by geocoronal line

    Parameters
    ----------
    target_row: `SkySurvey` row
        Row to match spectra to
    geo_results: `list` of `lmfit.model.ModelResult
        resutls of fits
        if None, will do fitting
    data_column: 'str', optional, must be keyword
        Name of data column, default of "DATA"
    velocity_column: 'str', optional, must be keyword
        Name of velocity column, default of "VELOCITY_GEO"
    new_velocity_column: 'str', optional, must be keyword
        Name of new Velocity column to set geo velocity to
        default of "VELOCITY_GEO"
    subtract: `bool`, optional, must be keyword
        if True, subtracts bright fit from data
    **kwargs: dict
        keywords passed to Model.fit()
    """
    if velocity_column is None:
        velocity_column = "VELOCITY_GEO"
    if data_column is None:
        data_column = "DATA"
    if new_velocity_column is None:
        new_velocity_column = "VELOCITY_GEO"

    if geo_results is None:
        geo_results = fit_all_geocoronals(target_data, 
            velocity_column = velocity_column, 
            data_column = data_column, **kwargs)

    # Create new velocity column if needed
    if new_velocity_column not in target_data.keys():
        target_data[new_velocity_column] = target_data[velocity_column][:]

    if "allow_jitter" not in target_data.keys():
        target_data["allow_jitter"] = np.ones(len(target_data), dtype = bool)

    for ell, (res, row) in enumerate(zip(geo_results, target_data)):
        if not hasattr(res, "assert_any_call"):
            if res.message.rsplit(".")[0] == 'Fit succeeded':
                offset = res.params["mean"].value - (-2.33)

                if (np.abs(offset) < 12):
                    row[new_velocity_column] = row[velocity_column] - offset
                    row["allow_jitter"] = False
                    if subtract:
                        comp = res.best_fit - res.params["baseline"].value
                        # try:
                        #     assert len(comp) == len(row[data_column])
                        # except AssertionError:
                        #     print("Warning: Shape mis-match for row {}".format(ell))
                        # else:
                        # Check for messed up shape issue
                        if len(comp) != len(row[data_column]):
                            comp = np.concatenate([[np.nan, np.nan, np.nan], comp])
                        row[data_column] -= comp
                else:
                    row[new_velocity_column][:] = np.nan

def set_geocentric_zero_from_oxy_nii(target_data, geo_results = None,
                                        velocity_column = None, 
                                        data_column = None,
                                        variance_column = None,
                                        new_velocity_column = None, 
                                        subtract = True,
                                        use_quadratic_fit = False,
                                        **kwargs):

    """
    shifts GEO Velocity to be set by oxygen line

    Parameters
    ----------
    target_row: `SkySurvey` row
        Row to match spectra to
    geo_results: `list` of `lmfit.model.ModelResult
        resutls of fits
        if None, will do fitting
    data_column: 'str', optional, must be keyword
        Name of data column, default of "DATA"
    velocity_column: 'str', optional, must be keyword
        Name of velocity column, default of "VELOCITY_GEO"
    variance_volumn: 'str', optional, must be keyword
        Name of velocity column, default of "VARIANCE"
    new_velocity_column: 'str', optional, must be keyword
        Name of new Velocity column to set geo velocity to
        default of "VELOCITY_GEO"
    subtract: `bool`, optional, must be keyword
        if True, subtracts bright fit from data
    use_quadratic_fit: `bool`, optional, must be keyword
        if True, uses quadratic fit method to find peak and sets velocity based on peak
        Warning: Cannot subtract a fit using this method, but useful to first do this 
        before fitting for peak
    **kwargs: dict
        keywords passed to Model.fit()
    """
    if velocity_column is None:
        velocity_column = "VELOCITY_GEO"
    if data_column is None:
        data_column = "DATA"
    if variance_column is None:
        variance_column = "VARIANCE"
    if new_velocity_column is None:
        new_velocity_column = "VELOCITY_GEO"

    

    # Create new velocity column if needed
    if new_velocity_column not in target_data.keys():
        target_data[new_velocity_column] = target_data[velocity_column][:]

    if "allow_jitter" not in target_data.keys():
        target_data["allow_jitter"] = np.ones(len(target_data), dtype = bool)

    if use_quadratic_fit:
        quad_fit = target_data.get_quadratic_centroid(data_column = data_column, 
            velocity_column = velocity_column, 
            variance_column = variance_column, 
            window_len = 5)
        for ell, row in enumerate(target_data):
            offset = quad_fit[ell,0] - (-281.3)
            if (np.abs(offset) < 12) & (np.abs(quad_fit[ell,0] - 
                                        np.nanmax(row[velocity_column])) < 5):
                row[new_velocity_column] = row[velocity_column] - offset
                row["allow_jitter"] = False
            else:
                row[new_velocity_column][:] = row[velocity_column][:]
    else:
        if geo_results is None:
            geo_results = fit_all_double_oxy_nii(target_data, 
                velocity_column = velocity_column, 
                data_column = data_column, **kwargs)

        for ell, (res, row) in enumerate(zip(geo_results, target_data)):
            if not hasattr(res, "assert_any_call"):
                if res.message.rsplit(".")[0] == 'Fit succeeded':
                    offset = res.params["mean"].value - (-281.3)

                    if (np.abs(offset) < 12) & (res.params["amp"].value > 1.1):
                        exclusion_mask = (row[velocity_column] < -315) | (row[velocity_column] > -215)
                        row[new_velocity_column] = row[velocity_column] - offset
                        row["allow_jitter"] = False
                        # print(applying new shift)
                        if subtract:
                            comp = res.best_fit - res.params["baseline"].value
                            row[data_column][~exclusion_mask] -= comp
                    else:
                        row[new_velocity_column][:] = np.nan


def set_geocentric_zero_from_oxy_ha(target_data, geo_results = None,
                                        velocity_column = None, 
                                        data_column = None,
                                        variance_column = None,
                                        new_velocity_column = None, 
                                        subtract = True, use_quadratic_fit = False,
                                        **kwargs):

    """
    shifts GEO Velocity to be set by oxygen line

    Parameters
    ----------
    target_row: `SkySurvey` row
        Row to match spectra to
    geo_results: `list` of `lmfit.model.ModelResult
        resutls of fits
        if None, will do fitting
    data_column: 'str', optional, must be keyword
        Name of data column, default of "DATA"
    velocity_column: 'str', optional, must be keyword
        Name of velocity column, default of "VELOCITY_GEO"
    variance_volumn: 'str', optional, must be keyword
        Name of velocity column, default of "VARIANCE"
    new_velocity_column: 'str', optional, must be keyword
        Name of new Velocity column to set geo velocity to
        default of "VELOCITY_GEO"
    subtract: `bool`, optional, must be keyword
        if True, subtracts bright fit from data
    use_quadratic_fit: `bool`, optional, must be keyword
        if True, uses quadratic fit method to find peak and sets velocity based on peak
        Warning: Cannot subtract a fit using this method, but useful to first do this 
        before fitting for peak
    **kwargs: dict
        keywords passed to Model.fit()
    """
    if velocity_column is None:
        velocity_column = "VELOCITY_GEO"
    if data_column is None:
        data_column = "DATA"
    if variance_column is None:
        variance_column = "VARIANCE"
    if new_velocity_column is None:
        new_velocity_column = "VELOCITY_GEO"

    # Create new velocity column if needed
    if new_velocity_column not in target_data.keys():
        target_data[new_velocity_column] = target_data[velocity_column][:]

    if "allow_jitter" not in target_data.keys():
        target_data["allow_jitter"] = np.ones(len(target_data), dtype = bool)

    if use_quadratic_fit:
        quad_fit = target_data.get_quadratic_centroid(data_column = data_column, 
            velocity_column = velocity_column, 
            variance_column = variance_column, 
            window_len = 5)
        
        for ell, row in enumerate(target_data):
            offset = quad_fit[ell,0] - 272.44
            if (np.abs(offset) < 12) & (np.abs(quad_fit[ell,0] - 
                                        np.nanmax(row[velocity_column])) > 5):

                row[new_velocity_column] = row[velocity_column] - offset
                row["allow_jitter"] = False
            else:
                row[new_velocity_column][:] = row[velocity_column][:]
    else:
        if geo_results is None:
            geo_results = fit_all_oxy_ha(target_data, 
                velocity_column = velocity_column, 
                data_column = data_column, **kwargs)
        for ell, (res, row) in enumerate(zip(geo_results, target_data)):
            if not hasattr(res, "assert_any_call"):
                if res.message.rsplit(".")[0] == 'Fit succeeded':
                    offset = res.params["mean"].value - (272.44)

                    if (np.abs(offset) < 12) & (res.params["amp"].value > 1.1):
                        row[new_velocity_column] = row[velocity_column] - offset
                        row["allow_jitter"] = False
                        if subtract:
                            comp = res.best_fit - res.params["baseline"].value
                            # bright = 
                            # bright_g = Gaussian1D(res.params["amp"a].value, 
                            #     res.params["mean"].value, 
                            #     res.params["std"].value)
                            row[data_column] -= comp
                    else:
                        row[new_velocity_column][:] = np.nan






def scale_spectrum(target_row, to_scale_row, 
                   velocity_column = None, 
                   data_column = None, 
                   variance_column = None, 
                   mask_geocoronal = True, 
                   mask_oxygen = True, 
                   **kwargs):
    """
    Find scaling factor to match amplitudes of spectra
    
    target_row: `SkySurvey` row
        Row to match spectra to
    to_scale_row: `SkySurvey` row
        Row to scale 
    data_column: 'str', optional, must be keyword
        Name of data column, default of "DATA"
    velocity_column: 'str', optional, must be keyword
        Name of velocity column, default of "VELOCITY"
    variance_column: 'str', optional, must be keyword
        Name od variance column, defaut of "VARIANCE"
    mask_geocoronal: `bool`
        if True, masks velocities associated with Geocoronal Atm line
    mask_oxygen: `bool`
        if True, masks velocities associated with bright oxygen Atm line

    """
    if data_column is None:
        data_column = "DATA"
    if variance_column is None:
        variance_column = "VARIANCE"
    if velocity_column is None:
        velocity_column = "VELOCITY_GEO"
        
    data_interpolator = interp1d(to_scale_row[velocity_column], to_scale_row[data_column], 
                                         bounds_error = False, fill_value = np.nan)

    #interpolate to same velocity frame
    to_scale_interped_data = data_interpolator(target_row[velocity_column])

    mask = np.isnan(to_scale_interped_data)
    if target_row["FSHNAME"] == "ha":
        if mask_geocoronal:
            mask |= ((target_row[velocity_column] <15) & (target_row[velocity_column] > -20))
        if mask_oxygen:
            mask |= ((target_row[velocity_column] < 285) & (target_row[velocity_column] > 252))
    elif target_row["FSHNAME"] == "nii_red":
        if mask_oxygen:
            mask |= ((target_row[velocity_column] < -260) & (target_row[velocity_column] > -300))
    
    to_scale_masked_data = np.ma.masked_array(data = to_scale_interped_data,
                                              mask = mask)

    
    if np.all(to_scale_masked_data.mask):
        res = mock.Mock()
        res.params = Parameters()
        res.params.add("baseline", value = np.nan)
        res.params.add("factor", value = np.nan)
        return res
    else:


        bad_vel_mask = np.isnan(target_row[velocity_column])

        target_data_masked = np.ma.masked_array(data = target_row[data_column],
                                                mask = np.isnan(target_row[data_column]) & 
                                                bad_vel_mask)
        target_variance_masked = np.ma.masked_array(data = target_row[variance_column],
                                                mask = np.isnan(target_row[variance_column]) & 
                                                bad_vel_mask)

        def scaled_model(x, baseline, factor, jitter):
            to_scale_interped_data = data_interpolator(x - jitter)

            mask = np.isnan(to_scale_interped_data)
            if target_row["FSHNAME"] == "ha":
                if mask_geocoronal:
                    mask |= ((x - jitter < 15) & (x - jitter > -20))
                if mask_oxygen:
                    mask |= ((x - jitter < 285) & (x - jitter > 252))
            elif target_row["FSHNAME"] == "nii_red":
                if mask_oxygen:
                    mask |= ((x - jitter< -260) & (x - jitter > -300))
            
            to_scale_masked_data = np.ma.masked_array(data = to_scale_interped_data,
                                              mask = mask)
            return np.zeros_like(x) + baseline + (factor * to_scale_masked_data)


        sm = Model(scaled_model)

        scale_guess = np.nanmedian(target_data_masked[(~to_scale_masked_data.mask) & 
                                                        (~target_data_masked.mask)]) / \
                    (np.nanmedian(to_scale_masked_data[(~to_scale_masked_data.mask) & 
                                                                            (~target_data_masked.mask)]) - 
                    np.ma.min(target_data_masked))


        params = Parameters()
        params.add("baseline", value = 0.0)
        params.add("factor", value = 1.0, min = 0)
        params.add("jitter", value = 0.0, min = -2, max = 2)
        try:
            vary = to_scale_row["allow_jitter"]
        except KeyError:
            vary = True

        params["jitter"].vary = vary

        res = sm.fit(target_data_masked, 
            x = target_row[velocity_column], 
            params = params, 
            nan_policy = "omit",
            **kwargs)


        # res = minimize(scaled_model, [0, 1.], options={"disp":False}, method = "Nelder-Mead")

        return res
    

def spectrum_model(target_row, atm_template, 
                   ncomp = 4,
                   velocity_column = None, 
                   data_column = None, 
                   variance_column = None, 
                   guess = None, 
                   std_bounds = None, 
                   mean_bounds = None, 
                   atm_jitter_bounds = None, 
                   params = None, 
                   auto_baseline = True, 
                   reset_baseline = False, 
                   reset_baseline_bounds = False, 
                   set_baseline_bounds = None, 
                   mask_vel_zone = None,
                   **kwargs):
    """
    Fit atmosphere to spectrum
    
    target_row: `SkySurvey` row
        Row to match spectra to
    atm_template: `astropy.table.Table` row
        Row to scale 
    ncomp: `number`, optional, must be keyword
        number of gaussian components to fit
    data_column: 'str', optional, must be keyword
        Name of data column, default of "DATA"
    velocity_column: 'str', optional, must be keyword
        Name of velocity column, default of "VELOCITY"
    variance_column: 'str', optional, must be keyword
        Name od variance column, defaut of "VARIANCE"
    guess: `list-like`, optional, must be keyword
        array of guess parameters
    std_bounds: `list-like`, optional, must be keyword
        bounds for line widths
    mean_bounds: `list-like`, optional, must be keyword
        bounds for line means
    params: `lmfit.Parameters`, optional, must be keyword
        initial guess parameters
        can include constraints/bounds
    atm_jitter_bounds: `list-like`, optional, must be keyword
        bounds for atm_jitter parameter
    auto_baseline: `bool`, optional, must be keyword
        Automatically sets baseline guess and bounds
    reset_baseline: `bool`, optional, must be keyword
        reset baseline parameter provided based on data percentiles
    reset_baseline_bounds: `bool`, optional, must be keyword
        reset bounds for baseline parameter based on data percentiles
    set_baseline_bounds: `list-like`, optional, must be keyword
        [min,max] values for baseline parameter
    mask_vel_zone: `list-like`, optional, must be keyword
        [min, max] velocity to fit within
    **kwargs:
        passed to lmfit fitting function

    """
    if data_column is None:
        data_column = "DATA"
    if variance_column is None:
        variance_column = "VARIANCE"
    if velocity_column is None:
        velocity_column = "VELOCITY_GEO"
        
    data_interpolator = interp1d(atm_template["VELOCITY"], atm_template["DATA"], 
                                         bounds_error = False, fill_value = np.nan)
    variance_interpolator = interp1d(atm_template["VELOCITY"], atm_template["VARIANCE"], 
                                         bounds_error = False, fill_value = np.nan)

    # transfer nans


    # #interpolate to same velocity frame
    atm_interped_data = data_interpolator(target_row[velocity_column])
    # atm_interped_variance = variance_interpolator(target_row[velocity_column])

    
    # atm_masked_data = np.ma.masked_array(data = atm_interped_data,
    #                                           mask = np.isnan(atm_interped_data))
    # atm_masked_variance = np.ma.masked_array(data = atm_interped_variance,
    #                                           mask = np.isnan(atm_interped_data))
    


    bad_vel_mask = np.isnan(target_row[velocity_column])
    bad_vel_mask |= np.isnan(atm_interped_data)
    if mask_vel_zone is not None:
        bad_vel_mask |= ((target_row[velocity_column] < mask_vel_zone[1]) & 
            (target_row[velocity_column] > mask_vel_zone[0]))

    target_data_masked = np.ma.masked_array(data = target_row[data_column],
                                            mask = np.isnan(target_row[data_column]) | 
                                            bad_vel_mask)
    target_variance_masked = np.ma.masked_array(data = target_row[variance_column],
                                            mask = np.isnan(target_row[variance_column]) | 
                                            bad_vel_mask)

    if "weights" in kwargs:
        kwargs["weights"] = np.ma.masked_array(data = kwargs["weights"], 
            mask = np.isnan(kwargs["weights"]) | bad_vel_mask)
    
    if guess is None:
        means_guess = np.linspace(np.nanmin(target_row[velocity_column])+10, 
                       np.nanmax(target_row[velocity_column])-10, 4)
        std_guess = 30/(2 * np.sqrt(2 * np.log(2)))
        guess = np.array([np.ma.min(target_data_masked), 10., 0,
                 1., means_guess[0], std_guess, 
                 1., means_guess[1], std_guess, 
                 1., means_guess[2], std_guess,
                 1., means_guess[3], std_guess, 
                 1., means_guess[3], std_guess, 
                 1., means_guess[3], std_guess], dtype = np.float64)





    if std_bounds is None:
        std_bounds = np.float64([22, 37]) / (2 * np.sqrt(2 * np.log(2)))

    if mean_bounds is None:
        mean_bounds = np.float64([np.nanmin(target_row[velocity_column]) - 10, 
                               np.nanmax(target_row[velocity_column]) + 10])
    if atm_jitter_bounds is None:
        atm_jitter_bounds = np.float64([-2, 2])


    
    
    if params is None:
        params = Parameters()
        params.add("ncomp", value = np.float64(ncomp), vary = False)
        min_err = np.float64(np.percentile(target_data_masked.data[~target_data_masked.mask], 5))
        if not auto_baseline:
            params.add("baseline", value = guess[0], 
                min = np.float64(np.ma.min(target_data_masked) - 
                        (min_err - np.ma.min(target_data_masked))), 
                max = np.float64(np.ma.median(target_data_masked)))
        else:
            params.add("baseline", 
                value = np.float64(np.ma.min(target_data_masked)), 
                min = np.float64(np.ma.min(target_data_masked) - (min_err - 
                        np.ma.min(target_data_masked))),
                max = np.float64(np.ma.median(target_data_masked)))
        params.add("atm", value = guess[1], min = np.float64(0.1))
        params.add("atm_jitter", 
            value = guess[2], 
            min = atm_jitter_bounds[0], 
            max = atm_jitter_bounds[1])
        for ell in range(ncomp):
            ell2 = ell*3 + 3
            params.add("f{}_amp".format(ell), 
                value = guess[ell2], 
                min = np.float64(0))
            params.add("f{}_mean".format(ell), 
                value = guess[ell2+1], 
                min = mean_bounds[0], 
                max = mean_bounds[1])
            params.add("f{}_std".format(ell), 
                value = guess[ell2+2], 
                min = std_bounds[0], 
                max = std_bounds[1])

    if np.sum(~target_data_masked.mask) == 0:
        return Mock()


    if reset_baseline:
        params["baseline"].value = np.float64(np.percentile(target_data_masked.data[~target_data_masked.mask], 5))
    if reset_baseline_bounds:
        min_err = np.float64(np.percentile(target_data_masked.data[~target_data_masked.mask], 5))
        params["baseline"].min = np.float64(np.percentile(target_data_masked.data[~target_data_masked.mask], 1))
        params["baseline"].max = np.float64(np.percentile(target_data_masked.data[~target_data_masked.mask], 10))
    if set_baseline_bounds is not None:
        params["baseline"].min = set_baseline_bounds[0]
        params["baseline"].max = set_baseline_bounds[1]


    def baseline_model(x, baseline):
        return np.zeros_like(x) + baseline
        
    def atm_model(x, atm, atm_jitter):
        #interpolate to same velocity frame
        atm_interped_data = data_interpolator(x + atm_jitter)

        atm_masked_data = np.ma.masked_array(data = atm_interped_data,
                                                  mask = np.isnan(atm_interped_data))

        atm_masked_data -= np.ma.min(atm_masked_data)

        return np.zeros_like(x) + atm * atm_masked_data
    
    def gaussian_model(x, amp, mean, std):
        g = c_component(amp, mean, std)
        return g(x)
    
    fitting_model = Model(baseline_model) + Model(atm_model)
    
    
    for ell in range(ncomp):
        fitting_model += Model(gaussian_model, prefix = "f{}_".format(ell))

    

    
    return fitting_model.fit(target_data_masked, 
                             x = target_row[velocity_column], 
                             params = params, 
                             nan_policy = "omit", 
                             ncomp = ncomp, 
                             **kwargs)

    
    
def apply_atmsub(row, model_result, subtract_gaussian = None, atm_temp_in = None):
    """
    Apply ATM subtractiton based on provided model results

    Parameters
    ----------

    row: `SkySurvey`
        row from SkySurvey
    model_result: `lmfit.model.ModelResult`, `str`
        ModelResult from lmfit or filename to load result
    subtract_gaussian: `list-like`, optional, must be keyword
        list of guassian fit components to subtract
    """

    if type(model_result) is not ModelResult:
        assert isinstance(model_result,str)
        model_result = load_modelresult(model_result)

    model_components = model_result.eval_components()
    atm_component = model_components["atm_model"]
    atm = model_result.params["atm"].value
    baseline_component = model_components["baseline_model"]
    stderr = model_result.params["atm"].stderr
    gaussian_component = np.zeros_like(baseline_component)
    if subtract_gaussian is not None:
        for ell in subtract_gaussian:
            gaussian_component += model_components["f{}_".format(ell)]
    if stderr is not None:
        atm_temp = atm_component / atm
        atm_upper = atm_temp * (atm + stderr)
        atm_lower = atm_temp * (atm - stderr)
        both_errs = np.vstack([atm_upper - atm_component, atm_component - atm_lower])
        atm_errs = np.ma.median(both_errs, axis = 0)



    subtracted_spectrum = row["DATA"] - atm_component - baseline_component - gaussian_component
    if stderr is not None:
        spectrum_error = np.sqrt(row["VARIANCE"]) + atm_errs
    else:
        spectrum_error = np.sqrt(row["VARIANCE"])

    if atm_temp_in is not None:
        atm_temp_var_inter = interp1d(atm_temp["VELOCITY"], atm_temp["VARIANCE"], bounds_error = False)
        atm_temp_var = atm_temp_var_inter(row["VELOCITY_GEO"])
        spectrum_error = np.sqrt(spectrum_error**2 + atm_temp_var*atm**2)

    return subtracted_spectrum, spectrum_error **2

def all_apply_atmsub(self, model_results, subtract_gaussian = None, atm_temp = None):
    """
    Apply ATM subtractiton based on provided model results

    Parameters
    ----------

    model_results: `listlike` of `lmfit.model.ModelResult`, `str`
        ModelResult from lmfit or filename to load result
    subtract_gaussian: `list-like`, optional, must be keyword
        list of guassian fit components to subtract
    """

    assert len(model_results) == len(self)


    self["DATA_ATMSUB"] = np.zeros_like(self["DATA"])
    self["VARIANCE_ATMSUB"] = np.zeros_like(self["DATA"])
    self["SUBGAUSS"] = np.zeros(len(self), dtype = int)
    self["SUBG_WCH"] = np.zeros((len(self),10), dtype = int)

    for row, model_result in zip(self, model_results):
        if model_result != "none":

            spec, var = apply_atmsub(row, model_result, subtract_gaussian = subtract_gaussian, 
                atm_temp = atm_temp)
            row["DATA_ATMSUB"] = spec 
            row["VARIANCE_ATMSUB"] = var
            if subtract_gaussian is not None:
                row["SUBGAUSS"] = 1
                for ell in subtract_gaussian:
                    row["SUBG_WCH"][ell] = 1

def prep_atmsub_extension(row, fit_results, 
    extension = None, 
    data_column = None, 
    velocity_column = None, 
    variance_column = None):
    """
    Prepare Header and Data FITS records for saving

    Parameters
    ----------

    fit_results: `lmfit.model.ModelResults', 'str'
        fit results or string to fit results saved file
    extension: `str`, optional, must be keyword
        FITS Extension name to use
    data_column: 'str', optional, must be keyword
        Name of data column, default of "DATA_ATMSUB"
    velocity_column: 'str', optional, must be keyword
        Name of velocity column, default of "VELOCITY_LSR"
    variance_column: 'str', optional, must be keyword
        Name of variance column, defaut of "VARIANCE_ATMSUB"
    Returns
    -------
    header
    data
    """
    # Check Keywords
    if extension is None:
        extension = "ATMSUBPY"

    if data_column is None:
        data_column = "DATA_ATMSUB"
    if variance_column is None:
        variance_column = "VARIANCE_ATMSUB"
    if velocity_column is None:
        velocity_column = "VELOCITY_LSR"

    if isinstance(fit_results, str):
        fit_results = load_modelresult(fit_results)

    # Prep Data (the easy part)
    ra = np.recarray(len(row[data_column]), 
                 formats = [">f8", ">f8", ">f8"], 
                 names = ["VELOCITY", "DATA", "VARIANCE"])
    ra["DATA"] = row[data_column]
    ra["VELOCITY"] = row[velocity_column]
    ra["VARIANCE"] = row[variance_column]


    header = fits.header.Header()

    # Add rows 
    header["XTENSION"] = ("BINTABLE", "binary table extension")
    header["BITPIX"] = (8, "8-bit bytes")
    header["NAXIS"] = (2, "2-dimensional binary table")
    header["NAXIS1"] = (24, "width of table in bytes")
    header["NAXIS2"] = (len(row[data_column]), "number of rows in table")
    header["PCOUNT"] = (0, "size of special data area")
    header["GCOUNT"] = (1, "one data group (required keyword")
    header["TFIELDS"] = (3, "number of fields in each row")
    header["TTYPE1"] = ("VELOCITY", "label for field 1")
    header["TFORM1"] = ("1D", "data format of field: 8-byte DOUBLE")
    header["TUNIT1"] = ("KM/S", "physical unit of field")
    header["TTYPE2"] = ("DATA", "label for field 2")
    header["TFORM2"] = ("1D", "data format of field: 8-byte DOUBLE")
    header["TUNIT2"] = ("ADU", "physical unit of field")
    header["TTYPE3"] = ("VARIANCE", "label for field 3")
    header["TFORM3"] = ("1D", "data format of field: 8-byte DOUBLE")
    header["TUNIT3"] = ("ADU^2", "physical unit of field")
    header["EXTNAME"] = (extension, "name of this binary table extension")
    header["VELFRAME"] = ("LSR", "Reference frame of the velocity vector")
    header["DATAFIT"] = ("DATA", "Fit subtracted data or the fit itself saved?")
    header["SUBFIT"] = (0, "Entire fit subtracted/saved")
    header["SUBBACK"] = (1, "Background Poly subtracted/saved?")
    header["SUBGAUSS"] = (row["SUBGAUSS"], "Some Gaussians subtracted/saved?")
    if row["SUBGAUSS"]:
        which = ["{}".format(ell) for ell in range(10) if row["SUBG_WCH"][ell]]
        header["SUBG_WCH"] = (str(which), "Which Gaussians are subtracted/saved")
    else:
        header["SUBG_WCH"] = ("", "Which Gaussians are subtracted/saved?")
    header["SUBATM"] = (1, "Atmospheric template subtracted/saved?")
    header["ORIGEXT"] = ("PROCSPEC", "Original extension where this fit is valid")
    if row["FSHNAME"] == "ha":
        header["FF"] = ("HA_FLAT", "Flat field (applied to this extension)")
    elif row["FSHNAME"] == "nii_red":
        header["FF"] = ("NII_FLAT", "Flat field (applied to this extension)")
    else:
        header["FF"] = ("HA_FLAT", "Flat field (applied to this extension)")
    header["CHI2"] = (fit_results.redchi, "Reduced Chi-squared of fit")
    header["IP"] = ('CTIO - [O I] Center', "instrument profile")
    header["RSTART1"] = (np.nanmin(row["VELOCITY_GEO"]), "Start of a region used for fitting")
    header["REND1"] = (np.nanmax(row["VELOCITY_GEO"]), "Start of a region used for fitting")
    header["BORDER"] = (0, "Polynomial order of background")
    header["NGAUSS"] = (fit_results.params["ncomp"].value, "Number of Gaussians used")
    header["BKG1"] = (fit_results.params["baseline"].value, "Background coefficient")
    header["BGKSD1"] = (fit_results.params["baseline"].stderr, "Backgrorund coefficient error")
    ncomp = int(fit_results.params["ncomp"].value)
    for ell in range(9):
        if ell < ncomp:
            header["MEAN{}".format(ell+1)] = (fit_results.params["f{}_mean".format(ell)].value, 
                "Gaussian Mean")
            header["MEANSD{}".format(ell+1)] = (fit_results.params["f{}_mean".format(ell)].stderr, 
                "Gaussian Mean error")
            header["WIDTH{}".format(ell+1)] = (fit_results.params["f{}_std".format(ell)].value * \
                2 * np.sqrt(2 * np.log(2)), 
                "Gaussian FWHM")
            if fit_results.params["f{}_std".format(ell)].stderr is not None:
                header["WIDTHSD{}".format(ell+1)] = (fit_results.params["f{}_std".format(ell)].stderr * \
                    2 * np.sqrt(2 * np.log(2)),  
                    "Gaussian FWHM error")
            else:
                header["WIDTHSD{}".format(ell+1)] = (fit_results.params["f{}_std".format(ell)].stderr,  
                    "Gaussian FWHM error")
            area = fit_results.params["f{}_amp".format(ell)].value * np.sqrt(np.pi * 2) 
            area *= fit_results.params["f{}_std".format(ell)].value
            if (fit_results.params["f{}_amp".format(ell)].stderr is not None) & (fit_results.params["f{}_std".format(ell)].stderr is not None):
                areasd = fit_results.params["f{}_amp".format(ell)].stderr * np.sqrt(np.pi * 2) 
                areasd *= fit_results.params["f{}_std".format(ell)].stderr
            else:
                areasd = None
            header["AREA{}".format(ell+1)] = (area, 
                "Gaussian Area")
            header["AREASD{}".format(ell+1)] = (areasd,  
                "Gaussian Area error")
        else:
            header["MEAN{}".format(ell+1)] = (None, "Gaussian Mean")
            header["MEANSD{}".format(ell+1)] = (None, "Gaussian Mean eror")
            header["WIDTH{}".format(ell+1)] = (None, "Gaussian FWHM")
            header["WIDTHSD{}".format(ell+1)] = (None, "Gaussian FWHM eror")
            header["AREA{}".format(ell+1)] = (None, "Gaussian Area")
            header["AREASD{}".format(ell+1)] = (None, "Gaussian Area eror")

    header["ATMOS"] = (fit_results.params["atm"].value, "Atmospheric Level")
    header["ATMOSSD"] = (fit_results.params["atm"].stderr, "Atmospheric Level error")
    if row["FSHNAME"] == "ha":
        header["ATEMP"] = ("HA_ATMOSPHERE_FULL_BRIGHTFIT.fits", "Atmospheric Template")
    elif row["FSHNAME"] == "nii_red":
        header["ATEMP"] = ("NII_ATMOSPHERE_FULL_BRIGHTFIT.fits", "Atmospheric Template")

    header["ATMJIT"] = (fit_results.params["atm_jitter"].value, "Atmospheric Jitter")
    header["ATMJITSD"] = (fit_results.params["atm_jitter"].stderr, "Atmospheric Jitter error")

    header.add_comment("########## FIT PARAMETERS ##########", before = "FF")

    header.add_comment("########## Reduced using whampy ##########", before = "FF")

    return header, ra

def save_atmsub(row, fit_results, filename, 
    extension = None, 
    data_column = None, 
    velocity_column = None, 
    variance_column = None, 
    new_file = False, 
    new_filename = None):
    """
    Save ATMSUB fit results to file

    Parameters
    ----------
    fit_results: `lmfit.model.ModelResults', 'str'
        fit results or string to fit results saved file
    filename: `str`
        name/path to original data file
    extension: `str`, optional, must be keyword
        FITS Extension name to use
    data_column: 'str', optional, must be keyword
        Name of data column, default of "DATA_ATMSUB"
    velocity_column: 'str', optional, must be keyword
        Name of velocity column, default of "VELOCITY_LSR"
    variance_column: 'str', optional, must be keyword
        Name of variance column, defaut of "VARIANCE_ATMSUB"
    new_file: `bool`, optional must be keyword
        if True, writes atmsub extension to a new file
        if Ture, must providew new_filename
    new_filename: `str`, optional must be keyword
        filename to write new extension data to
        only if new_file = True
    """

    # Check Keywords
    if extension is None:
        extension = "ATMSUBPY"

    if data_column is None:
        data_column = "DATA_ATMSUB"
    if variance_column is None:
        variance_column = "VARIANCE_ATMSUB"
    if velocity_column is None:
        velocity_column = "VELOCITY_LSR"


    header, ra = prep_atmsub_extension(row, 
        fit_results, 
        extension = extension,
        data_column = data_column,
        velocity_column = velocity_column, 
        variance_column = variance_column)

    if new_file:
        assert new_filename is not None

        with fits.open(filename) as f:
            hdulist = f
            t = fits.BinTableHDU()
            t.data = ra
            t.header = header
            t.name = extension
            hdulist.append(t)
            hdulist.writeto(new_filename)


    else:
        with fits.open(filename, mode = "update") as hdulist:
            t = fits.BinTableHDU()
            t.data = ra
            t.header = header
            t.name = extension
            hdulist.append(t)
            hdulist.flush()






    
                   



    








    
    