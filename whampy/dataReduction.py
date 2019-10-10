import logging

from astropy import units as u
import numpy as np 
import matplotlib.pyplot as plt

from.skySurvey import SkySurvey

from scipy import stats
from astropy.io import fits
import glob
import os.path

def get_calibration_files(directory, lines = None, calibrator = None):
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
        lines = ["ha", "hb", "nii_red", "sii", "oiii", "oi", "hei"]
    elif isinstance(lines, str):
        lines = [lines]
    elif not hasattr(lines, "__iter__"):
        raise TypeError("Invalid lines Type, lines must be a string, list of strings, or None.")
        
    
    # Get all filenames in the target directory
    filename_lines_dict = {}
    for line in lines:
        filename_dict = {}
        for calibrator_name in calibrator:
            files = glob.glob(os.path.join(directory, 
                                           line, 
                                           "combo", 
                                           "{}-*.fts".format(calibrator_name.lower())))
            if len(files) > 0:
                filename_dict[calibrator_name] = files
        if len(filename_dict) > 0:
            filename_lines_dict[line] = filename_dict
    
    
    return filename_lines_dict


def read_calibration_data(directory, lines = None, calibrator = None):
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
    
    """
    file_lists = get_calibration_files(directory, lines = lines, calibrator = calibrator)
    
    data_dict = {}
    for line in file_lists.keys():
        data_dict[line] = {}
        for calibrator_name in file_lists[line].keys():
            try:
                data_dict[line][calibrator_name] = SkySurvey(file_list = file_lists[line][calibrator_name])
            except ValueError:
                logging.warning("No {0}/{1} calibrators have the right fits extension!".format(line, 
                                                                                               calibrator_name))
            
    return data_dict

def compute_transmissions(directory, lines = None, calibrator = None, plot = False, alpha = None):
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
    """
    # Default confidence degree
    if alpha is None:
        alpha = 0.95
    
    # Read data
    data_dict = read_calibration_data(directory, lines = lines, calibrator = calibrator)
    
    
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
                
                siegel_result = stats.siegelslopes(np.log(data_dict[line][calibrator_name]["INTEN"]), 
                                                   data_dict[line][calibrator_name]["AIRMASS"])
                
                transmissions[line][calibrator_name]["SLOPE_SIEGEL"] = siegel_result[0]
                transmissions[line][calibrator_name]["INTERCEPT_SIEGEL"] = siegel_result[1]

                if plot:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.scatter(data_dict[line][calibrator_name]["AIRMASS"], 
                               np.log(data_dict[line][calibrator_name]["INTEN"]), 
                               color = 'k', label = "Observations")
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
    
    