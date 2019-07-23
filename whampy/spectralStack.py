import logging

import numpy as np 

from scipy.interpolate import interp1d

def stack_spectra_bootstrap(survey, 
                              data_column = None, 
                              variance_column = None, 
                              velocity = None, 
                              n_boot = None, 
                              ci = None, 
                              estimator = None, 
                              set_name = None):
    """
    Stack all spectra in provided SkySurvey Table
    
    Parameters
    ----------
    survey: 'whampy.skySurvey'
        input skySurvey
    data_column: 'str', optional, must be keyword
        Name of data column, default of "DATA"
    variance_column: 'str', optional, must be keyword
        Name od variance column, defaut of "VARIANCE"
    velocity: 'np.array, list', optional, must be keyword
        velocity to interpolate data to for stacking
        defaults to np.round of first row "VELOCITY"
    n_boot: 'int', optional, must be keyword
        number of bootstrap samples for each spectrum to stack
        default of 1000
    ci: 'number', optional, must be keyword
        Size of confidence interval to return as errors
    estimator: 'callable', optional, must be keyword
        estimator method to use, defaults to `numpy.mean`
        must take axis keyword
    set_name: 'str', optional, must be keyword
        if provided, sets "NAME" column to this in returned stack
    """
    
    # Set Defaults
    if data_column is None:
        data_column = "DATA"
    if variance_column is None:
        variance_column = "VARIANCE"
    if velocity is None:
        velocity = np.round(survey[0]["VELOCITY"])
    if n_boot is None:
        n_boot = 1000
    if ci is None:
        ci = 95
    if estimator is None:
        estimator = np.mean
        

    data_interp = np.zeros_like(survey[data_column])
    for ell, (velocity_window, 
              data_window) in enumerate(zip(survey["VELOCITY"], 
                                            survey[data_column])):
        f = interp1d(velocity_window, 
                                 data_window, 
                                 fill_value= "extrapolate")
        data_interp[ell,:] = f(velocity)

        
    stacked = survey[0:2]
        
    for ell, (data_window, 
              var_window) in enumerate(zip(survey[data_column], 
                                           survey[variance_column])):
        
        if ell == 0:
            samples_data = [np.sqrt(var_window) * 
                            np.random.randn(len(data_window)) + 
                            data_window for ell in range(n_boot)]
        else:
            samples_data = np.vstack([samples_data, 
                                      [np.sqrt(var_window) * 
                                       np.random.randn(len(data_window)) + 
                                       data_window for ell in range(n_boot)]])
            
    stacked[0]["VELOCITY"] = velocity
    stacked[0]["DATA"] = estimator(samples_data, 
                                axis = 0) * survey["DATA"].unit
    stacked[0]["VARIANCE"] = np.var(samples_data, 
                                 axis = 0) * survey["VARIANCE"].unit
    stacked["CI"] = np.zeros((2, 2,len(velocity)))
    stacked[0]["CI"] = np.percentile(samples_data, 
                                  (100-ci, ci), 
                                  axis = 0) * survey["DATA"].unit

    if set_name is not None:
        stacked["NAME"] = [set_name, " "]
    
    return stacked[0]
    
