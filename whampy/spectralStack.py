import logging

import numpy as np 

from scipy.interpolate import interp1d
from astropy.table import Table


def stack_spectra_bootstrap(survey, 
                              data_column = None, 
                              velocity_column = None,
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
    velocity_column: 'str', optional, must be keyword
        Name of velocity column, default of "VELOCITY"
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
    if velocity_column is None:
        velocity_column = "VELOCITY"
    if velocity is None:
        velocity = survey[0][velocity_column]
    if n_boot is None:
        n_boot = 10000
    if ci is None:
        ci = 95
    if estimator is None:
        estimator = np.nanmean
        

    # data_interp = np.zeros_like(survey[data_column])
    # for ell, (velocity_window, 
    #           data_window) in enumerate(zip(survey["VELOCITY"], 
    #                                         survey[data_column])):
    #     f = interp1d(velocity_window, 
    #                              data_window, 
    #                              fill_value= "extrapolate")
    #     data_interp[ell,:] = f(velocity)

        
    stacked = Table()
    stacked["VELOCITY"] = np.zeros((1, len(velocity)))
    stacked["DATA"] = np.zeros((1, len(velocity)))
    stacked["VARIANCE"] = np.zeros((1, len(velocity)))
    stacked["CI"] = np.zeros((1, 2, len(velocity)))

    data_length = len(survey[data_column][0])

    samples_data = []
    samples_inds = np.arange(len(survey), dtype = int)
    for ell in range(n_boot):
        samples_data_sub = []
        choice_size = len(survey)
        chosen_inds = np.random.choice(samples_inds, size = choice_size)
        # samples_data[ell].append(np.random.choice(survey[data_column][:,ell], size = choice_size))
        samples_data_sub.append(survey[data_column][chosen_inds, :])
        stack_samples_data_sub = np.vstack(samples_data_sub)
        samples_data.append(estimator(stack_samples_data_sub, axis = 0))

    stack_samples_data = np.stack(samples_data)
    
    stacked[0]["VELOCITY"] = velocity
    stacked[0]["DATA"] = estimator(stack_samples_data, 
                                axis = 0) * survey["DATA"].unit
    stacked[0]["VARIANCE"] = np.ma.var(stack_samples_data.data, 
                                 axis = 0) * survey["VARIANCE"].unit
    stacked[0]["CI"] = np.nanpercentile(stack_samples_data.data, 
                                  (100-ci, ci), 
                                  axis = 0) * survey["DATA"].unit


    if set_name is not None:
        stacked["NAME"] = [set_name]
        
    return survey.__class__(from_table = stacked)

def combine_velocity_windows_bootstrap(obs_list, 
                                       survey_spectra = None,
                                       exp_times = None, 
                              data_column = None, 
                              velocity_column = None,
                              variance_column = None, 
                              velocity = None, 
                              n_boot = None, 
                                       n_survey_sample = None,
                              ci = None, 
                              estimator = None, 
                              set_name = None, 
                                       vmin = None, 
                                       vmax = None, 
                                       v_res = None):
    """
    Parameters
    ----------
    obs_list: 'list'
        input SkySurveys as a list
    survey_spectra: `SkySurvey`, optional
        single spectrum from WHAM-SS
        obs variances will be used for random draws
    exp_times: 'list', optional
        list of exposure times for weighting
        NOT YET IMPLEMENTED
    data_column: 'str', optional, must be keyword
        Name of data column, default of "DATA"
    velocity_column: 'str', optional, must be keyword
        Name of velocity column, default of "VELOCITY"
    variance_column: 'str', optional, must be keyword
        Name od variance column, defaut of "VARIANCE"
    velocity: 'np.array, list', optional, must be keyword
        velocity to interpolate data to for stacking
    n_boot: 'int', optional, must be keyword
        number of bootstrap samples for each spectrum to stack
        default of 1000
    n_survey_sample: 'int', optional, must be keyword
        number of bootstrap samples for each spectrum to stack
        default of 1000
    ci: 'number', optional, must be keyword
        Size of confidence interval to return as errors
    estimator: 'callable', optional, must be keyword
        estimator method to use, defaults to `numpy.mean`
        must take axis keyword
    set_name: 'str', optional, must be keyword
        if provided, sets "NAME" column to this in returned stack
    vmin: `number`, optional, must be keyword
        min velocity to consider
    vmax: `number`, optional, must be keyword
        max velocity to consider
    v_res: `number` optional, must be keyword
        velocity resolution / channel width in km/s
    """
    
    if data_column is None:
        data_column = "DATA"
    if variance_column is None:
        variance_column = "VARIANCE"
    if velocity_column is None:
        velocity_column = "VELOCITY"
    if n_boot is None:
        n_boot = 10000
    if ci is None:
        ci = 95
    if estimator is None:
        estimator = np.nanmean
        
    # Ensure obslist is not a single SkySurvey Object
    if hasattr(obs_list, "stack_spectra_bootstrap"):
        raise TypeError("obs_list must be SkySurvey's wrapped in a list!")
        
        
    # Determine number of windows to merge
    n_windows = len(obs_list)
    
    # Determine highest and lowest velocities if needed
    if velocity is None:
        if vmin is None:
            all_vmins = [np.nanmin(obs[velocity_column]) for obs in obs_list]
            vmin = np.nanmin(all_vmins)

        if vmax is None:
            all_vmaxs = [np.nanmax(obs[velocity_column]) for obs in obs_list]
            vmax = np.nanmax(all_vmaxs)

        # Set Velocity Window
        if v_res is None:
            v_res = 1.5
        
        velocity = np.arange(vmin, vmax + v_res, v_res)

    data_length = len(velocity)
        
    # Interpolate spectra:
    interp_data = []
    for obs in obs_list:
        data_this_row = []
        for row in obs:
            data_interpolator = interp1d(row[velocity_column], row[data_column], 
                                         bounds_error = False, fill_value = np.nan)
            data_this_row.append([data_interpolator(velocity)])
        interp_data.append(np.vstack(data_this_row))
        
    if survey_spectra is not None:
        survey_data_interpolator = interp1d(survey_spectra[velocity_column], survey_spectra[data_column], 
                                            bounds_error = False, fill_value = np.nan)
        survey_data = survey_data_interpolator(velocity)
        survey_variance_interpolator = interp1d(survey_spectra[velocity_column], survey_spectra[variance_column], 
                                            bounds_error = False, fill_value = np.nan)
        survey_variance = survey_variance_interpolator(velocity)
        
        # Apply Random Perturbations
        
        if n_survey_sample is None:
            n_survey_sample = 1000
            
        data_this_row = []
        for ell in range(n_survey_sample):
            random_errors = np.sqrt(survey_variance) * 3 * np.random.randn(len(velocity))
            data_random_survey = survey_data + random_errors
            data_this_row.append(data_random_survey)
            
        interp_data.append(np.vstack(data_this_row))
    
    
    stacked = Table()
    stacked["VELOCITY"] = np.zeros((1, len(velocity)))
    stacked["DATA"] = np.zeros((1, len(velocity)))
    stacked["VARIANCE"] = np.zeros((1, len(velocity)))
    stacked["CI"] = np.zeros((1, 2, len(velocity)))
    

    samples_data = []
    for iteration in range(n_boot):
        samples_data_sub = []
        for data in interp_data:
            data_inds = np.arange(len(data), dtype = int)
            choice_size = len(data)
            chosen_inds = np.random.choice(data_inds, size = choice_size)
            samples_data_sub.append(data[chosen_inds])
       
        stack_samples_data_sub = np.vstack(samples_data_sub)
        x = estimator(stack_samples_data_sub, axis = 0)
        samples_data.append(x)
            
            

    stack_samples_data = np.stack(samples_data)

    stacked[0]["VELOCITY"] = velocity
    stacked[0]["DATA"] = estimator(stack_samples_data, 
                                axis = 0) * obs_list[0]["DATA"].unit
    stacked[0]["VARIANCE"] = np.nanvar(stack_samples_data, 
                                 axis = 0) * obs_list[0]["VARIANCE"].unit
    stacked[0]["CI"] = np.nanpercentile(stack_samples_data, 
                                  (100-ci, ci), 
                                  axis = 0) * obs_list[0]["DATA"].unit
    
    if set_name is not None:
        stacked["NAME"] = [set_name]
        
    return obs_list[0].__class__(from_table = stacked)
    
