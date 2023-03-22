import pytest
import numpy as np
from ..skySurvey import SkySurvey
from ..skySurvey import directory

from ..scaleHeight import fit_scale_heights

from dustmaps.bayestar import BayestarWebQuery
# Set up the random number generator
np.random.seed(1234)

# Load global survey
survey = SkySurvey()
bayestar = BayestarWebQuery()

data, df, masks = survey.get_scale_height_data(track = 'SgN',
                                            return_pandas_dataframe = True, 
                                            longitude_mask_width = 3, 
                                            lrange = [30, 35])
data_dered, df_dered, masks_dered = survey.get_scale_height_data(track = 'SgN',
                                            return_pandas_dataframe = True, 
                                            longitude_mask_width = 3, 
                                            lrange = [30, 35], deredden = bayestar)

def test_basic():
    """
    Basic test
    """
    results = fit_scale_heights(data, masks, return_smoothed = True)
    results_2 = fit_scale_heights(data_dered, masks_dered, 
        return_smoothed = True, 
        fig_names = "test_fits", 
        smoothed_width = 5, 
        min_lat = 5, 
        max_lat = 35, 
        deredden = True)

    assert np.allclose(results["slopes_pos"], results_2["slopes_pos"], equal_nan = True, atol = 1e-2)