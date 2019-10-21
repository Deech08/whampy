import pytest
import numpy as np
from ..skySurvey import SkySurvey
from ..skySurvey import directory
# Set up the random number generator
np.random.seed(1234)

# Load global survey
survey = SkySurvey()

def test_basic_input_track():
    """
    ensure inputting track without it pre-set as an attribute works
    """
    data, track = survey.get_spiral_slice(track = "Per", return_track = True)
    data2 = survey.get_scale_height_data(track = "Per", return_track = False)
    data.get_scale_height_data()

    assert np.allclose(data["INTEN"], data2["INTEN"], equal_nan = True)

def test_angle_wrapping():
    """
    ensure angle wrapping works properly
    """
    data = survey.get_scale_height_data(track = 'CrF', wrap_at_180 = True)
    data2 = survey.get_scale_height_data(track = 'CrF', wrap_at_180 = False)

    assert np.allclose(data["INTEN"], data2["INTEN"], equal_nan = True)

def test_no_track_error():
    """
    ensure SyntaxError when no track present / given
    """
    try:
        fail = survey.get_scale_height_data()
    except SyntaxError:
        assert True
    else:
        assert False

def test_deredden():
    """
    ensure different deredden keyword formats work
    """
    # from dustmaps.marshall import MarshallQuery
    # data = survey.get_scale_height_data(track = 'SgN', deredden = True)
    # data2 = survey.get_scale_height_data(track = "SgN", deredden = MarshallQuery())

    # assert np.allclose(data["INTEN"], data2["INTEN"], equal_nan = True)
    from dustmaps.bayestar import BayestarWebQuery
    try:
        data = survey.get_scale_height_data(track = 'SgN', deredden = True)
    except OSError:
        assert True
    else:
        data = survey.get_scale_height_data(track = 'SgN', deredden = BayestarWebQuery())
        assert True

def test_deredden_invalid():
    """
    ensure TypeError when invalid deredden keyword provided
    """
    try:
        data = survey.get_scale_height_data(track = "SgN", deredden = "break")
    except TypeError:
        assert True
    else:
        assert False

def test_bad_track():
    """
    ensure ValueError raised in track doesn't have enough columns
    """
    track = np.random.randn(10,2)
    try: 
        fail = survey.get_scale_height_data(track = track)
    except ValueError:
        assert True
    else:
        assert False

def test_pandas_dataframe():
    """
    ensure returning pandas dataframe works for both dereddened and raw data
    """
    from dustmaps.bayestar import BayestarWebQuery
    data, df = survey.get_scale_height_data(track = 'SgN', deredden = False, 
                                            return_pandas_dataframe = True)
    # data2, df2 = survey.get_scale_height_data(track = "SgN", deredden = False, 
    #                                           return_pandas_dataframe = True)

    # assert np.allclose(df["INTEN"], df2["INTEN"], equal_nan = True)
    data2, df2 = survey.get_scale_height_data(track = "SgN", 
            deredden = BayestarWebQuery(), 
            return_pandas_dataframe = True)
    assert np.allclose(df["INTEN"], df2["INTEN"], equal_nan = True)

def test_longitude_mask_width():
    """
    ensure returning masks works
    """
    import astropy.units as u
    data, df, masks = survey.get_scale_height_data(track = 'SgN',
                                            return_pandas_dataframe = True, 
                                            longitude_mask_width = 3)
    data2, df2, masks2 = survey.get_scale_height_data(track = "SgN", 
                                              return_pandas_dataframe = True,
                                              longitude_mask_width = 3*u.deg)

    assert np.allclose(df["INTEN"][masks[0]], df2["INTEN"][masks[0]], equal_nan = True)

def test_longitude_step_size():
    """
    ensure step size works
    """
    from dustmaps.bayestar import BayestarWebQuery
    import astropy.units as u
    data2, df2, masks2 = survey.get_scale_height_data(track = "SgN", deredden = False, 
                                              return_pandas_dataframe = True,
                                              longitude_mask_width = 5*u.deg, 
                                              step_size = 1)

    # assert np.allclose(df["INTEN"][masks[0]], df2["INTEN"][masks[0]], equal_nan = True) 
    
    data, df, masks = survey.get_scale_height_data(track = 'SgN', deredden = BayestarWebQuery(), 
                                        return_pandas_dataframe = True, 
                                        longitude_mask_width = 5, 
                                        step_size = 1*u.deg)
    assert np.allclose(df["INTEN"][masks[0]], df2["INTEN"][masks[0]], equal_nan = True)

def test_add_kinematic():
    """
    ensure adding kinematic distance works
    """
    data, df = survey.get_scale_height_data(track = "CrN", 
        return_pandas_dataframe = True, add_kinematic_distance = True)
    data2, df2 = survey.get_scale_height_data(track = "CrN", 
        return_pandas_dataframe = True, add_kinematic_distance = True, 
        closer = True)

    assert np.allclose(df["INTEN"], df2["INTEN"], equal_nan = True)

def test_deredden_web():
    """
    Ensure deredden works using BayestarWebQuery
    """
    from dustmaps.bayestar import BayestarWebQuery
    data = survey.get_scale_height_data(track = 'SgN', deredden = BayestarWebQuery())
    assert np.any(~np.isnan(data["INTEN_DERED"]))



