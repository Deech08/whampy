import pytest
import numpy as np
from ..skySurvey import SkySurvey
from ..skySurvey import directory

# Set up the random number generator
np.random.seed(1234)

# Load global survey
survey = SkySurvey()

def test_get_spiral_slice():
    """
    basic spiral slice test
    """
    spiral_arm = survey.get_spiral_slice(track = "perseus")
    spiral_arm2 = survey.get_spiral_slice(track = "Per")

    assert np.allclose(spiral_arm["INTEN"], spiral_arm2["INTEN"], equal_nan = True)

def test_get_spiral_slice_by_file():
    """
    load custom data file with less than 6 columns
    """
    import os.path
    filename = os.path.join(directory, "tests", "test_data", "test_lbv.dat")
    spiral_arm = survey.get_spiral_slice(filename = filename)

    track = np.array([np.arange(10)+1, np.arange(10)+1, np.arange(10)+1]).T
    spiral_arm2 = survey.get_spiral_slice(track = track)

    assert np.allclose(spiral_arm["INTEN"], spiral_arm2["INTEN"], equal_nan = True)

def test_custom_lon_lat_range():
    """
    ensure custom bounds work
    """
    import astropy.units as u
    lrange = [-50,-40]
    brange = [-10,10]
    spiral_arm = survey.get_spiral_slice(track = "Carina_far", 
        lrange = lrange, 
        brange = brange)
    spiral_arm2 = survey.get_spiral_slice(track = "CrF", 
        lrange = lrange*u.deg, 
        brange = brange*u.deg)

    assert np.allclose(spiral_arm["INTEN"], spiral_arm2["INTEN"], equal_nan = True)

def test_track_shape_error():
    """
    ensure TypeError raised if bad track format provided
    """
    track = np.random.randn(50,1)
    try:
        bad_arm = survey.get_spiral_slice(track = track)
    except TypeError:
        assert True
    else:
        assert False

def test_vel_width():
    """
    ensure different velocity widths work
    """
    import astropy.units as u
    spiral_arm = survey.get_spiral_slice(track = "Perseus", 
        vel_width = 20.)
    spiral_arm2 = survey.get_spiral_slice(track = "Perseus", 
        vel_width = 20.*u.km/u.s)

    assert np.allclose(spiral_arm["INTEN"], spiral_arm2["INTEN"], equal_nan = True)

def test_wrap_at_180():
    """
    ensure longitude angle wrapping works
    """
    spiral_arm = survey.get_spiral_slice(track = "CrF", 
        wrap_at_180 = True)
    spiral_arm2 = survey.get_spiral_slice(track = "Carina_far", 
        wrap_at_180 = True)

    assert np.allclose(spiral_arm["INTEN"], spiral_arm2["INTEN"], equal_nan = True)

def test_interpolation():
    """
    ensure interpolation of central velocities work
    current a "fake" placeholder test - not yet implemented
    """
    spiral_arm = survey.get_spiral_slice(track = "perseus", 
        interpolate = True)
    spiral_arm2 = survey.get_spiral_slice(track = "Per", 
        interpolate = False)

    assert np.allclose(spiral_arm["INTEN"], spiral_arm2["INTEN"], equal_nan = True)



