import pytest
import numpy as np 
import astropy.units as u
from ..skySurvey import SkySurvey 

from ..skySurvey import directory

# Set up the random number generator.
np.random.seed(1234)

# When running tests locally: filename = "/Users/dk/Data/WHAM/wham-ss-DR1-v161116-170912.fits"

def test_remote_load():
    """
    Ensure survey loads from default remote link
    """

    survey = SkySurvey(mode = "remote")
    assert survey["VELOCITY"].unit == u.km/u.s

survey = SkySurvey()

def test_idlsav_load():
    import os.path
    """
    Ensure IDL Save File loading also works
    """
    filename = os.path.join(directory, "data/wham-ss-DR1-v161116-170912.sav")
    survey_idl = SkySurvey(filename = filename)
    assert survey["DATA"].unit == u.R * u.s / u.km

def test_idlsav_load_varerror():
    import os.path
    """
    Ensure IDL Save File loading fails if WHAM data not there
    """
    filename = os.path.join(directory, "data/wham-ss-DR1-v161116-170912.sav")
    try:
        surveey_idl = SkySurvey(filename = filename, idl_var = "test_fail")
    except TypeError:
        assert True
    else:
        assert False

# def test_idlsav_load_nointen():
#         import os.path
#     """
#     Ensure IDL Save File loading also works for developer versions that
#     don't have a "INTEN" field

#     Note: This test is not yet implemented - need a test IDL Save File
#     """
#     cur_directory = os.path.dirname(__file__)
#     filename = os.path.join(cur_directory, "test_data/test_no_inten.sav")
#     survey_idl = SkySurvey(filename = filename)
#     assert survey["INTEN"].unit == u.R
    

def test_section_circle():
    from astropy.coordinates import Angle
    """
    Ensure survey section extraction works
    """

    l = Angle(np.random.random(1) * 360.*u.deg).wrap_at("180d")
    b = Angle((np.random.random(1) * 180.*u.deg) - 90*u.deg)

    center = [l.value,b.value]
    radius = np.random.random()*30.

    circle = survey.sky_section(center, radius)

    assert circle["DATA"].unit == u.R / u.km * u.s

def test_section_circle_coord_radius_number():
    from astropy.coordinates import Angle
    from astropy.coordinates import SkyCoord
    """
    Ensure survey section extraction works
    """

    l = Angle(np.random.random(1) * 360.*u.deg).wrap_at("180d")
    b = Angle((np.random.random(1) * 180.*u.deg) - 90*u.deg)

    center = SkyCoord(l = l, b = b, frame = 'galactic')
    radius = np.random.random()*30.
    if radius < 5:
      radius = 5

    circle = survey.sky_section(center, radius)

    assert circle["DATA"].unit == u.R / u.km * u.s

def test_section_rect_coord():
    from astropy.coordinates import Angle
    from astropy.coordinates import SkyCoord
    """
    Ensure survey section extraction works
    """

    l = Angle(np.random.random(2) * 360.*u.deg).wrap_at("180d")
    b = Angle((np.random.random(2) * 180.*u.deg) - 90*u.deg)

    bounds = SkyCoord(l = l, b = b, frame = 'galactic')

    rect = survey.sky_section(bounds)

    assert rect["DATA"].unit == u.R / u.km * u.s

def test_section_rect():
    from astropy.coordinates import Angle
    from astropy.coordinates import SkyCoord
    """
    Ensure survey section extraction works
    """

    l = Angle(np.random.random(2) * 360.*u.deg).wrap_at("180d")
    b = Angle((np.random.random(2) * 180.*u.deg) - 90*u.deg)

    bounds = [l.min().value, l.max().value, b.min().value, b.max().value]

    rect = survey.sky_section(bounds)

    assert rect["DATA"].unit == u.R / u.km * u.s

def test_section_no_wrap():
    from astropy.coordinates import Angle
    from astropy.coordinates import SkyCoord
    """
    Ensure survey section extraction works
    """

    l = Angle(np.random.random(2) * 360.*u.deg).wrap_at("360d")
    b = Angle((np.random.random(2) * 180.*u.deg) - 90*u.deg)

    bounds = [l.min().value, l.max().value, b.min().value, b.max().value]

    rect = survey.sky_section(bounds, wrap_at_180 = False)

    assert rect["DATA"].unit == u.R / u.km * u.s

def test_bounds_error():
    try:
        survey.sky_section(np.random.random(5))
    except TypeError:
        assert True
    else:
        assert False

def test_no_radius():
    try:
        survey.sky_section(np.random.random(1), radius = None)
    except TypeError:
        assert True
    else:
        assert False

def test_no_radius_coord():
    from astropy.coordinates import SkyCoord
    try:
        survey.sky_section(SkyCoord(l = np.random.random(1) * u.deg, b = np.random.random(1) * u.deg, 
                                    frame = 'galactic'), 
                            radius = None)
    except TypeError:
        assert True
    else:
        assert False

def test_no_radius_len2():
    try:
        survey.sky_section(np.random.random(2), 
                            radius = None)
    except TypeError:
        assert True
    else:
        assert False



