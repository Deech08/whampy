import pytest
import numpy as np 
import astropy.units as u
from ..skySurvey import SkySurvey 

# Set up the random number generator.
np.random.seed(1234)

# When running tests locally: filename = "/Users/dk/Data/WHAM/wham-ss-DR1-v161116-170912.fits"

def test_remote_load():
    from ..skySurvey import SkySurvey 
    """
    Ensure survey loads from default remote link
    """

    survey = SkySurvey(mode = "remote")
    assert survey["VELOCITY"].unit == u.km/u.s

survey = SkySurvey()

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

def test_section_circle_coord():
  from astropy.coordinates import Angle
  from astropy.coordinates import SkyCoord
  """
  Ensure survey section extraction works
  """

  l = Angle(np.random.random(1) * 360.*u.deg).wrap_at("180d")
  b = Angle((np.random.random(1) * 180.*u.deg) - 90*u.deg)

  center = SkyCoord(l = l, b = b, frame = 'galactic')
  radius = np.random.random()*30. * u.deg
  if radius < 5 * u.deg:
      radius = 5 * u.deg

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

def test_not_quantity_radius_coord():
    try:
        survey.sky_section(SkyCoord(l = np.random.random(1) * u.deg, b = np.random.random(1) * u.deg, 
                                    frame = 'galactic'), 
                            radius = 5.)
    except TypeError:
        assert True
    else:
        assert False



