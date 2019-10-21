import pytest
import numpy as np
from ..skySurvey import SkySurvey

# Set up the random number generator
np.random.seed(1234)

survey = SkySurvey()



def test_basic():
    """
    basic test of equivalent width
    """
    import astropy.units as u
    survey["INTEN"] = survey["INTEN"].data * u.R

    ha_wave = 656.3 * u.nm
    optical_ha_equiv = u.doppler_optical(ha_wave)
    dv = (survey["VELOCITY"][0][1] - survey["VELOCITY"][0][0]) * u.km/u.s
    dlambda = (dv).to(u.AA, equivalencies=optical_ha_equiv) - \
        (0 * u.km/u.s).to(u.AA, equivalencies=optical_ha_equiv)

    survey["BKG"] = np.abs(np.random.randn(len(survey["INTEN"]))) * 100.
    survey["BKGSD"] = survey["BKG"] * .1


    ew, ew_error = survey.get_equivalent_width(continuum = survey["BKG"] / 22.8 * u.R / dlambda, 
        continuum_error = survey["BKGSD"] / 22.8 * u.R / dlambda, return_sigma = True)

    ew2, ew2_error = survey.get_equivalent_width(intensity = survey["INTEN"].data * u.R, 
        intensity_error = survey["ERROR"].data * u.R)

    assert np.allclose(ew.value, ew2.value)

def test_continuum_unit_error():
    """
    continuum must have units
    """
    try:
        survey.get_equivalent_width(continuum = 5)
    except TypeError:
        assert True
    else:
        assert False

def test_intensity_error():
    """
    test intensity error providing
    """
    import astropy.units as u
    error = survey["ERROR"].data * u.R

    ew, ew_error = survey.get_equivalent_width(intensity_error = error)
    ew2, ew_error2 = survey.get_equivalent_width()

    assert np.allclose(ew.value, ew2.value)

def test_intensity_error_with_intensity():
    """
    test intensity error not provided but intensity provided
    """
    try:
        survey.get_equivalent_width(intensity = survey["INTEN"])
    except TypeError:
        assert True
    else:
        assert False

def test_no_sigma():
    """
    test return sigma
    """
    ew = survey.get_equivalent_width(return_sigma = False)
    ew2, ew_error2 = survey.get_equivalent_width()

    assert np.allclose(ew.value, ew2.value)

