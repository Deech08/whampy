import pytest
import astropy.units as u
import numpy as np
from ..skySurvey import SkySurvey 

# Set up the random number generator.
np.random.seed(1234)

# Load global survey
survey = SkySurvey()

def test_get_spectrum():
	from astropy.coordinates import SkyCoord
	"""
	basic extraction test
	"""
	l = np.random.random() * 360.
	b = np.random.random() * 180. - 90
	coordinate = SkyCoord(l = l*u.deg, b = b*u.deg, frame = 'galactic')
	spec = survey.get_spectrum(coordinate)
	index = survey.get_spectrum(coordinate, index = True)

	assert np.allclose(spec["DATA"],survey[index]["DATA"], equal_nan = True)

def test_get_spectrum_quantity():
	"""
	ensure not using SkyCoord works
	"""
	l = np.random.random() * 360.
	b = np.random.random() * 180. - 90
	coordinate = [l,b] * u.deg
	coordinate2 = coordinate.value
	spec = survey.get_spectrum(coordinate)
	index = survey.get_spectrum(coordinate2, index = True)

	assert np.allclose(spec["DATA"],survey[index]["DATA"], equal_nan = True)