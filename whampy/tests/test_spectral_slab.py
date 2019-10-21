import pytest
import numpy as np
from ..skySurvey import SkySurvey

# Set up the random number generator
np.random.seed(1234)




def test_basic():
	"""
	basic test of spectral stacking
	"""
	import astropy.units as u
	survey = SkySurvey()
	survey2 = SkySurvey()
	survey.get_spectral_slab(-50 * u.km/u.s, 50 * u.km/u.s)
	survey2.get_spectral_slab(-50, 50)
	assert np.allclose(survey["DATA"], survey2["DATA"], equal_nan = True)

def test_no_slab():
	"""
	test no region in velocity range
	"""
	survey = SkySurvey()
	try:
		survey.get_spectral_slab(-300,-250)
	except ValueError:
		assert True
	else:
		assert False