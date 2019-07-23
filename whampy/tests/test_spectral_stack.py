import pytest
import numpy as np
from ..skySurvey import SkySurvey

# Set up the random number generator
np.random.seed(1234)

# Load global survey
survey = SkySurvey()



def test_basic():
	"""
	basic test of spectral stacking
	"""
	stack_1 = survey[0:5].stack_spectra_bootstrap()
	stack_2 = survey[0:5].stack_spectra_bootstrap()
	assert np.allclose(stack_1["DATA"], stack_2["DATA"], equal_nan = True)

def test_columns():
	"""
	column test of spectral stacking
	"""
	stack_1 = survey[0:5].stack_spectra_bootstrap(data_column = "DATA")
	stack_2 = survey[0:5].stack_spectra_bootstrap(variance_column = "VARIANCE")
	assert np.allclose(stack_1["DATA"], stack_2["DATA"], equal_nan = True)


def test_velocity_name():
	"""
	velocity and name input test of spectral stacking
	"""
	stack_1 = survey[0:5].stack_spectra_bootstrap(velocity = np.round(survey[0]["VELOCITY"]))
	stack_2 = survey[0:5].stack_spectra_bootstrap(set_name = "Test")
	assert np.allclose(stack_1["DATA"], stack_2["DATA"], equal_nan = True)

def test_boot():
	"""
	bootstrap parameter input test of spectral stacking
	"""
	stack_1 = survey[0:5].stack_spectra_bootstrap(n_boot = 1000)
	stack_2 = survey[0:5].stack_spectra_bootstrap(ci = 95)
	assert np.allclose(stack_1["DATA"], stack_2["DATA"], equal_nan = True)	

def test_estimator():
	"""
	estimator input test of spectral stacking
	"""
	stack_1 = survey[0:5].stack_spectra_bootstrap(estimator = np.mean)
	stack_2 = survey[0:5].stack_spectra_bootstrap()
	assert np.allclose(stack_1["DATA"], stack_2["DATA"], equal_nan = True)		