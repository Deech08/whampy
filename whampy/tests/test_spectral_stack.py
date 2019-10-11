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
	stack_2 = survey[0:5].stack_spectra_bootstrap(set_name = "test")
	assert np.allclose(stack_1["DATA"], stack_2["DATA"], equal_nan = True, atol = 1e-4)

def test_columns():
	"""
	column test of spectral stacking
	"""
	stack_1 = survey[0:5].stack_spectra_bootstrap(data_column = "DATA")
	stack_2 = survey[0:5].stack_spectra_bootstrap(variance_column = "VARIANCE")
	assert np.allclose(stack_1["DATA"], stack_2["DATA"], equal_nan = True, atol = 1e-4)


def test_velocity_name():
	"""
	velocity and name input test of spectral stacking
	"""
	stack_1 = survey[0:5].stack_spectra_bootstrap(velocity = np.round(survey[0]["VELOCITY"]))
	stack_2 = survey[0:5].stack_spectra_bootstrap(set_name = "Test")
	assert np.allclose(stack_1["DATA"], stack_2["DATA"], equal_nan = True, atol = 1e-4)

def test_boot():
	"""
	bootstrap parameter input test of spectral stacking
	"""
	stack_1 = survey[0:5].stack_spectra_bootstrap(n_boot = 10000)
	stack_2 = survey[0:5].stack_spectra_bootstrap(ci = 95)
	assert np.allclose(stack_1["DATA"], stack_2["DATA"], equal_nan = True, atol = 1e-4)	

def test_estimator():
	"""
	estimator input test of spectral stacking
	"""
	stack_1 = survey[0:5].stack_spectra_bootstrap(estimator = np.nanmean)
	stack_2 = survey[0:5].stack_spectra_bootstrap()
	assert np.allclose(stack_1["DATA"], stack_2["DATA"], equal_nan = True, atol = 1e-4)	

def test_window_combine_basic():
	"""
	test combining windows
	"""	

	from ..spectralStack import combine_velocity_windows_bootstrap as cvwb
	window_1 = survey[10:20]
	window_2 = survey[30:40]
	window_1["VELOCITY"] += 50
	window_2["VELOCITY"] -= 50
	window_single = survey[50]

	stack_1 = cvwb([window_1, window_2], survey_spectra = window_single)
	stack_2 = cvwb([window_2, window_1], survey_spectra = window_single, set_name = "test")
	works = np.allclose(stack_1["DATA"], stack_2["DATA"], equal_nan = True, atol = 1e-2)
	print(works)
	assert works

