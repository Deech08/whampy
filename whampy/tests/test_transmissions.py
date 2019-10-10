import pytest
import numpy as np

from ..dataReduction import compute_transmissions
from ..skySurvey import directory

# Set up the random number generator
np.random.seed(1234)

import os.path
cal_directory = os.path.join(directory, "tests/test_data/test_dataReduction")

def test_basic():
	"""
	Basic transmission Calculation test
	"""

	transmissions_1 = compute_transmissions(cal_directory)
	transmissions_2 = compute_transmissions(cal_directory, 
		lines = ["ha", "nii"], calibrator = ["G300_0"], plot = True)

	assert np.allclose(transmissions_1["ha"]["g300_0"]["SLOPE_THEIL"], 
		transmissions_2["ha"]["G300_0"]["SLOPE_THEIL"])

def test_string_inputs():
	"""
	Basic transmission Calculation test with string input
	"""

	transmissions_1 = compute_transmissions(cal_directory)
	transmissions_2 = compute_transmissions(cal_directory, 
		lines = "ha", calibrator = "G300_0", plot = True)

	assert np.allclose(transmissions_1["ha"]["g300_0"]["SLOPE_THEIL"], 
		transmissions_2["ha"]["G300_0"]["SLOPE_THEIL"])

def test_type_errors():
	"""
	Ensure errors are raised when lines / calibrators not iterable
	"""

	try:
		transmissions = compute_transmissions(cal_directory, lines = 3.0)
	except TypeError:
		try:
			transmissions = compute_transmissions(cal_directory, calibrator = 300.0)
		except TypeError:
			assert True
		else:
			assert False
	else:
		assert False
