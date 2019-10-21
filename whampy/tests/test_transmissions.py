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

def test_read_file_list():
	"""
	Check file_list format
	"""
	from ..skySurvey import SkySurvey
	file_list = os.path.join(cal_directory, "ha/combo/g300_0-000814.fts")

	data = SkySurvey(file_list = file_list)
	data2 = SkySurvey(file_list = [file_list])
	assert np.allclose(data["INTEN"], data2["INTEN"])

def test_read_file_invalid():
	"""
	Rejects invalid file_list format
	"""
	from ..skySurvey import SkySurvey
	file_list = 0
	try:
		SkySurvey(file_list = file_list)
	except TypeError:
		assert True
	else:
		assert False

def test_no_extension():
	"""
	Rejects when extension not found
	"""
	from ..skySurvey import SkySurvey
	file_list = os.path.join(cal_directory, "ha/combo/g300_0-000814.fts")
	try:
		SkySurvey(file_list = file_list, extension = "FAKE")
	except ValueError:
		assert True
	else:
		assert False


