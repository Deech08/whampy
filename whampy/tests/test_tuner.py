import pytest
import numpy as np

from ..dataReduction import compute_transmissions
from ..skySurvey import directory


# Set up the random number generator
np.random.seed(1234)

import os.path
cal_directory = os.path.join(directory, "tests/test_data/test_dataReduction/ha")

def test_basic():
	"""
	Basic tuner spectrum load
	"""
	from ..skySurvey import SkySurvey
	file_list = os.path.join(cal_directory, "lamp_H-Alpha_45.0_65.2_203312_spec.fts")
	spec = SkySurvey(tuner_list = file_list)
	spec2 = SkySurvey(tuner_list = [file_list])

	assert np.allclose(spec["DATA"], spec2["DATA"], equal_nan = True)

def test_iter_error():
	"""
	Ensure tuner_list is iterable
	"""
	from ..skySurvey import SkySurvey
	file_list = 0
	try:
		spec = SkySurvey(tuner_list = file_list)
	except TypeError:
		assert True
	else:
		assert False




