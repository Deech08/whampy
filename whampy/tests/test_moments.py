import pytest
import numpy as np 
import astropy.units as u
from ..skySurvey import SkySurvey 

survey = SkySurvey()

def test_moment_0_err():
	"""
	zeroth order moment error
	"""
	moment, err = survey.moment(return_sigma = True)
	assert moment.unit == err.unit

def test_moment_0():
	"""
	zeroth order moment
	"""
	moment, err = survey.moment(return_sigma = True)
	moment2 = survey.moment()
	assert np.allclose(moment.value, moment2.value)

def test_moment_1_err():
	"""
	first order moment error
	"""
	moment, err = survey.moment(order = 1, return_sigma = True)
	assert moment.unit == err.unit

def test_moment_1():
	"""
	zeroth order moment
	"""
	moment, err = survey.moment(order = 1, return_sigma = True)
	moment2 = survey.moment(order = 1)
	assert np.allclose(moment.value, moment2.value)

def test_moment_2_err():
	"""
	second order moment error
	"""
	moment, err = survey.moment(order = 2, return_sigma = True)
	assert moment.unit == err.unit

def test_moment_2():
	"""
	zeroth order moment
	"""
	moment, err = survey.moment(order = 2, return_sigma = True)
	moment2 = survey.moment(order = 2)
	assert np.allclose(moment.value, moment2.value)

def test_moment_2_vel_range():
	"""
	second order moment
	"""
	vmin = np.random.random(1) * -150.
	vmax = np.random.random(1) * 150.
	moment, err = survey.moment(order = 2, return_sigma = True, vmin = vmin, vmax = vmax)
	assert moment.unit == err.unit

def test_moment_2_vel_range_unit():
	"""
	second order moment
	"""
	vmin = np.random.random(1) * -150. * u.km/u.s
	vmax = np.random.random(1) * 150. * u.km/u.s
	moment, err = survey.moment(order = 2, return_sigma = True, vmin = vmin, vmax = vmax)
	assert moment.unit == err.unit