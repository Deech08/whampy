import pytest
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from ..skySurvey import SkySurvey 
from unittest.mock import Mock

# Set up the random number generator.
np.random.seed(1234)

# Load survey
survey = SkySurvey()

BASELINE_DIR = 'baseline'

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_basic_click_event():
	"""
	Test click map with sample click event
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111)
	click_map = survey.click_map(fig = fig, image_ax = ax)
	event = Mock()
	event.button = 1
	event.inaxes = ax
	event.xdata = 30
	event.ydata = 5
	click_map.on_click(event)
	return plt.gcf()

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_basic_click_event_outside():
	"""
	Test click map with sample click event
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111)
	click_map = survey.click_map(fig = fig, image_ax = ax)
	event = Mock()
	event.button = 1
	event.inaxes = None
	event.xdata = 30
	event.ydata = 5
	click_map.on_click(event)
	return plt.gcf()

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_basic_click_no_fig():
	"""
	Test init without fig
	"""
	click_map = survey.click_map()
	return plt.gcf()

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_basic_click_no_imax():
	"""
	Test init without fig
	"""
	fig = plt.figure()
	click_map = survey.click_map(fig = fig)
	return plt.gcf()

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_basic_click_no_specax():
	"""
	Test init without fig
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111)
	click_map = survey.click_map(fig = fig, image_ax = ax)
	return plt.gcf()

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_basic_click_projection():
	import cartopy.crs as ccrs
	"""
	Test init without fig
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = ccrs.Mollweide())
	click_map = survey.click_map(fig = fig, image_ax = ax)
	event = Mock()
	event.button = 1
	event.inaxes = ax
	event.xdata = 30
	event.ydata = 5
	click_map.on_click(event)
	return plt.gcf()

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_basic_click_projection_kwarg():
	import cartopy.crs as ccrs
	"""
	Test init without fig
	"""
	fig = plt.figure()
	click_map = survey.click_map(fig = fig, projection = ccrs.Mollweide())
	return plt.gcf()