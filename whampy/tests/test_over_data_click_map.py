import pytest
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from ..skySurvey import SkySurvey 
from unittest.mock import Mock

from ..skySurvey import directory

# Set up the random number generator.
np.random.seed(1234)

# Load survey
survey = SkySurvey()

BASELINE_DIR = 'baseline'

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_basic_click_event_over():
    """
    Test click map with sample click event on SkySurvey over
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    click_map = survey.click_map(fig = fig, image_ax = ax, over_data = survey, share_yaxis = True)
    event = Mock()
    event.button = 1
    event.inaxes = ax
    event.xdata = 1
    event.ydata = -4
    click_map.on_click(event)
    return plt.gcf()

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_basic_click_event_over_notsharey():
    """
    Test click map with sample click event on SkySurvey over
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    click_map = survey.click_map(fig = fig, image_ax = ax, over_data = survey, share_yaxis = False)
    event = Mock()
    event.button = 1
    event.inaxes = ax
    event.xdata = 1
    event.ydata = -4
    click_map.on_click(event)
    return plt.gcf()

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_str_click_event_over():
    """
    Test click map with sample click event on string load over
    """
    import os.path
    over_data = os.path.join(directory, "tests/test_data/test_cube.fits")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    click_map = survey.click_map(fig = fig, image_ax = ax, over_data = over_data)
    event = Mock()
    event.button = 1
    event.inaxes = ax
    event.xdata = 1
    event.ydata = -4
    click_map.on_click(event)
    return plt.gcf()

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=20)
def test_cube_click_event_over():
    """
    Test click map with sample click event on string load over
    """
    import os.path
    from spectral_cube import SpectralCube
    over_data_path = os.path.join(directory, "tests/test_data/test_cube.fits")
    over_data = SpectralCube.read(over_data_path)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    click_map = survey.click_map(fig = fig, image_ax = ax, over_data = over_data, 
        average_beam = True, radius = 0.5)
    event = Mock()
    event.button = 1
    event.inaxes = ax
    event.xdata = 1
    event.ydata = -4
    click_map.on_click(event)
    return plt.gcf()

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=20)
def test_wcs_click_event_over():
    """
    Ensure click map works with wcs axes
    """
    import os.path
    from spectral_cube import SpectralCube
    over_data_path = os.path.join(directory, "tests/test_data/test_cube.fits")
    over_data = SpectralCube.read(over_data_path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = over_data.wcs, slices = ['x', 'y', 0])
    ax.imshow(over_data[0,:,:].data)

    click_map = survey.click_map(fig = fig, image_ax = ax, over_data = over_data)
    x,y,_  = over_data.wcs.wcs_world2pix(1,-4,0,0)
    event = Mock()
    event.button = 1
    event.inaxes = ax
    event.xdata = x
    event.ydata = y
    click_map.on_click(event)
    return plt.gcf()

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=20)
def test_cube_click_event_over_radius_deg():
    """
    Test click map with sample click event on string load over with radius quantity
    """
    import os.path
    from spectral_cube import SpectralCube
    over_data_path = os.path.join(directory, "tests/test_data/test_cube.fits")
    over_data = SpectralCube.read(over_data_path)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    click_map = survey.click_map(fig = fig, image_ax = ax, over_data = over_data, 
        average_beam = True, radius = 0.5*u.deg)
    event = Mock()
    event.button = 1
    event.inaxes = ax
    event.xdata = 1
    event.ydata = -4
    click_map.on_click(event)
    return plt.gcf()

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=20)
def test_cube_click_event_over_radius_none():
    """
    Test click map with sample click event on string load over with radius not set
    """
    import os.path
    from spectral_cube import SpectralCube
    over_data_path = os.path.join(directory, "tests/test_data/test_cube.fits")
    over_data = SpectralCube.read(over_data_path)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    click_map = survey.click_map(fig = fig, image_ax = ax, over_data = over_data, 
        average_beam = True)
    event = Mock()
    event.button = 1
    event.inaxes = ax
    event.xdata = 1
    event.ydata = -4
    click_map.on_click(event)
    return plt.gcf()

def test_cube_over_readerror():
    """
    Test click map with over data read error
    """
    try:
        click_map = survey.click_map(over_data = "fakefilename.fits")
    except FileNotFoundError:
        assert True
    else:
        assert False


















