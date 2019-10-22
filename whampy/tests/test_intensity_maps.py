import pytest
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from ..skySurvey import SkySurvey 

# Set up the random number generator.
np.random.seed(1234)

# Local directory:
# "/Users/dk/Data/WHAM/wham-ss-DR1-v161116-170912.fits"

survey = SkySurvey()

BASELINE_DIR = 'baseline'

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_basic():
    """
    Basic intensity plot
    """
    fig = survey.intensity_map()
    return fig

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_fig_input():
    """
    fig kwarg
    """
    fig = plt.figure()
    return survey.intensity_map(fig = fig)

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_ax_input():
    """
    ax kwarrg
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    return survey.intensity_map(ax = ax)

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_ranges():
    """
    lrange and brange
    """
    lrange = [30,-30]
    brange = [-30,30]
    return survey.intensity_map(lrange = lrange * u.deg, brange = brange)

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_ranges_unit():
    """
    lrange and brange
    """
    lrange = [30,-30]
    brange = [-30,30]
    return survey.intensity_map(lrange = lrange, brange = brange*u.deg)

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_colorbar():
    """
    colorbar
    """
    return survey.intensity_map(colorbar = True, 
        cbar_kwargs = {"orientation":"horizontal"})

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_sc():
    """
    return_sc test
    """
    sc, fig = survey.intensity_map(return_sc = True)
    cb = fig.colorbar(sc, orientation = "horizontal")
    return fig

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_cartopy_stereo():
    import cartopy.crs as ccrs
    """
    test stereo projection with cartopy
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = ccrs.Stereographic())
    fig = survey.intensity_map(ax = ax, fig = fig, s_factor=2, 
        lrange = [40,-40], brange = [-20,20])
    return fig

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_vel_range():
    """
    test velocity range input
    """
    vel_range = [-20,20]
    return survey.intensity_map(vel_range = vel_range)

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_vel_range_units():
    """
    test velocity range input
    """
    vel_range = [-20,20]*u.km/u.s
    return survey.intensity_map(vel_range = vel_range)

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_basic_smooth():
    """
    Basic intensity plot smooth
    """
    fig = survey.intensity_map(smooth = True)
    return fig
    
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance = 20)
def test_basic_smooth_lrange():
    """
    Basic intensity plot smooth
    """
    fig = survey.intensity_map(smooth = True, lrange = [50,-50])
    return fig



