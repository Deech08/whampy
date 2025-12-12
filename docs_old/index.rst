Whampy Documentation
========================

The whampy package provides an easy way to load, view, and do science with the 
Wisconsin H-Alpha Mapper (WHAM) Sky Survey. 
It provides the following main features:

-The ability to load the Sky Survey Data from the FITS table
-The ability to quickly plot sections of the sky with beam maps
-The ability to calculate moment maps and arithmetic using the data

Quick Start
-----------

Here is a simple script demonstrating the modspectra package:

    >>> from whampy.skySurvey import SkySurvey
    >>> # Load the Survey
    >>> survey = SkySurvey()

    >>> # Compute Moments
    >>> moment_0, err_0 = survey.moment(order = 0, return_sigma = True)
    >>> moment_1, err_1 = survey.moment(order = 1, return_sigma = True)
    >>> moment_2, err_2 = survey.moment(order = 2, return_sigma = True)


Using whampy
----------------

This package is built using the `astropy` package and matplotlib. 


Getting started
^^^^^^^^^^^^^^^

.. toctree::
  :maxdepth: 1

  installing.rst
  loading_data.rst
  intensity_maps.rst
  moment_maps.rst
  click_maps.rst
  spiral_arms.rst
  stacking.rst