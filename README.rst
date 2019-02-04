Python Package to Interact with, Visualize, and Analyze the Wisconsin H-Alpha Mapper - SKy Survey
-------------------------------------------------------------------------------------------------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

.. image:: https://travis-ci.com/Deech08/whampy.svg?branch=master
    :target: https://travis-ci.com/Deech08/whampy

.. image:: https://coveralls.io/repos/github/Deech08/whampy/badge.svg?branch=master&service=github
	:target: https://coveralls.io/github/Deech08/whampy?branch=master&service=github

.. image:: https://readthedocs.org/projects/whampy/badge/?version=latest
	:target: https://whampy.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status

`WHAM <www.astro.wisc.edu/wham>`_


The `whampy` package provides an easy way to load, view, and do science with the 
Wisconsin H-Alpha Mapper (WHAM) Sky Survey. 
It provides the following main features:

-The ability to load the Sky Survey Data from the FITS table
-The ability to quickly plot sections of the sky with beam maps
-The ability to calculate moment maps and arithmetic using the data

Installation
------------

To install the latest developer version of whampy you can type::

    git clone https://github.com/Deech08/whampy.git
    cd whampy
    python setup.py install

You may need to add the ``--user`` option to the last line `if you do not
have root access <https://docs.python.org/2/install/#alternate-installation-the-user-scheme>`_.
You can also install the latest developer version in a single line with pip::

    pip install git+https://github.com/Deech08/whampy.git


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

License
-------

This project is Copyright (c) DK (Dhanesh Krishnarao) and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.



