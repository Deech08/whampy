Python Package to Interact with, Visualize, and Analyze the Wisconsin H-Alpha Mapper - Sky Survey
-------------------------------------------------------------------------------------------------

.. image:: https://joss.theoj.org/papers/10.21105/joss.01940/status.svg
   :target: https://doi.org/10.21105/joss.01940

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

.. image:: https://github.com/Deech08/whampy/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/Deech08/whampy/actions/workflows/tests.yml

.. image:: https://coveralls.io/repos/github/Deech08/whampy/badge.svg?branch=master&service=github
	:target: https://coveralls.io/github/Deech08/whampy?branch=master&service=github

.. image:: https://readthedocs.org/projects/whampy/badge/?version=latest
	:target: https://whampy.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status


The `whampy` package provides an easy way to load, view, and do science with the 
Wisconsin H-Alpha Mapper (`WHAM <http://www.astro.wisc.edu/wham-site/>`_) Sky Survey. 
It provides the following main features:

* The ability to load the Sky Survey Data from the FITS table
* The ability to quickly plot sections of the sky with beam maps
* The ability to calculate moment maps and arithmetic using the data

Installation
------------

You can install whampy using pip::

	pip install whampy

To install the latest developer version of whampy you can type::

    git clone https://github.com/Deech08/whampy.git
    cd whampy
    python setup.py install

You may need to add the ``--user`` option to the last line `if you do not
have root access <https://docs.python.org/2/install/#alternate-installation-the-user-scheme>`_.
You can also install the latest developer version in a single line with pip::

    pip install git+https://github.com/Deech08/whampy.git


Optional Dependencies
---------------------

`whampy` contains some features that require additonal python packages that are included by default. If you would like to use some of these extra features, you will need to separately install those packages as well. These packages, with links to their documentation and installation instructions, are listed below.

* `cartopy <https://scitools.org.uk/cartopy/docs/latest/>`_
* `spectral cube <https://spectral-cube.readthedocs.io/en/latest/#>`_

The following two packages can be useful if working with the spectral-cube package along with whampy.

* `pyregion <https://pyregion.readthedocs.io/en/latest/>`_
* `Regions <https://astropy-regions.readthedocs.io/en/latest>`_


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

Full Documentation
------------------

Complete documentation of the package and its funcationality can be found at the link below:
`whampy Documentation <https://whampy.readthedocs.io/en/latest/>`_

Contributing
------------

Contributions, feedback, and bug reports are welcome from all users of whampy! If you discover a bug, or have a request for a new feature, please open an issue on this repository using one of the templates. If you would like to make a more direct contribution to the code, please submit a pull request to this repository. This package relies upon the work of the astropy team, and we also ask that all contributions follow the `Astropy Community Code of Conduct <https://www.astropy.org/about.html#codeofconduct>`_. 

License
-------

This project is Copyright (c) DK (Dhanesh Krishnarao) and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.



