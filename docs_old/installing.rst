Installing ``whampy``
=====================

Requirements
------------

This package has the following dependencies:

* `Python <http://www.python.org>`_ 3.6 or later
* `Numpy <http://www.numpy.org>`_ 1.8 or later
* `Astropy <http://www.astropy.org>`_ 1.0 or later
* `matplotlib <https://matplotlib.org/>`_ 3.0 or later
* `seaborn <https://seaborn.pydata.org/index.html>`_ 0.9 or later

Optional to overplot FITS Data cubes in click_map and use different map projections
* `cartopy <https://scitools.org.uk/cartopy/docs/latest/>`_0.17.0 or later
* `spectral cube <https://spectral-cube.readthedocs.io/en/latest/#>`_ >=0.4.4
* `pyregion <https://pyregion.readthedocs.io/en/latest/>`_>=2.0
* `Regions <https://astropy-regions.readthedocs.io/en/latest>`_ >=0.3dev, optional
  (Serialises/Deserialises DS9/CRTF region files and handles them. Used when
  extracting a subcube from region)

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


