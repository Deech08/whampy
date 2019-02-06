Making Interactive Maps with `whampy`
====================================

You can make interactive maps of the WHAM data where you can click on the map 
and have a spectrum plotted from the nearest WHAM pointing using the `~whampyTableMixin.click_map` method::

	>>> from whampy.skySurvey import SkySurvey
	>>> survey = SkySurvey()

	>>> click_map = survey.click_map()

.. image:: images/quick_click_map.png
   :width: 600


`~whampyTableMixin.click_map` will accept and pass keywords to `~whampyTableMixin.intensity_map`. You can 
also pass in your own set of figure and axes instances to customize the orientation, shape, and size of axes::

	>>> import cartopy.crs as ccrs
	>>> fig = plt.figure()
	>>> image_ax = fig.add_subplot(111, projection = ccrs.Mollweide())

	>>> click_map = survey.click_map(fig = fig, image_ax = image_ax, 
									spectra_kwargs = {"c":'b', "ls": ":"})

.. image:: images/custom_click_map.png
   :width: 600

