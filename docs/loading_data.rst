Loading Maps with `whampy`
==========================

By default, `whampy` can load the DR1 Full sky release of WHAM data from the fits file release.
This can be done with a provided local copy of the file::

	>>> from whampy.skySurvey import SkySurvey
	>>> # Load from provided default file
	>>> survey = SkySurvey()

WHAM natively works in IDL, so data files are initially stored as IDL Save files. These files can 
also be loaded in as a :class:`~whampy.skySurvey.SkySurvey` and use the functionality built in. 
This is especially useful for members of the WHAM team to work with preliminary data pre-release::

	>>> filename = "WHAM_IDL_SAVE_FILE.sav"
	>>> # Load WHAM data from save file
	>>> survey_idl = SkySurvey(filename = filename)


You can also load the H-Alpha surveys of the Magellanic Clouds from 'Smart et al. (2019)'_. 
.. _Smart et al. (2019): https://ui.adsabs.harvard.edu/abs/2019ApJ...887...16S/abstract>`::
and Smart et al. (2023).
This can be done by specifying the survey as a keyword when calling :class:`~whampy.skySurvey.SkySurvey`::

	>>> from whampy.skySurvey import SkySurvey
	>>> # Load SMC Survey
	>>> smc = SkySurvey(survey = "smc_ha")
	>>> # Load LMC Survey
	>>> lmc = SkySurvey(survey = "lmc_ha")


