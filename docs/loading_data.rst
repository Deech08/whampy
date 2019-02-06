Loading Maps with `whampy`
==========================

By default, `whampy` can load the DR1 Full sky release of WHAM data from the fits file release.
This can be done with a remote load of the data or with a provided local copy of the file::

	>>> from whampy.skySurvey import SkySurvey
	>>> # Load from remote
	>>> survey_remote = SkySurvey(mode = 'remote')

	>>> # Load from provided default file
	>>> survey = SkySurvey()

WHAM natively works in IDL, so data files are initially stored as IDL Save files. These files can 
also be loaded in as a :class:`~whampy.skySurvey.SkySurvey` and use the fiunctionality built in. 
This is especially useful for members of the WHAM team to work with preliminary data pre-release::

	>>> filename = "WHAM_IDL_SAVE_FILE.sav"
	>>> # Load WHAM data from save file
	>>> survey_idl = SkySurvey(filename = filename)




