import logging

import numpy as np 
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle

from .whampyTableMixin import SkySurveyMixin
from scipy.io import readsav

import os.path 

directory = os.path.dirname(__file__)



class SkySurvey(SkySurveyMixin, Table):
    """
    Core WHAM SkySurvey CLass

    Load, view, manipulate, and plot basic results from WHAM Sky Survey

    Parameters
    ----------
    filename: 'str', optional, must be keyword
        filename of WHAM survey fits file
        Defaults to URL
        can also be idl sav file
    mode: 'str', optional, must be keyword
        can be "local" or "remote" to signify using local file or remote access
    idl_var: 'str', optional, must be keyword
        if provided, points to name of IDL structure in sav file containing relevant data
        by default, will take the first variable that is a `numpy.recarray`


    """


    def __init__(self, filename = None, mode = 'local', idl_var = None,
                 **kwargs):
        if filename is None:
            if mode == 'local':
                filename = os.path.join(directory, "data/wham-ss-DR1-v161116-170912.fits")
            elif mode == 'remote':
                filename = "http://www.astro.wisc.edu/wham/ss/wham-ss-DR1-v161116-170912.fits"

        if filename[-4:] == ".sav":
            # IDL Save File
            idl_data = readsav(filename)

            # Find right data entry
            if idl_var is None:
                for key in idl_data.keys():
                    if idl_data[key].__class__ is np.recarray:
                        idl_var = key
                        break
            if idl_var not in idl_data.keys():
                raise TypeError("Could not find WHAM data structure in IDL Save File")

            survey_data = idl_data[idl_var]

            # Covert some columns to float arrays and add to new dictionary
            data_dict = {}
            # Standard WHAM IDL Save File Format
            data_dict["GAL-LON"] = survey_data["GLON"] * u.deg
            data_dict["GAL-LAT"] = survey_data["GLAT"] * u.deg
            data_dict["VELOCITY"] = np.vstack(survey_data["VEL"][:][:]) * u.km/u.s
            data_dict["DATA"] = np.vstack(survey_data["DATA"][:][:]) * u.R * u.s / u.km
            data_dict["VARIANCE"] = np.vstack(survey_data["VAR"][:][:]) * (u.R * u.s / u.km)**2

            # Extra info from IDL Save Files Dependent on type of File / origin
            if "INTEN" in survey_data.dtype.names:
                data_dict["INTEN"] = survey_data["INTEN"] * u.R
            if "OINTEN" in survey_data.dtype.names:
                data_dict["OINTEN"] = survey_data["OINTEN"] * u.R
            if "ERROR" in survey_data.dtype.names:
                data_dict["ERROR"] = survey_data["ERROR"] * u.R

            for name in survey_data.dtype.names:
                if name not in ("GLON", "GLAT", "VEL", "DATA", "VAR", "INTEN", "OINTEN", "ERROR"):
                    data_dict[name] = survey_data[name]

            super().__init__(data = data_dict, **kwargs)

            if "INTEN" not in data_dict:
                self["INTEN"] = self.moment()
            if "ERROR" not in data_dict:
                _, self["ERROR"] = self.moment(return_sigma = True)

            del data_dict
            del idl_data



        else:
            with fits.open(filename) as hdulist:
                self.header = hdulist[0].header
                self.table_header = hdulist[1].header

                self.date = hdulist[0].header["DATE"]
                self.tag = hdulist[0].header["TAG"]
                self.version = hdulist[0].header["VERSION"]

            t = Table.read(filename)

            # Set / Fix Units to comply with astropy.units
            for column in t.columns:
                if t[column].unit == "DEG":
                    t[column].unit = u.deg
                elif t[column].unit == "KM/S":
                    t[column].unit = u.km/u.s 
                elif t[column].unit == "RAYLEIGH/(KM/S)":
                    t[column].unit = u.R / u.km * u.s
                elif t[column].unit == "(RAYLEIGH/(KM/S))^2":
                    t[column].unit = (u.R / u.km * u.s)**2
                elif t[column].unit == "RAYLEIGH":
                    t[column].unit = u.R

            super().__init__(data = t.columns, meta = t.meta, **kwargs)

            del t



    def sky_section(self, bounds, radius = None, wrap_at_180 = True):
        """
        Extract a sub section of the survey from the sky

        Parameters
        ----------

        bounds: `list` or `Quantity` or `SkyCoord`
            if `list` or `Quantity` must be formatted as:
                [min Galactic Longitude, max Galactic Longitude, min Galactic Latitude, max Galactic Latitude]
                or 
                [center Galactic Longitude, center Galactic Latitude] and requires radius keyword to be set
                default units of u.deg are assumed
            if `SkyCoord', must be length 4 or length 1 or length 2
                length 4 specifies 4 corners of rectangular shape
                length 1 specifies center of circular region and requires radius keyword to be set
                length 2 specifies two corners of rectangular region
        radius: 'number' or 'Quantity', optional, must be keyword
            sets radius of circular region
        wrap_at_180: `bool`, optional, must be keyword
            if True, wraps longitude angles at 180d
            use if mapping accross Galactic Center
        """
        if wrap_at_180:
            wrap_at = "180d"
        else:
            wrap_at = "360d"

        if not isinstance(bounds, u.Quantity) | isinstance(bounds, SkyCoord):
            bounds *= u.deg
            logging.warning("No units provided for bounds, assuming u.deg")

        wham_coords = self.get_SkyCoord()

        if isinstance(bounds, SkyCoord):
            if len(bounds) == 1:
                if radius is None:
                    raise TypeError("Radius must be provided if only a single coordinate is given")
                elif not isinstance(radius, u.Quantity):
                    radius *= u.deg
                    logging.warning("No units provided for radius, assuming u.deg")
                center = bounds
            elif len(bounds) >= 2:
                min_lon, max_lon = bounds.l.wrap_at(wrap_at).min(), bounds.l.wrap_at(wrap_at).max()
                min_lat, max_lat = bounds.b.min(), bounds.l.max()
        elif len(bounds) == 2:
            if radius is None:
                raise TypeError("Radius must be provided if only a single coordinate is given")
            elif not isinstance(radius, u.Quantity):
                radius *= u.deg
                logging.warning("No units provided for radius, assuming u.deg")
            center = SkyCoord(l = bounds[0], b = bounds[1], frame = 'galactic')
        elif len(bounds) == 4:
            min_lon, max_lon, min_lat, max_lat = Angle(bounds)
            min_lon = min_lon.wrap_at(wrap_at)
            max_lon = max_lon.wrap_at(wrap_at)
        else:
            raise TypeError("Input bounds and/or radius are not understood")

        # rectangular extraction
        if radius is None:
            # Mask of points inside rectangular region
            inside_mask = wham_coords.l.wrap_at(wrap_at) <= max_lon
            inside_mask &= wham_coords.l.wrap_at(wrap_at) >= min_lon
            inside_mask &= wham_coords.b <= max_lat
            inside_mask &= wham_coords.b >= min_lat

        else: # Circle extraction
            # Compute Separation
            # Warning to self: This is VERY slow
            sep = wham_coords.separation(center)

            # Mask of points inside circular region
            inside_mask = sep <= radius

        return self[inside_mask]
