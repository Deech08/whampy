import logging

import numpy as np 
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle

from .whampyTableMixin import SkySurveyMixin

import os.path 

directory = os.path.dirname(whampy.__file__)



class SkySurvey(SkySurveyMixin, Table):
    """
    Core WHAM SkySurvey CLass

    Load, view, manipulate, and plot basic results from WHAM Sky Survey

    Parameters
    ----------
    filename: 'str', optional, must be keyword
        filename of WHAM survey fits file
        Defaults to URL
    mode: 'str', optional, must be keyword
        can be "local" or "remote" to signify using local file or remote access


    """


    def __init__(self, filename = None, mode = 'local',
                 **kwargs):
        if filename == None:
            if mode = 'local':
                filename = os.path.dirname(directory, "data/wham-ss-DR1-v161116-170912.fits")
            elif mode = 'remote':
            filename = "http://www.astro.wisc.edu/wham/ss/wham-ss-DR1-v161116-170912.fits"

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



    def sky_section(self, bounds, radius = None):
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
        """

        if not isinstance(bounds, u.Quantity) | isinstance(bounds, SkyCoord):
            bounds *= u.deg
            logging.warning("No units provided for bounds, assuming u.deg")

        wham_coords = self.get_SkyCoord()

        if isinstance(bounds, SkyCoord):
            if len(bounds) == 1:
                if radius == None:
                    raise TypeError
                    print("Radius must be provided if only a single coordinate is given")
                elif not isinstance(radius, u.Quantity):
                    radius *= u.deg
                    logging.warning("No units provided for radius, assuming u.deg")
                center = bounds
            elif len(bounds) >= 2:
                min_lon, max_lon = bounds.l.wrap_at("180d").min(), bounds.l.wrap_at("180d").max()
                min_lat, max_lat = bounds.b.min(), bounds.l.max()
        elif len(bounds) == 2:
            if radius == None:
                raise TypeError
                print("Radius must be provided if only a single coordinate is given")
            elif not isinstance(radius, u.Quantity):
                radius *= u.deg
                logging.warning("No units provided for radius, assuming u.deg")
            center = SkyCoord(l = bounds[0], b = bounds[1], frame = 'galactic')
        elif len(bounds) == 4:
            min_lon, max_lon, min_lat, max_lat = Angle(bounds).wrap_at("180d")
        else:
            raise TypeError
            print("Input bounds and/or radius are not understood")

        # rectangular extraction
        if radius == None:
            # Mask of points inside rectangular region
            inside_mask = wham_coords.l.wrap_at("180d") <= max_lon
            inside_mask &= wham_coords.l.wrap_at("180d") >= min_lon
            inside_mask &= wham_coords.b <= max_lat
            inside_mask &= wham_coords.b >= min_lat

        else: # Circle extraction
            # Compute Separation
            # Warning to self: This is VERY slow
            sep = wham_coords.separation(center)

            # Mask of points inside circular region
            inside_mask = sep <= radius

        return self[inside_mask]
