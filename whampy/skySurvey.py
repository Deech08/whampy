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
    file_list: 'str', 'listlike', optional, must be keyword
        if provided, reads combo fits files directly
    extension: 'str', optional, must be keyword
        extension to load from fits file
        default to "ATMSUB"
    max_raw_vel: 'bool', optional, must be keyword
        if True, returns max RAWSPEC velocity
    from_table: `astropy.table.Table`, `dictionary`, optional, must be keyword
        if provided, initializes directly from a Table
    tuner_list: 'str', 'listlike', optional, must be keyword
        if provided, reads tuner spectra fits files directly
    unreduced_list: `str`, `listtlike`, optional, must be keyword
        if provided, reads unreduced data (RAW/PROC SPEC) 
    input_data_R: `bool`, optional, must be keyword
        if True, assumed input data unit is Rayleighs
    survey: 'str', optional, must be keyword
        specifies which WHAM survey data to load
        options are 
            "mw_ha":Milky Way Sky Survey of H-Alpha
            "mw_sii":Milky Way [SII] Survey (preliminary)
            "lmc_ha":Large Magellanic Cloud H-Alpha Survey (Smart et al. 2023)
            "smc_ha":Small Magellanic Cloud H-Alpha Survey (Smart et al. 2019)


    """


    def __init__(self, filename = None, mode = 'local', idl_var = None, 
                 file_list = None, extension = None, max_raw_vel = False,
                 from_table = None, tuner_list = None, unreduced_list = None, 
                 input_data_R = False, survey = "mw_ha",
                 **kwargs):


        if from_table is not None:
            super().__init__(data = from_table, **kwargs)

        elif tuner_list is not None:
            if isinstance(tuner_list, str):
                tuner_list = [tuner_list]
            elif not hasattr(tuner_list, "__iter__"):
                raise TypeError("Invalid tuner_list Type, tuner_list must be a string, list of strings, or None.")

            if not isinstance(tuner_list, np.ndarray):
                tuner_list = np.array(tuner_list)

            if extension is None:
                extension = "RAWSPEC"

            gaussian_keys = ["MEAN1", "MEAN2", "MEAN3", "MEAN4", "MEAN5", "MEAN6", "MEAN7", 
                                 "WIDTH1", "WIDTH2", "WIDTH3", "WIDTH4", "WIDTH5", "WIDTH6", "WIDTH7", 
                                 "AREA1", "AREA2", "AREA3", "AREA4", "AREA5", "AREA6", "AREA7"]

            file_dict = {}
            for key in gaussian_keys:
                file_dict[key] = np.full_like(tuner_list, np.nan, dtype = float)
            for ell, filename in enumerate(tuner_list):
                with fits.open(filename) as hdulist:
                    primary_header = hdulist["PRIMARY"].header
                    raw_data = hdulist["RAWSPEC"].data
                    extension_header = hdulist[extension].header
                    extension_data = hdulist[extension].data

                # Read in Spectra
                for key in raw_data.dtype.names:
                    if ell > 0:
                        file_dict[key].append(raw_data[key])
                    else:
                        file_dict[key] = [raw_data[key]]

                for key in extension_data.dtype.names:
                    if ell > 0:
                        file_dict["extension_{}".format(key)].append(extension_data[key])
                    else:
                        file_dict["extension_{}".format(key)] = [extension_data[key]]

                # Read in some Primary Header Data
                for key in primary_header.keys():
                    if key not in extension_header.keys():
                        if ell > 0:
                            file_dict[key].append(primary_header[key])
                        else:
                            file_dict[key] = [primary_header[key]]

                

                for key in gaussian_keys:
                    if key in extension_header:
                        file_dict[key][ell] = extension_header[key]


            # Units for Data
            file_dict["VELOCITY"] *= u.km/u.s
            if not input_data_R:
                file_dict["DATA"] *= u.R / 22.8 # ADU to R
                file_dict["VARIANCE"] *= u.R**2 / 22.8**2

            super().__init__(data = file_dict, **kwargs)

            if "INTEN" not in file_dict:
                self["INTEN"] = self.moment()
            if "ERROR" not in file_dict:
                _, self["ERROR"] = self.moment(return_sigma = True)

            del file_dict
                       
        

        # Check calibrators keyword
        elif file_list is not None:
            if isinstance(file_list, str):
                file_list = [file_list]
            elif not hasattr(file_list, "__iter__"):
                raise TypeError("Invalid file_list Type, file_list must be a string, list of strings, or None.")

            if not isinstance(file_list, np.ndarray):
                file_list = np.array(file_list)

            if extension is None:
                extension = "ATMSUB"

            # Check extension exists
            extension_exists_mask = []
            for filename in file_list:
                with fits.open(filename) as hdulist:
                    extension_exists_mask.append(extension in hdulist)

            if np.sum(extension_exists_mask) < len(file_list):
                logging.warning("Not all files in file_list have the extension: {}!".format(extension))

            if np.sum(extension_exists_mask) == 0:
                raise ValueError("No files in file_list have the extension: {}!".format(extension))

            file_dict = {}
            for ell, filename in enumerate(file_list[extension_exists_mask]):

                with fits.open(filename) as hdulist:
                    primary_header = hdulist["PRIMARY"].header
                    atmsub_header = hdulist[extension].header
                    atmsub_data = hdulist[extension].data
                    if max_raw_vel:
                        raw_data = hdulist["RAWSPEC"].data

                        # Read in Max Raw Velocity
                        if ell > 0:
                            file_dict["MAX_RAW_VEL"].append(np.max(raw_data["VELOCITY"]))
                        else:
                            file_dict["MAX_RAW_VEL"] = [np.max(raw_data["VELOCITY"])]

                # Read in ATMSUB Header Data
                for key in atmsub_header.keys():
                    if ell > 0:
                        file_dict[key].append(atmsub_header[key])
                    else:
                        file_dict[key] = [atmsub_header[key]]

                # Read in Spectra
                for key in atmsub_data.dtype.names:
                    if ell > 0:
                        file_dict[key].append(atmsub_data[key])
                    else:
                        file_dict[key] = [atmsub_data[key]]

                # Read in some Primary Header Data
                for key in primary_header.keys():
                    if key not in atmsub_header.keys():
                        if ell > 0:
                            file_dict[key].append(primary_header[key])
                        else:
                            file_dict[key] = [primary_header[key]]



            # Units for Data
            file_dict["VELOCITY"] *= u.km/u.s
            if not input_data_R:
                file_dict["DATA"] *= u.R / 22.8 # ADU to R
                file_dict["VARIANCE"] *= u.R**2 / 22.8**2

            super().__init__(data = file_dict, **kwargs)

            if "INTEN" not in file_dict:
                self["INTEN"] = self.moment()
            if "ERROR" not in file_dict:
                _, self["ERROR"] = self.moment(return_sigma = True)

            del file_dict

        elif unreduced_list is not None:
            file_dict = read_unreduced_data(unreduced_list, extension = extension)

            super().__init__(data = file_dict, **kwargs)

            del file_dict


        else:


            if filename is None:
                if mode == 'local':
                    if survey in ["mw", "ha", "sky_survey", "ss", "mw_ha"]:
                        filename = os.path.join(directory, "data/wham-ss-DR1-v161116-170912.fits")
                    elif survey in ["sii", "mw_sii"]:
                        filename = os.path.join(directory, "data/sii.sav")
                    elif survey in ["lmc", "lmc_ha"]:
                        filename = os.path.join(directory, "data/lmc_combined_map.sav")
                    elif survey in ["smc", "smc_ha"]:
                        filename = os.path.join(directory, "data/smc_ha_corrected.sav")
                elif mode == 'remote':
                    # if survey in ["mw", "ha", "sky_survey", "ss", "mw_ha"]:
                    # Currently only supports remote loading of ha data. 
                    filename = "https://uwmadison.box.com/shared/static/4kccrw2bgad7muss3z2po7rezklenxxz.fits"

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
                if not input_data_R:
                    data_dict["DATA"]/=22.8
                    data_dict["VARIANCE"]/=22.8**2

                # Extra info from IDL Save Files Dependent on type of File / origin
                if "INTEN" in survey_data.dtype.names:
                    data_dict["INTEN"] = survey_data["INTEN"] * u.R
                if "OINTEN" in survey_data.dtype.names:
                    data_dict["OINTEN"] = survey_data["OINTEN"] * u.R
                if "ERROR" in survey_data.dtype.names:
                    data_dict["ERROR"] = survey_data["ERROR"] * u.R

                extra_keys_str = ("NAME", "DATE", "FSHNAME", "IP", "ATEMP")
                extra_keys_vstack = ("BKG", "BKGSD", "MEAN", "MEANSD", "AREA", "AREASD", 
                    "WIDTH", "WIDTHSD")

                for name in survey_data.dtype.names:
                    if name not in ("GLON", "GLAT", "VEL", "DATA", "VAR", "INTEN", "OINTEN", "ERROR"):
                        if name in extra_keys_vstack:
                            data_dict[name] = np.vstack(survey_data[name])
                        elif name in extra_keys_str:
                            data_dict[name] = np.array(survey_data[name]).astype(str)
                        else:
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
                    if "TAG" in hdulist[0].header.keys():
                        self.tag = hdulist[0].header["TAG"]
                    if "VERSION" in hdulist[0].header.keys():
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


def read_unreduced_data(file_list, extension = None):
    """
    Reads unreduced (RAWSPEC/PROCSPEC) data from list of files

    Parameters
    ----------

    file_list: `list-like`
        list of files to read
    extension: `str`
        fits extension to load
    """

    if isinstance(file_list, str):
                file_list = np.array([file_list])

    elif not hasattr(file_list, "__iter__"):
                raise TypeError("Invalid file_list Type, file_list must be a string, list of strings, or None.")

    if not isinstance(file_list, np.ndarray):
        file_list = np.array(file_list)

    if extension is None:
        extension = "PROCSPEC"

    # Check extension exists
    extension_exists_mask = []
    for filename in file_list:
        with fits.open(filename, memmap = False) as hdulist:
            extension_exists_mask.append(extension in hdulist)
            hdulist.close()

    if np.sum(extension_exists_mask) < len(file_list):
        logging.warning("Not all files in file_list have the extension: {}!".format(extension))

    if np.sum(extension_exists_mask) == 0:
        raise ValueError("No files in file_list have the extension: {}!".format(extension))

    if extension == "AVG":
        primary_keys = ['SIMPLE',
                         'BITPIX',
                         'NAXIS',
                         'EXTEND',
                         'DATE',
                         'BLOCK',
                         'FTSEXT',
                         'NUMPTGS',
                         'DGAL-LON',
                         'DGAL-LAT',
                         'VLSR',
                         'ZENITH_D',
                         'PAMON',
                         'PBMON']
    else:
        primary_keys = ["DATE-OBS", "TIME-OBS", "OBJECT", "WAVELEN", 
             "ZENITH_D", "AIRMASS", "VLSR", "GAL-LON", "GAL-LAT", 
             "PACMD", "PAMON", "PBCMD", "PBMON", "PAERR", "PBERR", "PATEMP", "PBTEMP", 
             "FSHNAME", "FLNGNAME", "FCENTER", "CCDTEMP", "HUMID1", "EXPTIME"]

    file_dict = {}

    unit_dict = {
    "VELOCITY":u.km/u.s, "DATA":u.ph/u.s, "VARIANCE":(u.ph/u.s)**2
    }

    file_dict["FILE"] = []

    for ell, filename in enumerate(file_list[extension_exists_mask]):
        file_dict["FILE"].append(filename)
        with fits.open(filename, memmap = False) as hdulist:
            primary_header = hdulist["PRIMARY"].header
            extension_header = hdulist[extension].header
            extension_data = hdulist[extension].data

        for key in primary_keys:
            if ell >0:
                file_dict[key].append(primary_header[key])
            else:
                file_dict[key] = [primary_header[key]]

        # Read in Spectra
        for key in extension_data.dtype.names:
            if key in unit_dict.keys():
                try:
                    data = extension_data[key] * unit_dict[key]
                except TypeError:
                    data = np.full(99, np.nan) * unit_dict[key]

                if len(data) != 99:
                    filler = np.full(99 - len(data), np.nan)*unit_dict[key]
                    data = np.hstack([filler, data])
            else:
                data = extension_data[key]



            if ell > 0:
                if key in unit_dict.keys():
                    file_dict[key].append(data.value)
                else:
                    file_dict[key].append(data)
            else:
                if key in unit_dict.keys():
                    file_dict[key] = [data.value]
                else:
                    file_dict[key] = [data]
                    

    # for key in unit_dict.keys():
    #     file_dict[key] = np.array(file_dict[key])
    #     file_dict[key] *= unit_dict[key]
    # # Units for Data
    file_dict["VELOCITY"] *= u.km/u.s
    file_dict["DATA"] *= u.ph/u.s
    file_dict["VARIANCE"] *= (u.ph/u.s)**2 

    return file_dict

