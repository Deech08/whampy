import logging

from astropy import units as u
import numpy as np 
import matplotlib.pyplot as plt
from scipy import interpolate

from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle

from extinction import fm07 as extinction_law
from dustmaps.marshall import MarshallQuery

import pandas as pd





def get_scale_height_data(data, track = None, deredden = False, 
                          return_pandas_dataframe = False, 
                          longitude_mask_width = None, 
                          step_size = None, **kwargs):
    """
    Return data needed for scale height analysis
    
    Parameters
    ----------
    data: `skySurvey`
        WHAM skySurvey object of full sky (requires track keyword), or spiral arm section
    track: `str`, optional, must be keyword
        if provided, will apply skySurvey.get_spiral_slice for provided track
        if None, will check for track as an attribute of data
    deredden: `bool`, `dustmap`, optional, must be keyword
        if True, will apply dereddening using 3D dustmaps of Marshall et al. (2006)
        or can input a dustmap to query from using the dustmaps package
        default to `dustmaps.marshall.MarshallQuery`
        Warning: Currently only supports Marshall Dustmap
    return_pandas_dataframe: `bool`, optional, must be keyword
        if True, returns pandas dataframe with subset of data specific to scale height analysis
    longitude_mask_width: `number`, `u.Quantity`, optional, must be keyword
        if provided, returns list of masks splitting data into sky sections 
        of given width in degrees
    step_size: `number`, `u.Quantity`, optional, must be keyword
        if provided, sets step_size for longitude masks
        default to half width
    **kwargs: `dict`
        keywords passed to data.get_spiral_slice if track is provided
    """
    # Check Wrapping
    if "wrap_at_180" in kwargs:
        if kwargs["wrap_at_180"]:
            wrap_at = "180d"
        else:
            wrap_at = "360d"
    else:
        wrap_at = "360d"
        
    # Must have return_track
    try:
        test = np.all(kwargs["return_track"])
    except KeyError:
        kwargs["return_track"] = True
    finally: 
        if kwargs["return_track"] is False:
            kwargs["return_track"] = True
            logging.warning("keyword 'return_track' must be set to True!")
    
    # Get / ensure proper track is available
    if (track is None):
        if not hasattr(data, "lbv_RBD_track"):
            raise SyntaxError("No track provided - distance information is required")
    else:
        data, data.lbv_RBD_track = data.get_spiral_slice(track = track, **kwargs)
    
    if data.lbv_RBD_track.shape[1] < 3:
        raise ValueError("provided track does not have distance information!")
    
    # Setup dustmaps if needed
    if deredden.__class__ is bool:
        if deredden:
            deredden = MarshallQuery()
    elif not hasattr(deredden, "query"):
        raise TypeError("invaled dustmap provided - must provide a dustmap class that can query or set to \
                        True to set defualt dustmap to MarshallQuery.")
        
    data["tan(b)"] = np.tan(np.round(data["GAL-LAT"], decimals = 1)*u.deg)
    
    # Apply dereddening
    if not deredden.__class__ is bool:
        # Get all distances assuming plane parallel
        distance_interpolator = interpolate.interp1d(Angle(data.lbv_RBD_track[:,0]*u.deg).wrap_at(wrap_at), 
                                                     data.lbv_RBD_track[:,-1])
        distances = distance_interpolator(Angle(data["GAL-LON"].data*u.deg).wrap_at(wrap_at))
        coordinates = data.get_SkyCoord(distance = distances * u.kpc)
        
        # Get A_Ks
        AKs = deredden(coordinates)
        wave_Ks = 2.17 *u.micron
        A_KS_to_A_v = 1. / extinction_law(np.array([wave_Ks.to(u.AA).value]), 1.)
        wave_ha = np.array([6562.8])
        A_V_to_A_ha = extinction_law(wave_ha, 1.)
        data["DISTANCE"] = distances * u.kpc
        data["Z"] = data["DISTANCE"] * data["tan(b)"]
        data["Av"] = A_KS_to_A_v * AKs
        data["INTEN_DERED"] = data["INTEN"][:]
        data["INTEN_DERED"][~np.isnan(AKs)] = \
                data["INTEN"][~np.isnan(AKs)] * 10**(0.4 * A_V_to_A_ha * A_KS_to_A_v * AKs[~np.isnan(AKs)])
    
    if not longitude_mask_width is None:
        if not isinstance(longitude_mask_width, u.Quantity):
            longitude_mask_width *= u.deg
            logging.warning("No units provided for longitude_mask_width, assuming u.deg.")
        if step_size is None:
            step_size = longitude_mask_width / 2.
        elif not isinstance(step_size, u.Quantity):
            step_size *= u.deg
            logging.warning("No units provided for step_size, assuming u.deg.")

        # Construct masks
        wrapped_lon = Angle(data["GAL-LON"]).wrap_at(wrap_at)
        lon_range = np.min(wrapped_lon), np.max(wrapped_lon)
        n_steps = int(np.ceil(np.round(lon_range[1] - lon_range[0]) / step_size))
        print(n_steps)
        lon_edge = np.linspace(lon_range[0], lon_range[1], n_steps)
        lon_edges = np.zeros((len(lon_edge)-1, 2)) * u.deg
        lon_edges[:,0] = lon_edge[:-1] 
        lon_edges[:,1] = lon_edge[:-1] + longitude_mask_width
        masks = [((wrapped_lon < lon_upper) & (wrapped_lon >= lon_lower)) \
                 for (lon_lower, lon_upper) in lon_edges]

    
    
    if return_pandas_dataframe:
        try:
            df = pd.DataFrame({
                "INTEN":data["INTEN"].byteswap().newbyteorder(), 
                "INTEN_DERED":data["INTEN_DERED"].byteswap().newbyteorder(),
                "tan(b)":data["tan(b)"].byteswap().newbyteorder(),
                "GAL-LON":data["GAL-LON"].byteswap().newbyteorder(),
                "GAL-LAT":data["GAL-LAT"].byteswap().newbyteorder(),
                "Av":data["Av"].byteswap().newbyteorder(),
                "DISTANCE":data["DISTANCE"],
                "Z":data["Z"]
            })
        except KeyError:
            df = pd.DataFrame({
                "INTEN":data["INTEN"].byteswap().newbyteorder(), 
                "tan(b)":data["tan(b)"].byteswap().newbyteorder(),
                "GAL-LON":data["GAL-LON"].byteswap().newbyteorder(),
                "GAL-LAT":data["GAL-LAT"].byteswap().newbyteorder(),
            })
        if longitude_mask_width is None:
            return data, df
        else:
            return data, df, masks
    else:
        return data


  
