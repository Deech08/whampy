import logging

from astropy import units as u
from astropy.coordinates import Angle

import numpy as np 

from scipy.interpolate import interp1d

try:
    from spectral_cube import SpectralCube
except ModuleNotFoundError:
    # Error handling
    pass


import os.path 

directory = os.path.dirname(__file__)

track_dict = {
    "3kf":"3kF_lbvRBD.dat", "3kpc_far":"3kF_lbvRBD.dat", 
    "3kn":"3kN_lbvRBD.dat", "3kpc_near":"3kN_lbvRBD.dat", 
    "aqr":"AqR_lbvRBD.dat", "aquila_rift":"AqR_lbvRBD.dat", 
    "aqs":"AqS_lbvRBD.dat", "aquila_spur":"AqS_lbvRBD.dat", 
    "cnn":"CnN_lbvRBD.dat", "connecting_near":"CnN_lbvRBD.dat", 
    "cnx":"CnX_lbvRBD.dat", "connecting_extension":"CnX_lbvRBD.dat", 
    "crn":"CrN_lbvRBD.dat", "carina_near":"CrN_lbvRBD.dat", 
    "crf":"CrF_lbvRBD.dat", "carina_far":"CrF_lbvRBD.dat", 
    "ctn":"CtN_lbvRBD.dat", "centaurus_near":"CtN_lbvRBD.dat", 
    "ctf":"CtF_lbvRBD.dat", "centaurus_far":"CtF_lbvRBD.dat", 
    "loc":"Loc_lbvRBD.dat", "local":"Loc_lbvRBD.dat", 
    "los":"LoS_lbvRBD.dat", "local_spur":"LoS_lbvRBD.dat", 
    "nor":"Nor_lbvRBD.dat", "norma":"Nor_lbvRBD.dat", 
    "osc":"OSC_lbvRBD.dat", "outer_centaurus":"OSC_lbvRBD.dat", 
        "outer_scutum":"OSC_lbvRBD.dat", "outer_scutum_centaurus":"OSC_lbvRBD.dat", 
    "outer":"Out_lbvRBD.dat", "out":"Out_lbvRBD.dat", 
    "per":"Per_lbvRBD.dat", "perseus":"Per_lbvRBD.dat", 
    "scn":"ScN_lbvRBD.dat", "scutum_near":"ScN_lbvRBD.dat", 
    "scf":"ScF_lbvRBD.dat", "scutum_far":"ScF_lbvRBD.dat", 
    "sgn":"SgN_lbvRBD.dat", "sagittarius_near":"SgN_lbvRBD.dat",
    "sgf":"SgF_lbvRBD.dat", "sagittarius_far":"SgF_lbvRBD.dat"
}

def get_lbv_track(filename = None, reid_track = None, **kwargs):
    """
    returns (longitude, latitude, velocity, radius, beta, distance) for spiral arm tracks

    Parameters
    ----------
    filename:'str', optional, must be keyword
        Filename of custom LBV track to be read in
    reid_track:'str', optional, must be keyword
        Spiral Arm track from Reid et al. (2016) to read in
        Options are not case sensitive:
            3kpc Arm:
                Far: "3kf", "3kpc_far"
                Near: "3kn", "3kpc_near"
            Aquila Rift: "AqR", "Aquila_Rift"
            Aquila Spur: "AqS", "Aquila_Spur"
            Connecting Arm:
                Near: "CnN", "Connecting_Near"
                Extension: "CnX", "Connecting_Extension"
            Carina Arm:
                Near: "CrN", "Carina_Near"
                Far: "CrF", "Carina_Far"
            Centaurus Arm:
                Near: "CtN", "Centaurus_Near"
                Far: "CtF", "Centaurus_Far"
            Local Arm:
                "Loc", "Local"
                Spur?: "LoS", "Local_Spur"
            Norma Arm:
                "Nor", "Norma"
            Outer Scutum Centaurus Arm:
                "OSC", "Outer_Scutum", "Outer_Centaurus", "Outer_Scutum_Centaurus"
            Outer Arm:
                "Outer", "Out"
            Perseus Arm:
                "Per", "Perseus"
            Scutum Arm:
                Near: "ScN", "Scutum_Near"
                Far: "ScF", "Scutum_Far"
            Sagittarius Arm:
                Near: "SgN", "Sagittarius_Near"
                Far: "SgF", "Sagittarius_Far"
    kwargs:
        optional keywords passed to np.genfromtxt
    """
    # Check for Reid Track Match:
    if reid_track is not None:
        try:
            filename = os.path.join(directory, "data", "Reid16_SpiralArms", track_dict[reid_track.lower()])
        except KeyError:
            raise KeyError("Reid et al. (2016) Spiral Arm track for {} was not found!".format(reid_track))
    else: 
        logging.warning("Not using provided data files - expecting columns to be ordered as lbv_RBD.")

    # Try reading in data:
    if not "comments" in kwargs:
        # Default comments flag in Reid lbv_RBD Tracks
        kwargs["comments"] = "!"

    track = np.genfromtxt(filename, **kwargs)

    # Check shape
    if track.shape[1] != 6:
        logging.warning("Track file did not have expected number of columns - adding in columns of 0s")
        new_track = np.zeros((track.shape[0], 6))
        for column in range(track.shape[1]):
            new_track[:,column] = track[:,column]

        track = new_track

    return track


def get_spiral_slice(survey, track = None, filename = None, 
    brange = None, lrange = None, 
    interpolate = True, wrap_at_180 = False, 
    vel_width = None, return_track = False):
    """
    Returns SkySurvey object isolated to velocity ranges corresponding to specified spiral arm

    Parameters
    ----------
    survey: 'whampy.skySurvey'
        input skySurvey
    track: 'np.ndarray', 'str', optional, must be keyword
        if 'numpy array', lbv_RBD track data
        if 'str', name of spiral arm from Reid et al. (2016)
    filename: 'str', optional, must be keyword
        filename of track file to read in using whampy.lbvTracks.get_lbv_track
    brange: 'list', 'u.Quantity'
        min,max latitude to restrict data to
        Default of +/- 40 deg
    lrange: 'list', 'u.Quantity'
        min,max longitude to restrict data to
        Default of full track extent
    interpolate: 'bool', optional, must be keyword
        if True, interpolates velocity to coordinate of pointings
        if False, ... do nothing ... for now
            Future: slices have velocities set by track 
    wrap_at_180: `bool`, optional, must be keyword
            if True, wraps longitude angles at 180d
            use if mapping accross Galactic Center
    vel_width: `number`, `u.Quantity`, optional, must be keyword
        velocity width to isolate in km/s
    return_track: `bool`, optional, must be keyword
        if True, will also return the spiral arm track as the second element of a tuple
    """
    # Check Track Data
    if track is not None:
        if not track.__class__ is np.ndarray:
            track = get_lbv_track(reid_track = track)
        else:
            if track.shape[1] < 2:
                raise TypeError("track should have at least two columns (l,v) or up to 6 (lbvRBD).")
    elif filename is None:
        raise SyntaxError("No track or track filename provided!")
    else:
        track = get_lbv_track(filename = filename)

    if wrap_at_180:
        wrap_at = "180d"
    else:
        wrap_at = "360d"

    # Check velocity width
    if vel_width is None:
        logging.warning("No Velocity width specified, using default of 16 km/s.")
        vel_width = 16*u.km/u.s
    elif not isinstance(vel_width, u.Quantity):
        vel_width *= u.km/u.s
        logging.warning("No units specified for vel_width, assuming km/s.")

    #Extract Needed track informattion
    if track.shape[1] == 2:
        lon_track = track[:,0] * u.deg
        lon_track = Angle(lon_track).wrap_at(wrap_at)
        vel_track = track[:,1] * u.km/u.s
    else: # Reid et al. (2016) format
        lon_track = track[:,0] * u.deg
        lon_track = Angle(lon_track).wrap_at(wrap_at)
        vel_track = track[:,2] * u.km/u.s


    # Set Default latitude range
    if brange is None:
        brange = [-40,40]*u.deg
    elif isinstance(brange, u.Quantity):
        brange = brange.to(u.deg)
    else: 
        brange = brange * u.deg
        logging.warning("No units specified for latitude range, assuming Degrees!")

    # Set Longitude Range
    if lrange is None:
        lrange = [np.min(lon_track.value), np.max(lon_track.value)]*u.deg
    elif isinstance(lrange, u.Quantity):
        lrange = lrange.to(u.deg)
    else: 
        lrange = lrange * u.deg
        logging.warning("No units specified for longitude range, assuming Degrees!")

    # Extract Relevant sky section
    bounds = [lrange[0], lrange[1], brange[0], brange[1]]
    sky_cut = survey.sky_section(bounds, wrap_at_180 = wrap_at_180)

    # Restrict Velocity Space with a mask
    wham_coords = sky_cut.get_SkyCoord()

    
    if interpolate:
        l_v_track = interp1d(lon_track, vel_track)
        central_velocities = l_v_track(wham_coords.l.wrap_at(wrap_at)) * vel_track.unit
    else:   ## For now - always interpolate
        l_v_track = interp1d(lon_track, vel_track)
        central_velocities = l_v_track(wham_coords.l.wrap_at(wrap_at)) * vel_track.unit

    # Get velocity masks
    # Is there a better way to do this without a loop?
    sky_cut["VEL_MASK"] = np.empty((len(sky_cut), len(sky_cut[0]["VELOCITY"])), dtype = bool)
    for ell, vel in enumerate(central_velocities):
        mask = sky_cut[ell]["VELOCITY"] <= (vel + vel_width/2).value
        mask &= sky_cut[ell]["VELOCITY"] >= (vel - vel_width/2).value
        sky_cut[ell]["VEL_MASK"] = mask

    # Set Intensities
    sky_cut["INTEN"], sky_cut["ERROR"] = sky_cut.moment(return_sigma = True, masked = True)


    # Save Track info as attribute
    sky_cut.lbv_RBD_track = track

    if not return_track:
        return sky_cut
    else:
        return sky_cut, track












