import logging

from astropy import units as u
import numpy as np 
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
# from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
from spectral_cube import SpectralCube


from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle

from .clickMap import SpectrumPlotter

from .lbvTracks import get_spiral_slice
from .scaleHeight import get_scale_height_data
from .spectralStack import stack_spectra_bootstrap






class SkySurveyMixin(object):
    """
    Mixin class with convenience functions for WHAM data
    """

    def get_SkyCoord(self, **kwargs):
        """
        returns SkyCoord object of coordinates for pointings in Table

        Parameters
        ----------

        **kwargs:
            passed to SkyCoord
        """
        return SkyCoord(l = self["GAL-LON"], b = self["GAL-LAT"], frame = "galactic", **kwargs)

    def get_spectrum(self, coordinate, index = False):
        """
        returns single WHAM pointing closest to provided coordinate

        Parameters
        ----------

        coordinate: 'SkyCoord' or 'list'
            SkyCoord or [lon,lat] list
            assumes Galactic Coordinates and u.deg
        index: 'bool', optional, must be keyword
            if True, returns the index of the closest coordinate 

        """
        if not isinstance(coordinate, SkyCoord):
            if not isinstance(coordinate, u.Quantity):
                logging.warning("No units provided for coordinate, assuming u.deg")
                coordinate *= u.deg
            coordinate = SkyCoord(l = coordinate[0], b = coordinate[1], frame = 'galactic')

        # Ensure coordinate is in proper frame
        coordinate = coordinate.transform_to('galactic')

        closest_ind = coordinate.separation(self.get_SkyCoord()).argmin()

        if index:
            return closest_ind
        else:
            return self[closest_ind]

    def moment(self, order = None, vmin = None, vmax = None, 
        return_sigma = False, masked = False):
        """
        compute moment maps 

        Parameters
        ----------
        order: 'number', optional, must be keyword
            moment order to return, default = 0
        vmin: 'number' or 'Quantuty', optional, must be keyword
            min Velocity, default units of km/s
        vmax: 'number' or 'Quantity', optional, must be keyword
            max Velocity, default units of km/s
        return_sigma: 'bool', optional, must be keyword
            if True, will also return one-sigma gaussian error estimate
        masked: `bool`, optional, must be keyword
            if True, used masked velocity axis
        """

        if order is None:
            order = 0 # Assume default value

        # Mask out nan values
        nan_msk = np.isnan(self["DATA"]) | np.isnan(self["VELOCITY"])
        # Mask out negative data values
        nan_msk |= self["DATA"] < 0.

        # masked velocity axis
        if masked:
            nan_msk |= np.invert(self["VEL_MASK"])

        if return_sigma:
            nan_msk |= np.isnan(self["VARIANCE"])

        # Velocity mask if applicable:
        if vmin is not None:
            if not isinstance(vmin, u.Quantity):
                logging.warning("No units specified for vmin, assuming u.km/u.s")
                vmin *= u.km/u.s

            nan_msk |= self["VELOCITY"] <= vmin.to(u.km/u.s).value

        if vmax is not None:
            if not isinstance(vmax, u.Quantity):
                logging.warning("No units specified for vmax, assuming u.km/u.s")
                vmax *= u.km/u.s

            nan_msk |= self["VELOCITY"] >= vmax.to(u.km/u.s).value

        data_masked = np.ma.masked_array(self["DATA"], mask = nan_msk)
        vel_masked = np.ma.masked_array(self["VELOCITY"], mask = nan_msk)
        var_masked = np.ma.masked_array(self["VARIANCE"], mask = nan_msk)

        # Zeroth Order Moment
        moment_0 = np.trapz(data_masked, x = vel_masked, 
            axis = 1) * self["DATA"].unit * self["VELOCITY"].unit
        err_0 = np.trapz(np.sqrt(var_masked), x = vel_masked, axis = 1) * self["DATA"].unit * self["VELOCITY"].unit

        if order > 0:
            moment_1 = np.trapz(data_masked * vel_masked, x = vel_masked, 
                axis = 1) * self["DATA"].unit * self["VELOCITY"].unit**2 / moment_0
            err_1_subover_mom_1 = np.trapz(np.sqrt(var_masked) * vel_masked**2, x = vel_masked, 
                            axis = 1) * self["DATA"].unit * self["VELOCITY"].unit**2 / (moment_1 * moment_0)
            err_1 = moment_1 * np.sqrt(err_1_subover_mom_1**2 + (err_0 / moment_0)**2)
            if order > 1:
                moment_2 = np.trapz(data_masked * (vel_masked - moment_1.value[:,None])**2, 
                    x = vel_masked, 
                    axis = 1) * self["DATA"].unit * self["VELOCITY"].unit**3 / moment_0
                print(moment_2.unit)
                err_2_subover_mom2 = np.trapz(data_masked * (vel_masked - moment_1.value[:,None])**2 * np.sqrt(var_masked / 
                                data_masked**2 + 2*(err_1[:,None] / moment_1[:,None])**2), 
                    x = vel_masked, 
                    axis = 1) * self["DATA"].unit * self["VELOCITY"].unit**3 / (moment_2 * moment_0)
                print(err_2_subover_mom2.unit)
                err_2 = moment_2 * np.sqrt(err_2_subover_mom2**2 + (err_0 / moment_0)**2)
                if return_sigma:
                    return moment_2, err_2
                else:
                    return moment_2
            else:
                if return_sigma:
                    return moment_1, err_1
                else:
                    return moment_1
        else:
            if return_sigma:
                return moment_0, err_0
            else:
                return moment_0


    # Plotting Functions
    def intensity_map(self, fig = None, ax = None, lrange = None, brange = None, 
                        vel_range = None,
                        s_factor = 1., colorbar = False, cbar_kwargs = {}, 
                        return_sc = False, **kwargs):
        """
        plots the intensity map using WHAM beams as scatter plots

        Parameters
        -----------
        
        fig: 'plt.figure', optional, must be keyword
            if provided, will create axes on the figure provided
        ax: 'plt.figure.axes' or 'cartopy.axes`, optional, must be keyword
            if provided, will plot on these axes
            can provide ax as a cartpy projection that contains different map projections
        lrange: 'list', optional, must be keyword
            provides longitude range to plot
        brange: 'list', optional, must be keyword
            provides latitude range to plot
        vel_range: 'list' or 'Quantity', optional, must be keyword
            velocity range to integrate data over
        s_factor: 'number', optional, must be keyword
            multiplied by supplied default s value to set size of WHAM beams
        colorbar: 'bool', optional, must be keyword
            if True, plots colorbar
        cbar_kwargs: 'dict', optional, must be keyword
            dictionary of kwargs to pass to colorbar
        return_sc: 'bool', optional, must be keyword
            if True, returns ax.scatter object before figure
        **kwarrgs: `dict`
            passed to scatter plot

        """
        if not hasattr(ax, 'scatter'):
            if not hasattr(fig, 'add_subplot'):
                fig = plt.figure()
                ax = fig.add_subplot(111)
            else:
                ax = fig.add_subplot(111)
        else:
            if not hasattr(fig, 'add_subplot'):
                fig = plt.gcf()


        wham_coords = self.get_SkyCoord()

        if lrange is None:
            lrange_s = [wham_coords.l.wrap_at("180d").max().value, wham_coords.l.wrap_at("180d").min().value]
        elif isinstance(lrange, u.Quantity):
            lrange = Angle(lrange).wrap_at("180d").value
        else:
            logging.warning("No units provided for lrange, assuming u.deg")
            lrange = Angle(lrange*u.deg).wrap_at("180d").value
        if brange is None:
            brange_s = [wham_coords.b.min().value, wham_coords.b.max().value]
        elif isinstance(brange, u.Quantity):
            brange = brange.to(u.deg).value

        if not "s" in kwargs:
            size = fig.get_size_inches()*fig.dpi
            if brange is not None:
                brange_s = brange
            if lrange is not None:
                lrange_s = lrange
            s = np.min([size / np.abs(np.diff(lrange_s)), size / np.abs(np.diff(brange_s))]) * s_factor
            kwargs["s"] = s

        if not "c" in kwargs:
            if vel_range is None:
                kwargs["c"] = self["INTEN"]
            elif not isinstance(vel_range, u.Quantity):
                logging.warning("No units provided for vel_range, assuming u.km/u.s")
                vel_range *= u.km/u.s
            else:
                vel_range = vel_range.to(u.km/u.s)
                kwargs["c"] = self.moment(order = 0, vmin = vel_range.min(), vmax = vel_range.max())

        if not "cmap" in kwargs:
            kwargs["cmap"] = 'plasma'

        if not "vmin" in kwargs:
            kwargs["vmin"] = 0.1

        if not "vmax" in kwargs:
            kwargs["vmax"] = 100.

        if not "norm" in kwargs:
            kwargs["norm"] = LogNorm()
        if hasattr(ax, "coastlines"):
            if not "transform" in kwargs:
                kwargs["transform"] = ccrs.PlateCarree()
                print("No transform specified with cartopy axes projection, assuming PlateCarree")

        lon_points = wham_coords.l.wrap_at("180d")
        lat_points = wham_coords.b.wrap_at("180d")

        if hasattr(ax, "wcs"):
            if ax.wcs.naxis == 3:
                lon_points, lat_points, _ = ax.wcs.wcs_world2pix(lon_points, lat_points, np.zeros_like(lon_points.value), 0)
            elif ax.wcs.naxis == 2:
                lon_points, lat_points = ax.wcs.wcs_world2pix(lon_points, lat_points, 0)



        # Plot the WHAM beams
        sc = ax.scatter(lon_points, lat_points, **kwargs)

        if not hasattr(ax, "coastlines"):
            if (lrange is not None) & (brange is not None):
                ax.set_xlim(lrange)
                ax.set_ylim(brange)
            else:
                ax.invert_xaxis()
            ax.set_xlabel("Galactic Longitude (deg)", fontsize = 12)
            ax.set_ylabel("Galactic Latitude (deg)", fontsize = 12)
        else:
            ax.invert_xaxis()
            if (lrange is not None) & (brange is not None):
                ax.set_extent([lrange[0], lrange[1], brange[0], brange[1]])   
            try:
                ax.gridlines(draw_labels = True)
            except TypeError:
                ax.gridlines()



        if colorbar:
            if not "label" in cbar_kwargs:
                cbar_kwargs["label"] = "H-Alpha Intensity ({})".format(kwargs["c"].unit)

            cb = plt.colorbar(sc, **cbar_kwargs)

        if return_sc:
            return sc, fig
        else:
            return fig

    def click_map(self, fig = None, image_ax = None, spec_ax = None, 
                    projection = None, spectra_kwargs = {}, 
                    over_data = None, average_beam = False, 
                    radius = None, over_spectra_kwargs = {}, 
                    over_spec_ax = None, share_yaxis = False,
                     **kwargs):
                    
        """
        Interactive plotting of WHAM data to plot spectra when clicking on map

        Parameters
        ----------

        fig: 'plt.figure', optional, must be keyword
            if provided, will create axes on the figure provided
        image_ax: 'plt.figure.axes' or 'cartopy.axes`, optional, must be keyword
            if provided, will plot image/map on these axes
            can provide ax as a cartpy projection that contains different map projections 
        spec_ax: 'plt.figure.axes', optional, must be keyword
            if provided, will plot spectra on these axes
        projection:   'ccrs.projection'
            if provided, will be passed to creating a map with specified cartopy projection
        spectra_kwargs: 'dict', optional, must be keyword
            kwargs passed to plot command for spectra
        over_data:  'SkySurvey' or 'str' or 'spectral_cube.SpectralCube', optional, must be keyword
            Extra data to over plot spectra from
            if SkySurvey, assumed to be WHAM observations, perhaps at another wavelength
            if 'str', assumed to be a 3D FITS Data cube filename to be loaded as SpectralCube
            if 'SpectralCube', defaults to extracting closest spectra to click
        average_beam: 'bool', optional, must be keyword
            if True, instead over plots average spectrum from over_data within nearest WHAM beam
        radius: 'Quantity' or  'number', optional, must be keyword
            beam radius to average beam over if avreage_beam is True
            default is 0.5 degrees (WHAM beam)
        over_spectra_kwargs: 'dict', optional, must be keyword
            kwargs passed to plot command for over_spectra
        over_spec_ax: 'plt.figure.axes', optional, must be keyword
            if provided, will plot over_spectra on these axes
        share_yaxis: 'bool', optional, must be keyword
            if True, over_spectra shares same y_axis as spectra
            if False, over_spec_ax has a unique y_axis
            if over_spec_ax is provided, share_yaxis is not used
        **kwargs: 'dict', must be keywords
            passed to `SkySurvey.intensity_map`
        """
        if not hasattr(image_ax, 'scatter'):
            if not hasattr(fig, 'add_subplot'):
                # Create Figure and subplot for image
                fig = plt.figure()
            if projection is not None:
                image_ax = fig.add_subplot(111, projection = projection)
            else:
                image_ax = fig.add_subplot(111)
        if not hasattr(spec_ax, 'scatter'):
            # make room for spectra at bottom
            fig.subplots_adjust(bottom = 0.5)
            spec_ax = fig.add_axes([0.1, .1, .8, .3])
        if not hasattr(over_spec_ax, 'scatter'):
            if over_data is not None:
                if share_yaxis:
                    # Same axis for both spectra
                    over_spec_ax = spec_ax
                else:
                    # Shared x axis but unique y axes
                    over_spec_ax = spec_ax.twinx()


        # Plot image
        fig = self.intensity_map(fig = fig, ax = image_ax, **kwargs)

        # Check for and assign default plot parameters
        if ("lw" not in spectra_kwargs) & ("linewidth" not in spectra_kwargs):
            spectra_kwargs["lw"] = 2
        if ("c" not in spectra_kwargs) & ("color" not in spectra_kwargs):
            spectra_kwargs["c"] = 'red'
        if ("ls" not in spectra_kwargs) & ("linestyle" not in spectra_kwargs):
            spectra_kwargs["ls"] = '-'



        # Empty line
        spec, = spec_ax.plot([0],[0], **spectra_kwargs)

        # Over Plot extra Spectra if needed
        if over_data is not None:
            if ("lw" not in over_spectra_kwargs) & ("linewidth" not in over_spectra_kwargs):
                over_spectra_kwargs["lw"] = 1
            if ("c" not in over_spectra_kwargs) & ("color" not in over_spectra_kwargs):
                over_spectra_kwargs["c"] = 'blue'
            if ("ls" not in over_spectra_kwargs) & ("linestyle" not in over_spectra_kwargs):
                over_spectra_kwargs["ls"] = '--'

            # Empty line 
            over_spec, = over_spec_ax.plot([0],[0], **over_spectra_kwargs)
        else:
            over_spec = None

        return SpectrumPlotter(image_ax, spec, 
                                data = self, 
                                over_data = over_data, 
                                over_line = over_spec, 
                                average_beam = average_beam, 
                                radius = radius)

    def get_spiral_slice(self, **kwargs):
        """
        Returns SkySurvey object isolated to velocity ranges corresponding to specified spiral arm

        Parameters
        ----------
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
        """
        return get_spiral_slice(self, **kwargs)

    def get_scale_height_data(self, **kwargs):
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
        **kwargs: `dict`
            keywords passed to data.get_spiral_slice if track is provided
        """
        return get_scale_height_data(self, **kwargs)

    def stack_spectra_bootstrap(self, **kwargs):
        """
        Stack all spectra in provided SkySurvey Table
        
        Parameters
        ----------
        survey: 'whampy.skySurvey'
            input skySurvey
        data_column: 'str', optional, must be keyword
            Name of data column, default of "DATA"
        variance_column: 'str', optional, must be keyword
            Name od variance column, defaut of "VARIANCE"
        velocity: 'np.array, list', optional, must be keyword
            velocity to interpolate data to for stacking
            defaults to np.round of first row "VELOCITY"
        no_interp: 'bool', optional, must be keyword
            if True, skips interpolating data to same velocity
            assumes it is already regularly gridded to given velocity
        n_boot: 'int', optional, must be keyword
            number of bootstrap samples for each spectrum to stack
            default of 1000
        ci: 'number', optional, must be keyword
            Size of confidence interval to return as errors
        estimator: 'callable', optional, must be keyword
            estimator method to use, defaults to `numpy.mean`
            must take axis keyword
        set_name: 'str', optional, must be keyword
            if provided, sets "NAME" column to this in returned stack
        """
        return stack_spectra_bootstrap(self, **kwargs)















