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


class SpectrumPlotter():
    """
    Class to interactivly plot spectra from WHAM maps
    Click on a point on a map and a spectrum will be extracted cooresponding to the 
    closest point on the Sky

    Parameters
    ----------

    image_ax: 'axes` 
        containing the image / map'
    line:   'matplotlib.line.Line2D'
        line that will be updated to match clicked spectra
    data:   'SkySurvey', optional, must be keyword
        WHAM data
    over_data:  'SkySurvey' or 'str' or 'spectral_cube.SpectralCube', optional, must be keyword
        Extra data to over plot spectra from
        if SkySurvey, assumed to be WHAM observations, perhaps at another wavelength
        if 'str', assumed to be a 3D FITS Data cube filename to be loaded as SpectralCube
        if 'SpectralCube', defaults to extracting closest spectra to click
    over_line: matplotlib.line.Line2D', optional, must be keyword
        if over_data, must be provided
        line that will be udpated to match clicked spectra from over_data
    average_beam: 'bool', optional, must be keyword
        if True, instead over plots average spectrum from over_data within nearest WHAM beam
    radius: 'Quantity' or  'number', optional, must be keyword
        beam radius to average beam over if avreage_beam is True
        default is 0.5 degrees (WHAM beam)





    """
    def __init__(self, image_ax, line, data = None, over_data = None, over_line = None, 
        average_beam = False, radius = None):
        self.line = line
        self.image_ax = image_ax
        self.line_ax = line.axes
        self.over_data = over_data
        self.average_beam = average_beam
        self.radius = radius

        if hasattr(self.image_ax, 'coastlines'):
            self.geo = True
        else:
            self.geo = False
        if hasattr(self.image_ax, 'wcs'):
            self.wcs_axes = True
        else:
            self.wcs_axes = False
        self.data = data
        if self.over_data is not None:
            if over_data.__class__ is str:
                self.over_data = SpectralCube.read(over_data)
                self.over_type = "SpectralCube"
            elif hasattr(over_data, 'minimal_subcube'):
                # Checks if it is a SpectralCube or wrapper class of SpectralCube
                self.over_data = over_data
                self.over_type = "SpectralCube"
            elif hasattr(over_data, 'intensity_map'):
                # Checks if it is a SkySurvey object
                self.over_data = over_data
                self.over_type = "SkySurvey"
            if self.average_beam:
                if self.radius is None:
                    self.radius = 0.5 * u.deg
                elif not isinstance(radius, u.Quantity):
                    self.radius = radius * u.deg # Assume Default Units
                else:
                    self.radius = radius
        self.over_line = over_line
        self.cid = self.line.figure.canvas.mpl_connect("button_press_event", self.on_click)
    def on_click(self, event):
        if event.button is 1: # left mouse click
            if event.inaxes is not self.image_ax: 
                return # if click is not on image, nothing happens
            else:
                lon = event.xdata
                lat = event.ydata
                if self.geo: # if cartopy axes with projection info
                    # Convert coordinates to standard longitude and latitude
                    lon, lat = ccrs.PlateCarree().transform_point(lon, lat, self.image_ax.projection)
                elif self.wcs_axes:
                    # Covert coordinates from pixel to world
                    lon, lat, _ = self.image_ax.wcs.wcs_pix2world(lon, lat, 0, 0)

                # Create SKyCoord
                click_coord = SkyCoord(l = lon*u.deg, b = lat*u.deg, frame = 'galactic')
                # Find closest Spectrum index
                closest = self.data.get_spectrum(click_coord, index = True)
                galCoord = self.data.get_SkyCoord()[closest]

                # Update line to be the spectral data
                self.line.set_data(self.data[closest]["VELOCITY"], 
                                   self.data[closest]["DATA"])
                if self.over_line is not None:
                    if self.over_line.axes == self.line_ax:
                        self.over_line.set_data([0],[0])
                self.line_ax.relim()
                self.line_ax.autoscale_view() # Rescale axes to cover data

                # Draw over_line if needed:
                if self.over_line is not None:
                    if self.over_type == 'SkySurvey':
                        closest = self.over_data.get_spectrum(click_coord, index = True)
                        galCoord = self.data.get_SkyCoord()[closest]

                        # Update line to be the spectral data
                        self.over_line.set_data(self.over_data[closest]["VELOCITY"], 
                                           self.over_data[closest]["DATA"])
                        self.over_line.axes.relim()
                        self.over_line.axes.autoscale_view() # Rescale axes to cover data

                    elif self.over_type == 'SpectralCube':
                        if not self.average_beam:
                            # find index of closest value
                            _, lat_axis_values, _ = self.over_data.world[int(self.over_data.shape[0]/2), :, 
                                                                            int(self.over_data.shape[2]/2)]
                            lat_slice = np.nanargmin(np.abs(lat_axis_values-click_coord.b))

                            _, _, lon_axis_values = self.over_data.world[int(self.over_data.shape[0]/2), 
                                                                            int(self.over_data.shape[1]/2), :]
                            # Ensure all angles are wrapped at 180
                            lon_axis_values = Angle(lon_axis_values).wrap_at("180d")
                            lon_slice = np.nanargmin(np.abs(lon_axis_values-click_coord.l.wrap_at("180d")))
                            self.over_line.set_data(self.over_data.spectral_axis.to(u.km/u.s).value, 
                                                    self.over_data.unmasked_data[:,lat_slice,lon_slice].value)
                        else:
                            ds9_str = 'Galactic; circle({0:.3}, {1:.4}, {2:.4}")'.format(click_coord.l.wrap_at("180d").value, 
                                                                         click_coord.b.value, 
                                                                         self.radius.to(u.arcsec).value)
                            _, lat_axis_values, _ = self.over_data.world[int(self.over_data.shape[0]/2), :, int(self.over_data.shape[2]/2)]
                            lat_slice_up = np.nanargmin(np.abs(lat_axis_values-click_coord.b+self.radius*1.5))
                            lat_slice_down = np.nanargmin(np.abs(lat_axis_values-click_coord.b-self.radius*1.5))
                            lat_slices = np.sort([lat_slice_up, lat_slice_down])


                            _, _, lon_axis_values = self.over_data.world[int(self.over_data.shape[0]/2), int(self.over_data.shape[1]/2), :]
                            # Ensure all angles are wrapped at 180
                            lon_axis_values = Angle(lon_axis_values).wrap_at("180d")
                            lon_slice_up = np.nanargmin(np.abs(lon_axis_values-click_coord.l.wrap_at("180d")+self.radius*1.5))
                            lon_slice_down = np.nanargmin(np.abs(lon_axis_values-click_coord.l.wrap_at("180d")-self.radius*1.5))
                            lon_slices = np.sort([lon_slice_up, lon_slice_down])


                            smaller_cube = self.over_data[:,lat_slices[0]:lat_slices[1], lon_slices[0]:lon_slices[1]]
                            subcube = smaller_cube.subcube_from_ds9region(ds9_str)
                            spectrum = subcube.mean(axis = (1,2))
                            self.over_line.set_data(spectrum.spectral_axis.to(u.km/u.s).value, 
                                                    spectrum.value)
                        if self.over_line.axes != self.line_ax:
                            self.over_line.axes.relim()
                            self.over_line.axes.autoscale_view() # Rescale axes to cover data
                            if self.over_type == 'SkySurvey':
                                self.over_line.axes.set_ylabel("Intensity ({0})".format(self.over_data["DATA"].unit), 
                                                fontsize = 12)
                            else:
                                self.over_line.axes.set_ylabel("Intensity ({0})".format(self.over_data.unmasked_data[0,0,0].unit), 
                                                fontsize = 12)

                # Set Labels
                self.line_ax.set_ylabel("Intensity ({0})".format(self.data["DATA"].unit), 
                                        fontsize = 12)
                self.line_ax.set_xlabel("LSR Velocity ({0}) | (l,b) = ({1:3.1f},{2:3.1f})".format(self.data["VELOCITY"].unit, 
                                                                                                galCoord.l.wrap_at("180d").value, 
                                                                                                galCoord.b.value), 
                                        fontsize = 12)

                # Draw the Line
                self.line.figure.canvas.draw()
                




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

    def moment(self, order = None, vmin = None, vmax = None, return_sigma = False):
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
        """

        if order is None:
            order = 0 # Assume default value

        # Mask out nan values
        nan_msk = np.isnan(self["DATA"]) | np.isnan(self["VELOCITY"])
        # Mask out negative data values
        nan_msk |= self["DATA"] < 0.

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
            lon_points, lat_points, _ = ax.wcs.wcs_world2pix(lon_points, lat_points, np.zeros_like(lon_points.value), 0)



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













