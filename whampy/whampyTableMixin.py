import logging

from astropy import units as u
import numpy as np 
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
# from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs



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


    """
    def __init__(self, image_ax, line, data = None):
        self.line = line
        self.image_ax = image_ax
        self.line_ax = line.axes
        if hasattr(self.image_ax, 'coastlines'):
            self.geo = True
        else:
            self.geo = False
        self.data = data
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
                # Create SKyCoord
                click_coord = SkyCoord(l = lon*u.deg, b = lat*u.deg, frame = 'galactic')
                # Find closest Spectrum index
                closest = self.data.get_spectrum(click_coord, index = True)
                galCoord = self.data.get_SkyCoord()[closest]

                # Update line to be the spectral data
                self.line.set_data(self.data[closest]["VELOCITY"], 
                                   self.data[closest]["DATA"])
                self.line_ax.relim()
                self.line_ax.autoscale_view() # Rescale axes to cover data

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

        # Velocity mask if applicable:
        if vmin is not None:
            if not isinstance(vmin, u.Quantity):
                logging.warning("No units specified for vmin, assuming u.km/u.s")
                vmin *= u.km/u.s

            nan_msk &= np.invert(self["VELOCITY"] >= vmin.to(u.km/u.s).value)

        if vmax is not None:
            if not isinstance(vmax, u.Quantity):
                logging.warning("No units specified for vmax, assuming u.km/u.s")
                vmax *= u.km/u.s

            nan_msk &= np.invert(self["VELOCITY"] <= vmax.to(u.km/u.s).value)

        data_masked = np.ma.masked_array(self["DATA"], mask = nan_msk)
        vel_masked = np.ma.masked_array(self["VELOCITY"], mask = nan_msk)
        var_masked = np.ma.masked_array(self["VARIANCE"], mask = nan_msk)

        # Zeroth Order Moment
        moment_0 = np.trapz(data_masked, x = vel_masked, 
            axis = 1) * self["DATA"].unit * self["VELOCITY"].unit
        var_0 = np.trapz(var_masked, x = vel_masked**2, 
                axis = 1) * self["VARIANCE"].unit * self["VELOCITY"].unit**2

        if order > 0:
            moment_1 = np.trapz(data_masked * vel_masked, x = vel_masked, 
                axis = 1) * self["DATA"].unit * self["VELOCITY"].unit**2 / moment_0
            var_1_subover_mom_1 = np.sqrt(np.trapz(var_masked * vel_masked**2, x = vel_masked**2, 
                            axis = 1) * self["VARIANCE"].unit * self["VELOCITY"].unit**4) / (moment_1 * moment_0)
            err_1 = moment_1 * (var_1_subover_mom_1 + np.sqrt(var_0) / moment_0)
            if order > 1:
                moment_2 = np.trapz(data_masked * (vel_masked - moment_1.value[:,None])**2, 
                    x = vel_masked, 
                    axis = 1) * self["DATA"].unit * self["VELOCITY"].unit**3 / moment_0
                var_2_subover_mom_2 = np.sqrt(np.trapz(var_masked * (vel_masked - moment_1.value[:,None])**4, 
                    x = vel_masked**2, 
                    axis = 1) * self["VARIANCE"].unit * self["VELOCITY"].unit**6) / (moment_2 * moment_0)
                err_2 = moment_2 * (var_2_subover_mom_2 + np.sqrt(var_0) / moment_0)
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
                return moment_0, np.sqrt(var_0)
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



        # Plot the WHAM beams
        sc = ax.scatter(wham_coords.l.wrap_at("180d"), wham_coords.b.wrap_at("180d"), **kwargs)

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
                    projection = None, spectra_kwargs = {}, **kwargs):
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
            if provided, will plot apectra on these axes
        projection:   'ccrs.projection'
            if provided, will be passed to creating a map with specified cartopy projection
        spectra_kwargs: 'dict', optional, must be keyword
            kwargs passed to plot command for spectra
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

        return SpectrumPlotter(image_ax, spec, data = self)

        # return fig













