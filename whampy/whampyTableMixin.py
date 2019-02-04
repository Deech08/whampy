import logging

from astropy import units as u
import numpy as np 
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
# from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs



from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle

# from mpl_toolkits.basemap import Basemap



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

    # Plotting Functions
    def intensity_map(self, fig = None, ax = None, lrange = None, brange = None, 
                        s_factor = 1., colorbar = False, cbar_kwargs = {}, 
                        return_sc = False, **kwargs):
        """
        plots the intensity map using WHAM beams as scatter plots

        Parrameters
        -----------
        
        fig: 'plt.figure', optional, must be keyword
            if provided, will create axes on the figure provided
        ax: 'plt.figure.axes' or 'mpl_toolkits.basemap.Basemap, optional, must be keyword
            if provided, will plot on these axes
            can provide ax as a cartpy projection that contains different map projections
        lrange: 'list', optional, must be keyword
            provides longitude range to plot
        brange: 'list', optional, must be keyword
            provides latitude range to plot
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
            lrange = [wham_coords.l.wrap_at("180d").max().value, wham_coords.l.wrap_at("180d").min().value]
        elif isinstance(lrange, u.Quantity):
            lrange = Angle(lrange).wrap_at("180d").value
        else:
            logging.warning("No units provided for lrange, assuming u.deg")
            lrange = Angle(lrange*u.deg).wrap_at("180d").value
        if brange is None:
            brange = [wham_coords.b.min().value, wham_coords.b.max().value]
        elif isinstance(brange, u.Quantity):
            brange = brange.to(u.deg).value

        if not "s" in kwargs:
            size = fig.get_size_inches()*fig.dpi
            s = np.min([size / np.abs(np.diff(lrange)), size / np.abs(np.diff(brange))]) * s_factor
            kwargs["s"] = s

        if not "c" in kwargs:
            kwargs["c"] = self["INTEN"]

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
            ax.set_xlim(lrange)
            ax.set_ylim(brange)
            ax.set_xlabel("Galactic Longitude (deg)", fontsize = 12)
            ax.set_ylabel("Galactic Latitude (deg)", fontsize = 12)
        else:
            ax.set_extent([lrange[0], lrange[1], brange[0], brange[1]])
            if lrange[0] > lrange[1]:
                ax.invert_xaxis()
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




