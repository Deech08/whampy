import logging

from astropy import units as u
import numpy as np 
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
# from mpl_toolkits.basemap import Basemap
try:
    import cartopy.crs as ccrs
except ModuleNotFoundError:
    # Error handling
    pass
try:
    from spectral_cube import SpectralCube
except ModuleNotFoundError:
    # Error handling
    pass

from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy.table import Column, Table

from .clickMap import SpectrumPlotter

from .lbvTracks import get_spiral_slice
from .scaleHeight import get_scale_height_data
from .spectralStack import stack_spectra_bootstrap

from scipy.interpolate import griddata, interp1d






class SkySurveyMixin(object):
    """
    Mixin class with convenience functions for WHAM data
    """

    def __truediv__(self,other):
        if hasattr(other,"get_SkyCoord"):
            #Get skycoords
            c_self = self.get_SkyCoord()
            c_other = other.get_SkyCoord()
            match_idx, sep, _ = c_self.match_to_catalog_sky(c_other)
            #check seeparations
            true_match_mask = sep < 0.075 * u.deg
            true_match_idx = match_idx[true_match_mask]
            #new Table
            data = Table()
            data["GAL-LON"] = self["GAL-LON"][true_match_mask]
            data["GAL-LAT"] = self["GAL-LAT"][true_match_mask]

            min_velocity = np.min([np.nanmin(self["VELOCITY"]),
                                              np.nanmin(other["VELOCITY"])])
            max_velocity = np.max([np.nanmax(self["VELOCITY"]),
                                              np.nanmax(other["VELOCITY"])])
            velocity_ax = np.arange(min_velocity,max_velocity+1.5,1.5)

            data["DATA"] = [interp1d(vel_num, data_num, 
                                    bounds_error = False)(velocity_ax) / interp1d(vel_denom, data_denom, 
                                                                                bounds_error = False)(velocity_ax) 
                            for (vel_num, data_num, vel_denom, data_denom) 
                            in zip(self["VELOCITY"][true_match_mask],self["DATA"][true_match_mask], 
                                   other["VELOCITY"][true_match_idx],other["DATA"][true_match_idx])]*u.R / u.km * u.s

            data["VARIANCE"] = data["DATA"] * [np.sqrt(interp1d(vel_num, var_num, 
                                    bounds_error = False)(velocity_ax)/interp1d(vel_num, data_num, 
                                                                                bounds_error = False)(velocity_ax)**2 + interp1d(vel_denom, var_denom, 
                                                                                                                                bounds_error = False)(velocity_ax) /interp1d(vel_denom, data_denom, 
                                                                                                                                                                            bounds_error = False)(velocity_ax)**2) 
                            for (vel_num, data_num, var_num, vel_denom, data_denom, var_denom) 
                            in zip(self["VELOCITY"][true_match_mask],self["DATA"][true_match_mask],self["VARIANCE"][true_match_mask], 
                                   other["VELOCITY"][true_match_idx],other["DATA"][true_match_idx],other["VARIANCE"][true_match_idx])]

            # Just numerator Data
            data["DATA_NUM"] = [interp1d(vel_num, data_num, 
                                    bounds_error = False)(velocity_ax)
                            for (vel_num, data_num) 
                            in zip(self["VELOCITY"][true_match_mask],self["DATA"][true_match_mask])]*u.R / u.km * u.s
            data["VARIANCE_NUM"] = [interp1d(vel_num, var_num, 
                                    bounds_error = False)(velocity_ax)
                            for (vel_num, var_num) 
                            in zip(self["VELOCITY"][true_match_mask],self["VARIANCE"][true_match_mask])]*u.R / u.km * u.s

            # Just denominator data
            data["DATA_DENOM"] = [interp1d(vel_num, data_num, 
                                    bounds_error = False)(velocity_ax)
                            for (vel_num, data_num) 
                            in zip(other["VELOCITY"][true_match_idx],other["DATA"][true_match_idx])]*u.R / u.km * u.s
            data["VARIANCE_DENOM"] = [interp1d(vel_num, var_num, 
                                    bounds_error = False)(velocity_ax)
                            for (vel_num, var_num) 
                            in zip(other["VELOCITY"][true_match_idx],other["VARIANCE"][true_match_idx])]*u.R / u.km * u.s

            data["VELOCITY"] = [velocity_ax for idx in true_match_idx] * u.km/u.s
            return self.__class__(from_table = data)
        else:
            self["DATA"] = self["DATA"]/other
            self["VARIANCE"] = self["VARIANCE"]/other**2
            



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

    def get_spectral_slab(self, vmin, vmax):
        """
        returns survey with velocity window cut down

        Parameters
        ----------

        vmin: `number`, `u.Quantity`
            Min velocity
        vmax: `number`, `u.Quantity`
            Max velocity
        """
        if hasattr(vmin, "unit"):
            vmin = vmin.to(u.km/u.s).value
        if hasattr(vmax, "unit"):
            vmax = vmax.to(u.km/u.s).value

        velocity_cut_mask = self["VELOCITY"].data >= vmin
        velocity_cut_mask &= self["VELOCITY"].data <= vmax

        if velocity_cut_mask.sum() == 0:
            raise ValueError("No Data within specified velocity range!")
        else:
            masked_velocity_entries = np.sum(velocity_cut_mask, axis = 0, dtype = bool)

            new_velocity_column_full = np.copy(self["VELOCITY"])
            new_velocity_column_full[~velocity_cut_mask] = np.nan
            new_velocity_column = new_velocity_column_full[:,masked_velocity_entries]

            new_data_column_full = np.copy(self["DATA"]) * self["DATA"].unit
            new_data_column = new_data_column_full[:,masked_velocity_entries]

            new_variance_column_full = np.copy(self["VARIANCE"]) * self["VARIANCE"].unit
            new_variance_column = new_variance_column_full[:,masked_velocity_entries]

            self.remove_column("VELOCITY")
            self["VELOCITY"] = new_velocity_column * u.km/u.s

            self.remove_column("VARIANCE")
            self["VARIANCE"] = new_variance_column

            self.remove_column("DATA")
            self["DATA"] = new_data_column

    def bettermoment(self, order = None, vmin = None, vmax = None, 
        return_sigma = False, masked = False, ratio = False, 
        window_length = 5, rms = "auto", return_Fnu = False, 
        traditional = False):
        """
        compute moment maps using bettermoments (quadratic fitting approach; Teague & Foreman-Mackey 2018)

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
        ratio: `bool`, optional, must be keyword
            if True, assumes computnig for a line ratio
        filter_width: `int`, must be keyword
            window_length passed to savgol_filter, must be >2
        rms: `str, number`, optional must be keyword
            if "auto", estimates rms from data, or uses provided value
        return_Fnu: `bool`, optional, must be keyword
            if True, also returns Fnu and dFnu for first moment
        traditional: `bool`, optional, must be keyword
            if True, uses traditional moments for first and second order
        """
        # Make sure package exists and import it
        try:
            from bettermoments import collapse_cube
        except ImportError:
            raise ImportError("Unable to import bettermoments; try again after installing (pip install bettermoments)")

        # Setup smoothing filter
        from scipy.signal import savgol_filter
        window_length = int(window_length)
        assert (window_length > 2) & (window_length%2 == 1)

        if order is None:
            order = 0 # Assume default value

        def smooth_savgol(x, window_length = window_length):
            return savgol_filter(x, window_length, 2, mode='wrap', axis=0)

        if not ratio:
            all_velax = self["VELOCITY"].data
            all_data = self["DATA"].data
            all_var = self["VARIANCE"].data

            if vmin is None:
                vmin = np.nanmin(all_velax)
            if vmax is None:
                vmax = np.nanmax(all_velax)

            vel_masks = all_velax >= vmin
            vel_masks &= all_velax <= vmax
            
            if order == 0:
                if rms.__class__ == str:
                    result = np.array([collapse_cube.collapse_zeroth(velax[mask], 
                                                                     smooth_savgol(data[mask]), 
                                                                     np.median(np.sqrt(var[mask]))) 
                                       for velax, data, var, mask in zip(all_velax, all_data, all_var, vel_masks)])
                else:
                    result = np.array([collapse_cube.collapse_zeroth(velax[mask], 
                                                                     smooth_savgol(data[mask]), 
                                                                     rms) 
                                       for velax, data, mask in zip(all_velax, all_data, vel_masks)])

                return result.T * self["DATA"].unit * self["VELOCITY"].unit

            if order == 1:
                if not traditional:
                    if rms.__class__ == str:
                        result = np.array([collapse_cube.collapse_quadratic(velax[mask], 
                                                                         smooth_savgol(data[mask]), 
                                                                         np.median(np.sqrt(var[mask]))) 
                                           for velax, data, var, mask in zip(all_velax, all_data, all_var, vel_masks)])
                    else:
                        result = np.array([collapse_cube.collapse_quadratic(velax[mask], 
                                                                         smooth_savgol(data[mask]), 
                                                                         rms) 
                                           for velax, data, mask in zip(all_velax, all_data, vel_masks)])

                    v0, dv0, Fnu, dFnu = result.T
                    if not return_Fnu:
                        return v0 * self["VELOCITY"].unit, dv0 * self["VELOCITY"].unit
                    else:
                        return v0 * self["VELOCITY"].unit, dv0 * self["VELOCITY"].unit, Fnu * self["DATA"].unit, dFnu * self["DATA"].unit
                else:
                    if rms.__class__ == str:
                        result = np.array([collapse_cube.collapse_first(velax[mask], 
                                                                         smooth_savgol(data[mask]), 
                                                                         np.median(np.sqrt(var[mask]))) 
                                           for velax, data, var, mask in zip(all_velax, all_data, all_var, vel_masks)])
                    else:
                        result = np.array([collapse_cube.collapse_first(velax[mask], 
                                                                         smooth_savgol(data[mask]), 
                                                                         rms) 
                                           for velax, data, mask in zip(all_velax, all_data, vel_masks)])

        
                    return result.T * self["VELOCITY"].unit


            if order == 2:
                if not traditional:
                    if rms.__class__ == str:
                        result = np.array([collapse_cube.collapse_width(velax[mask], 
                                                                         smooth_savgol(data[mask]), 
                                                                         np.median(np.sqrt(var[mask]))) 
                                           for velax, data, var, mask in zip(all_velax, all_data, all_var, vel_masks)])
                    else:
                        result = np.array([collapse_cube.collapse_width(velax[mask], 
                                                                         smooth_savgol(data[mask]), 
                                                                         rms) 
                                           for velax, data, mask in zip(all_velax, all_data, vel_masks)])

                    return result.T * self["VELOCITY"].unit
                else:
                    if rms.__class__ == str:
                        result = np.array([collapse_cube.collapse_second(velax[mask], 
                                                                         smooth_savgol(data[mask]), 
                                                                         np.median(np.sqrt(var[mask]))) 
                                           for velax, data, var, mask in zip(all_velax, all_data, all_var, vel_masks)])
                    else:
                        result = np.array([collapse_cube.collapse_second(velax[mask], 
                                                                         smooth_savgol(data[mask]), 
                                                                         rms) 
                                           for velax, data, mask in zip(all_velax, all_data, vel_masks)])

                    return result.T * self["VELOCITY"].unit






    def moment(self, order = None, vmin = None, vmax = None, 
        return_sigma = False, masked = False, ratio = False):
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
        ratio: `bool`, optional, must be keyword
            if True, assumes computnig for a line ratio
        """

        if not ratio:

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


            # Caution - Errors not to be trusted...
            # Zeroth Order Moment
            moment_0 = np.trapz(data_masked, x = vel_masked, 
                axis = 1) * self["DATA"].unit * self["VELOCITY"].unit
            err_0 = np.trapz(np.sqrt(var_masked), x = vel_masked, axis = 1) * self["DATA"].unit * self["VELOCITY"].unit

            if order > 0:
                moment_1 = np.trapz(data_masked * vel_masked, x = vel_masked, 
                    axis = 1) * self["DATA"].unit * self["VELOCITY"].unit**2 / moment_0
                err_num = np.trapz(np.sqrt(var_masked) * vel_masked, x = vel_masked, axis = 1) * self["DATA"].unit * self["VELOCITY"].unit**2
                err_denom = err_0

                err_1 = np.sqrt((err_num/(moment_1*moment_0))**2 + (err_denom/moment_0)**2)*np.abs(moment_1)
                # err_1_subover_mom_1 = np.trapz(np.sqrt(var_masked) * vel_masked**2, x = vel_masked, 
                #                 axis = 1) * self["DATA"].unit * self["VELOCITY"].unit**2 / (moment_1 * moment_0)
                # err_1 = moment_1 * np.sqrt(err_1_subover_mom_1**2 + (err_0 / moment_0)**2)
                if order > 1:
                    moment_2 = np.trapz(data_masked * (vel_masked - moment_1.value[:,None])**2, 
                        x = vel_masked, 
                        axis = 1) * self["DATA"].unit * self["VELOCITY"].unit**3 / moment_0

                    err_2_subover_mom2 = np.trapz(data_masked * (vel_masked - moment_1.value[:,None])**2 * np.sqrt(var_masked.value / 
                                    data_masked**2 + 2*(err_1[:,None] / moment_1[:,None])**2), 
                        x = vel_masked, 
                        axis = 1) * self["DATA"].unit * self["VELOCITY"].unit**3 / (moment_2 * moment_0)

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
        else:
            if order > 0:
                raise NotImplementedError

            # Mask out nan values
            nan_msk = np.isnan(self["DATA_NUM"]) | np.isnan(self["VELOCITY"]) | (np.isnan(self["DATA_DENOM"]))
            # Mask out negative data values
            nan_msk |= (self["DATA_NUM"] < 0.) | (self["DATA_DENOM"] < 0.)

            # masked velocity axis
            if masked:
                nan_msk |= np.invert(self["VEL_MASK"])

            if return_sigma:
                nan_msk |= (np.isnan(self["VARIANCE_NUM"])) | (np.isnan(self["VARIANCE_DENOM"]))

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

            data_masked_num = np.ma.masked_array(self["DATA_NUM"], mask = nan_msk)
            data_masked_denom = np.ma.masked_array(self["DATA_DENOM"], mask = nan_msk)
            vel_masked = np.ma.masked_array(self["VELOCITY"], mask = nan_msk)
            var_masked_num = np.ma.masked_array(self["VARIANCE_NUM"], mask = nan_msk)
            var_masked_denom = np.ma.masked_array(self["VARIANCE_DENOM"], mask = nan_msk)


            # Caution - Errors not to be trusted...
            # Zeroth Order Moment
            moment_0_num = np.trapz(data_masked_num, x = vel_masked, 
                axis = 1) * self["DATA_NUM"].unit * self["VELOCITY"].unit
            err_0_num = np.trapz(np.sqrt(var_masked_num), x = vel_masked, axis = 1) * self["DATA_NUM"].unit * self["VELOCITY"].unit

            moment_0_denom = np.trapz(data_masked_denom, x = vel_masked, 
                axis = 1) * self["DATA_DENOM"].unit * self["VELOCITY"].unit
            err_0_denom = np.trapz(np.sqrt(var_masked_num), x = vel_masked, axis = 1) * self["DATA_DENOM"].unit * self["VELOCITY"].unit


            moment_0 = moment_0_num / moment_0_denom
            err_0 = np.sqrt((err_0_num / moment_0_num)**2 + (err_0_denom / moment_0_denom)**2) * moment_0
            if return_sigma:
                return moment_0 * u.R, err_0 * u.R
            else:
                return moment_0 * u.R


    # Plotting Functions
    def intensity_map(self, fig = None, ax = None, lrange = None, brange = None, 
                        vel_range = None,
                        s_factor = 1., colorbar = False, cbar_kwargs = {}, 
                        return_sc = False, 
                        smooth = False, 
                        smooth_res = None, 
                        wrap_at = "180d",
                        ratio = False,
                        **kwargs):
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
        smooth: `bool`, optional, must be keyword
            if True, smooths map using griddata and plots using pcolormesh
        smooth_res: `number`, optional, must be keyword
            pixel width in units of degrees
        wrap_at: 'str', optional, must be keyword - either "180d" or "360d"
            sets value to wrap longitude degree values to
            Defaults to "180d"
        ratio: `bool`, optional, must be keyword
            if True treated as a line ratio
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
            lrange_s = [wham_coords.l.wrap_at(wrap_at).max().value, wham_coords.l.wrap_at(wrap_at).min().value]
        elif isinstance(lrange, u.Quantity):
            lrange = Angle(lrange).wrap_at(wrap_at).value
        else:
            logging.warning("No units provided for lrange, assuming u.deg")
            lrange = Angle(lrange*u.deg).wrap_at(wrap_at).value
        if brange is None:
            brange_s = [wham_coords.b.min().value, wham_coords.b.max().value]
        elif isinstance(brange, u.Quantity):
            brange = brange.to(u.deg).value


        if not smooth:
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
                if "INTEN" in self.keys():
                    kwargs["c"] = self["INTEN"]
                else:
                    kwargs["c"] = self.moment(order = 0, ratio = ratio)
            elif not isinstance(vel_range, u.Quantity):
                logging.warning("No units provided for vel_range, assuming u.km/u.s")
                vel_range *= u.km/u.s
                kwargs["c"] = Column(data = self.moment(order = 0, 
                    vmin = vel_range.min(), 
                    vmax = vel_range.max(), ratio = ratio).to(u.R).value, unit = u.R)
            else:
                vel_range = vel_range.to(u.km/u.s)
                kwargs["c"] = Column(data = self.moment(order = 0, 
                    vmin = vel_range.min(), 
                    vmax = vel_range.max(), ratio = ratio).to(u.R).value, unit = u.R)

        if not "cmap" in kwargs:
            kwargs["cmap"] = 'plasma'

        if not "vmin" in kwargs:
            kwargs["vmin"] = 0.1

        if not "vmax" in kwargs:
            kwargs["vmax"] = 100.

        if not "norm" in kwargs:
            kwargs["norm"] = LogNorm(vmin = kwargs["vmin"], vmax = kwargs["vmax"])
            _ = kwargs.pop('vmin') 
            _ = kwargs.pop('vmax') 
        if hasattr(ax, "coastlines"):
            if not "transform" in kwargs:
                kwargs["transform"] = ccrs.PlateCarree()
                print("No transform specified with cartopy axes projection, assuming PlateCarree")

        lon_points = wham_coords.l.wrap_at(wrap_at)
        lat_points = wham_coords.b.wrap_at("180d")

        if smooth:
            if smooth_res is None:
                smooth_res = 0.2
            if lrange is None:
                lrange = lrange_s
            if brange is None:
                brange = brange_s
            if lrange[1] < lrange[0]:
                gridx = np.flip(np.arange(lrange[1], lrange[0] + smooth_res, smooth_res))
            else:
                gridx = np.arange(lrange[0], lrange[1] + smooth_res, smooth_res)
            gridy = np.arange(brange[0], brange[1] + smooth_res, smooth_res)

            zi = griddata((lon_points, lat_points), kwargs["c"],
                (gridx[None,:], gridy[:,None]), 
                method='cubic')
            
            c_kwarg = kwargs["c"]
            del kwargs["c"]

        if hasattr(ax, "wcs"):
            if ax.wcs.naxis == 3:
                if smooth:
                    gridx, gridy, _ = ax.wcs.wcs_world2pix(gridx, gridy, np.zeros_like(gridx), 0)
                else:
                    lon_points, lat_points, _ = ax.wcs.wcs_world2pix(lon_points, lat_points, np.zeros_like(lon_points.value), 0)
            elif ax.wcs.naxis == 2:
                if smooth:
                    gridx, gridy = ax.wcs.wcs_world2pix(gridx, gridy, 0)
                else:
                    lon_points, lat_points = ax.wcs.wcs_world2pix(lon_points, lat_points, 0)



        # Plot the WHAM beams
        if smooth:
            sc = ax.pcolormesh(gridx, gridy, zi, **kwargs)
        else:
            sc = ax.scatter(lon_points, lat_points, **kwargs)

        if not hasattr(ax, "coastlines"):
            if lrange is not None:
                ax.set_xlim(lrange)
            else:
                ax.invert_xaxis()

            if brange is not None:
                ax.set_ylim(brange)
            
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
                if "c" in kwargs:
                    cbar_kwargs["label"] = "H-Alpha Intensity ({})".format(kwargs["c"].unit)
                else:
                    cbar_kwargs["label"] = "H-Alpha Intensity ({})".format(c_kwarg.unit)


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
                    no_image = False,
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
        no_image: 'bool', optional, must be keyword
            if True, does not plot a WHAM intensity map
            Can instead plot any image using WCS axes or default coordinate axes and click 
            to plot spectra below it
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
        if not no_image:
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

    def get_equivalent_width(self, 
                            intensity = None, 
                            intensity_error = None, 
                            continuum = None, 
                            continuum_error = None, 
                            return_sigma = True):
        """
        Estimate emission line equivalent width

        Parameters
        ----------
        continuum: `u.Quantity`, optional, must be keyword
            Continuum value or values to use
        continuum_error: `u.Quantity`, optional, must be keyword
            Continuum error value or values to use
        intensity: 'u.Quantity', optional, must be keyword
            Intensity values to use
        """
        # Equivalencies
        ha_wave = 656.3 * u.nm

        optical_ha_equiv = u.doppler_optical(ha_wave)

        # Chanel Width
        dv = (self["VELOCITY"][0][1] - self["VELOCITY"][0][0]) * u.km/u.s
        dlambda = (dv).to(u.AA, equivalencies=optical_ha_equiv) - \
            (0 * u.km/u.s).to(u.AA, equivalencies=optical_ha_equiv)

        if continuum is None:
            if "BKG" in self.keys():
                continuum = self["BKG"] / 22.8 * u.R / dlambda
                if "BKGSD" in self.keys():
                    continuum_error = self["BKGSD"] / 22.8 * u.R / dlambda
                elif continuum_error is None:
                    logging.warning("No Continuum Error Found, using 10% errors instead.")
                    continuum_error = 0.1 * continuum

            elif not hasattr(continuum, "unit"):
                raise TypeError("continuum must be Quantity with units flux/Angstrum")

        if intensity is None:
            ew = self["INTEN"] / continuum
            if return_sigma:
                if intensity_error is None: 
                    ew_error = np.sqrt((self["ERROR"].data / self["INTEN"].data)**2 + \
                                        (continuum_error / continuum)**2) * ew
                else:
                    ew_error = np.sqrt((intensity_error / self["INTEN"])**2 + \
                                        (continuum_error / continuum)**2) * ew
        else:
            ew = intensity / continuum
            if return_sigma:
                if intensity_error is None: 
                    raise TypeError("if intensity is provided, must also provide intensity_error")

                else:
                    ew_error = np.sqrt((intensity_error / intensity)**2 + \
                                        (continuum_error / continuum)**2) * ew

        if return_sigma:
            return ew, ew_error
        else:
            return ew




    


    def get_quadratic_centroid(self, data_column = None, velocity_column  = None,
        vmin = None, vmax = None, 
        smooth = "hanning", window_len = 11, variance_column = None, uncertainty = None):

        """

        Use quadratic centroid method to find centroid and errors
        data_column: `str`, optional, must be keyword
            name of data column to use
        velocity_column: 'str', optional, must be keyword
            name of velocity column to use
        vmin: `number`, `u.Quantity`, optional, must be keyword
            min velocity to consider
        vmax: `number`, `u.Quantity`, optional, must be keyword
            max velocity to consider
        smooth: `str`, optional, must be keyword
            smoothing method to use
            if None, doesn't smooth
        window_len: `number`, optional, must be keyword
            size of smoothing window
        variance_column: `str`, optional, must be keyword
            name of variacne column to use
        uncertainty: 'number', `list-like`, optional, must be keyword
            if provided, uses this instead of variance column provided

        """

        if data_column is None:
            data_column = "DATA"
        if velocity_column is None:
            velocity_column = "VELOCITY"
        if variance_column is None:
            variance_column = "VARIANCE"

        if vmin is not None:
            if not hasattr(vmin, "unit"):
                logging.warning("No unit provided for vmin, assuming km/s")
                vmin *= u.km/u.s
        if vmax is not None:
            if not hasattr(vmax, "unit"):
                logging.warning("No unit provided for vmax, assuming km/s")
                vmax *= u.km/u.s

        def smooth_spectrum(data,
        window_len=11,
        window='hanning'):
            """
            NOTE: From scipy cookbook page

            smooth the data using a window with requested size.
            
            This method is based on the convolution of a scaled window with the signal.
            The signal is prepared by introducing reflected copies of the signal 
            (with the window size) in both ends so that transient parts are minimized
            in the begining and end part of the output signal.
            
            input:
                data_column: the input signal column name
                window_len: the dimension of the smoothing window; should be an odd integer
                window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                    flat window will produce a moving average smoothing.

            output:
                the smoothed signal
                
            example:

            t=linspace(-2,2,0.1)
            x=sin(t)+randn(len(t))*0.1
            y=smooth(x)
            
            see also: 
            
            numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
            scipy.signal.lfilter
         
            TODO: the window parameter could be the window itself if an array instead of a string
            NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
            """



            x = data

            if x.ndim != 1:
                raise ValueError("smooth only accepts 1 dimension arrays.")

            if x.size < window_len:
                raise ValueError("Input vector needs to be bigger than window size.")


            if window_len<3:
                return x


            if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


            s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
            if window == 'flat': #moving average
                w=np.ones(window_len,'d')
            else:
                w=eval('np.'+window+'(window_len)')

            y=np.convolve(w/w.sum(),s,mode='valid')
            return y[int((window_len-1)/2):-int((window_len-1)/2)]

        nan_mask = np.isnan(self[data_column])
        nan_mask |= np.isnan(self[variance_column])
        nan_mask |= np.isnan(self[velocity_column])
        data = np.ma.masked_array(data = self[data_column], mask = nan_mask)
        if smooth is None:
            data = data
        else:
            try:
                data = np.array([smooth_spectrum(data[ell], 
                                                    window_len = window_len, window = smooth) 
                                        for ell in range(len(self))])
            except ValueError:
                logging.warning("Invalid smoothing window - using raw data")
 

        if uncertainty is None:
            uncertainty = np.ma.masked_array(data = self[variance_column], mask = nan_mask)

        def quadratic(data, uncertainty=None, axis=0, x0=0.0, dx=1.0, linewidth=None):
            """
            NOTE: From richteague/bettermoments
            DOI: 10.5281/zenodo.1419754

            Compute the quadratic estimate of the centroid of a line in a data cube.
            The use case that we expect is a data cube with spatiotemporal coordinates
            in all but one dimension. The other dimension (given by the ``axis``
            parameter) will generally be wavelength, frequency, or velocity. This
            function estimates the centroid of the *brightest* line along the ``axis''
            dimension, in each spatiotemporal pixel.
            Following Vakili & Hogg we allow for the option for the data to be smoothed
            prior to the parabolic fitting. The recommended kernel is a Gaussian of
            comparable width to the line. However, for low noise data, this is not
            always necessary.
            Args:
                data (ndarray): The data cube as an array with at least one dimension.
                uncertainty (Optional[ndarray or float]): The uncertainty on the
                    intensities given by ``data``. If this is a scalar, all
                    uncertainties are assumed to be the same. If this is an array, it
                    must have the same shape as ``data'' and give the uncertainty on
                    each intensity. If not provided, the uncertainty on the centroid
                    will not be estimated.
                axis (Optional[int]): The axis along which the centroid should be
                    estimated. By default this will be the zeroth axis.
                x0 (Optional[float]): The wavelength/frequency/velocity/etc. value for
                    the zeroth pixel in the ``axis'' dimension.
                dx (Optional[float]): The pixel scale of the ``axis'' dimension.
            Returns:
                x_max (ndarray): The centroid of the brightest line along the ``axis''
                    dimension in each pixel.
                x_max_sig (ndarray or None): The uncertainty on ``x_max''. If
                    ``uncertainty'' was not provided, this will be ``None''.
                y_max (ndarray): The predicted value of the intensity at maximum.
                y_max_sig (ndarray or None): The uncertainty on ``y_max''. If
                    ``uncertainty'' was not provided, this will be ``None''.
            """
            # Cast the data to a numpy array

            data = np.moveaxis(np.atleast_1d(data), axis, 0)
            shape = data.shape[1:]
            data = np.reshape(data, (len(data), -1))

            # Find the maximum velocity pixel in each spatial pixel
            idx = np.argmax(data, axis=0)

            # Deal with edge effects by keeping track of which pixels are right on the
            # edge of the range
            idx_bottom = idx == 0
            idx_top = idx == len(data) - 1
            idx = np.ma.clip(idx, 1, len(data)-2)

            # Extract the maximum and neighboring pixels
            f_minus = data[(idx-1, range(data.shape[1]))]
            f_max = data[(idx, range(data.shape[1]))]
            f_plus = data[(idx+1, range(data.shape[1]))]

            # Work out the polynomial coefficients
            a0 = 13. * f_max / 12. - (f_plus + f_minus) / 24.
            a1 = 0.5 * (f_plus - f_minus)
            a2 = 0.5 * (f_plus + f_minus - 2*f_max)

            # Compute the maximum of the quadratic
            x_max = idx - 0.5 * a1 / a2
            y_max = a0 - 0.25 * a1**2 / a2

            # Set sensible defaults for the edge cases
            if len(data.shape) > 1:
                x_max[idx_bottom] = 0
                x_max[idx_top] = len(data) - 1
                y_max[idx_bottom] = f_minus[idx_bottom]
                y_max[idx_top] = f_plus[idx_top]
            else:
                if idx_bottom:
                    x_max = 0
                    y_max = f_minus
                elif idx_top:
                    x_max = len(data) - 1
                    y_max = f_plus

            # If no uncertainty was provided, end now
            if uncertainty is None:
                return (
                    np.reshape(x0 + dx * x_max, shape), None,
                    np.reshape(y_max, shape), None,
                    np.reshape(2. * a2, shape), None)

            # Compute the uncertainty
            try:
                uncertainty = float(uncertainty) + np.zeros_like(data)
            except TypeError:
                # An array of errors was provided
                uncertainty = np.moveaxis(np.atleast_1d(uncertainty), axis, 0)
                if uncertainty.shape[0] != data.shape[0] or \
                        shape != uncertainty.shape[1:]:
                    raise ValueError("the data and uncertainty must have the same "
                                     "shape")
                uncertainty = np.reshape(uncertainty, (len(uncertainty), -1))

            df_minus = uncertainty[(idx-1, range(uncertainty.shape[1]))]**2
            df_max = uncertainty[(idx, range(uncertainty.shape[1]))]**2
            df_plus = uncertainty[(idx+1, range(uncertainty.shape[1]))]**2

            x_max_var = 0.0625*(a1**2*(df_minus + df_plus) +
                                a1*a2*(df_minus - df_plus) +
                                a2**2*(4.0*df_max + df_minus + df_plus))/a2**4

            y_max_var = 0.015625*(a1**4*(df_minus + df_plus) +
                                  2.0*a1**3*a2*(df_minus - df_plus) +
                                  4.0*a1**2*a2**2*(df_minus + df_plus) +
                                  64.0*a2**4*df_max)/a2**4

            return (
                np.reshape(x0 + dx * x_max, shape),
                np.reshape(dx * np.sqrt(x_max_var), shape),
                np.reshape(y_max, shape),
                np.reshape(np.sqrt(y_max_var), shape))

        vel = np.ma.masked_array(data = self[velocity_column], mask = nan_mask)
        results = np.array([quadratic(data[ell][~nan_mask[ell,:]], 
                            uncertainty = uncertainty.data[ell,:][~nan_mask[ell,:]], 
                            axis = 0, 
                            x0 = np.min(vel.data[ell,:][~nan_mask[ell,:]]), 
                            dx = np.diff(vel.data[ell,:][~nan_mask[ell,:]])[0]) 
                            for ell in range(len(self))])

        return results
























