from astropy import units as u
import numpy as np 
import matplotlib.pyplot as plt


try:
    from spectral_cube import SpectralCube
except ModuleNotFoundError:
    # Error handling
    pass
try:
    import cartopy.crs as ccrs
except ModuleNotFoundError:
    # Error handling
    pass

from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle

try:
    from regions import Regions
except ModuleNotFoundError:
    #Error handling
    pass






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
        if event.button == 1: # left mouse click
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
                            regions_str = 'galactic; circle({0:.3}, {1:.4}, {2:.4}")'.format(click_coord.l.wrap_at("180d").value, 
                                                                         click_coord.b.value, 
                                                                         self.radius.to(u.arcsec).value)
                            regions = Regions.parse(regions_str, format='ds9')
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
                            subcube = smaller_cube.subcube_from_regions(regions)
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
                