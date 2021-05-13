import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from smoke.load.errors.errors import *

plt.style.use("default")


class GeographicalDataset:
    def __init__(self, data):
        """ Instantiation of GeographicalDataset which stores given data into self.data

        :param data: Dataset with coordinates lon, lat, and time, with data recorded at each point
        :type data: xr.Dataset
        """
        # Assert is xr Dataset with correct coordinates
        if (type(data)) != xr.Dataset:
            raise WrongTypeError(
                "data given is of type {} not xr.Dataset".format(type(data))
            )
        if set(data.coords.keys()) != set(["lat", "lon", "time"]):
            raise DimensionError(
                "dimensions of data given were {} not {{'time', 'lat', 'lon'}}".format(
                    set(data.coords.keys())
                )
            )

        # If good then put into data
        self._data = data

        # Create meta data for the object for ease of access
        lat_tup = self._get_shape_size_min_max(self.get_latitudes())
        lon_tup = self._get_shape_size_min_max(self.get_longitudes())
        time_tup = self._get_shape_size_min_max(self.get_times())
        self._meta_data = {
            "lat_shape": lat_tup[0],
            "lat_size": lat_tup[1],
            "lat_min": lat_tup[2],
            "lat_max": lat_tup[3],
            "lon_shape": lon_tup[0],
            "lon_size": lon_tup[1],
            "lon_min": lon_tup[2],
            "lon_max": lon_tup[3],
            "time_shape": time_tup[0],
            "time_size": time_tup[1],
            "time_min": time_tup[2],
            "time_max": time_tup[3],
            "features_size": self.get_features().size,
            "features_names": list(self.get_features()),
        }

    def _get_shape_size_min_max(self, arr):
        return (arr.shape, arr.size, np.min(arr), np.max(arr))

    def get_meta(self):
        """ Returns dictionary of meta data for the dataset

        :return: Meta data of number of things, maximums etc.
        :rtype: dict
        """
        return self._meta_data

    def get_latitudes(self):
        """ Returns array of latitude coordinates across arrays

        :return: latitude coordinates for arrays
        :rtype: np.array
        """
        return self._data.coords["lat"].values

    def get_longitudes(self):
        """ Returns array of longitude coordinates across arrays

        :return: longitude coordinates for arrays
        :rtype: np.array
        """
        return self._data.coords["lon"].values

    def get_times(self):
        """ Returns array of time coordinates across arrays

        :return: time coordinates for arrays
        :rtype: np.array
        """
        return self._data.coords["time"].values

    def get_features(self):
        """ Returns array of features available at each longitude/latitude/time

        :return: array of names of features available
        :rtype: np.array
        """
        return np.array(list(self._data.data_vars.keys()))

    def get_feature_data_array(self, feature):
        """ Returns data array of feature containing values at each available long, lat, and time

        :return: array of feature at each longitude latitude and time
        :rtype: xr.DataArray
        """
        if feature in self.get_features():
            return self._data[feature]
        else:
            raise FeatureNotInDataset(
                "feature {} not in dataset features of {}".format(
                    feature, self.get_features()
                )
            )

    def plot_feature_time_index(self, feature, time_index):
        """ Plot 3D surface and 2D contour of given feature at given index of
        time.

        :param feature: Feature of GeographicalDataset
        :type feature: str
        :param time_index: Index of time to plot
        :type time_index: int
        """
        # Define data values for plotting
        x = self.get_longitudes()
        y = self.get_latitudes()
        z = self.get_feature_data_array(feature).values[time_index]
        time = self.get_times()[time_index]

        # Convert to 2D arrays if lat and lon are 1D
        if x.ndim == 1 and y.ndim == 1:
            x, y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(10, 5), tight_layout=True)
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122)

        # Plot 3D
        ax1.plot_surface(x, y, z, cmap="hot")
        grey = 140 / 255
        ax1.w_xaxis.set_pane_color((grey, grey, grey, 1.0))
        ax1.w_yaxis.set_pane_color((grey, grey, grey, 1.0))
        ax1.w_zaxis.set_pane_color((grey, grey, grey, 1.0))
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax1.grid(True, which="both", color="black")

        # Plot Contour
        cont = ax2.contour(x, y, z, cmap="hot")
        contf = ax2.contourf(x, y, z, cmap="hot")
        ax2.text(
            0.05,
            0.95,
            "Time: {}".format(time),
            ha="left",
            va="top",
            transform=ax2.transAxes,
            bbox={"edgecolor": "none", "facecolor": "white", "alpha": 1},
        )
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")
        cbar = fig.colorbar(contf)
        cbar.add_lines(cont)
        ax2.grid(True)

        plt.show()
