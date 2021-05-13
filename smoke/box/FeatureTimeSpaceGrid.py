import os
import json
import tarfile
import tempfile
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pytz import timezone
from geopy.distance import distance
from scipy.optimize import Bounds, minimize
from collections import Counter

from box.Box import Box
#from smoke.box.Box import Box


class FeatureTimeSpaceGrid:

    def __init__(self, box, features,
                 datetime_start, datetime_stop, time_res_h):
        """ Create an instance of a 4D grid with shape
        (n_features, n_time, n_latitude, n_longitude) object acts as a
        wrapper around the 2D theoretical space grid of the given Box

        :param box: Theoretical space grid to use as last 2 dims of space
        :type box: smoke_tools.box.Box
        :param features: Array of unique features for grid
        :type features: np.array
        :param datetime_start: Time for FeatureTimeSpaceGrid to start exclusive
        :type datetime_start: datetime
        :param datetime_stop: Time for FeatureTimeSpaceGrid to end inclusive
        :type datetime_stop: datetime
        :param time_res_h: Resolution to use in between start and stop for time in hours
        :type time_res_h: int

        """
        # Assign box attributes
        self.box = box

        # Save exact args used to make Box
        box_args = self.box.get_orig_box_args()
        self.orig_box_args = {"nw_lat":box_args[0],
                              "nw_lon":box_args[1],
                              "sw_lat_est":box_args[2],
                              "sw_lon_est":box_args[3],
                              "dist_km":box_args[4],
                              "dist_res_km":box_args[5]}

        # Assign feature and time attributes ensuring UTC
        self.features = features
        self.datetime_start = datetime_start.replace(tzinfo=(timezone('UTC')))
        self.datetime_stop = datetime_stop.replace(tzinfo=(timezone('UTC')))
        self.time_res_h = time_res_h

        # Create times and empty feature time space grid (features, time, lat, lon)
        # use ends of each time bins to make it easier for cleaner selection
        # as just have to choose all times in day
        self.times = np.arange(
            datetime_start+timedelta(hours=time_res_h),
            datetime_stop+timedelta(hours=time_res_h),
            timedelta(hours=time_res_h)
        )
        self.feature_time_space_grid = np.empty((self.features.size,
                                                self.times.size,
                                                self.box.get_num_cells(),
                                                self.box.get_num_cells()
                                                ))
        self.feature_time_space_grid[:] = np.nan

    def get_feature(self, i):
        """ Return feature at given index in features

        """
        return self.features[i]

    def get_features(self):
        """ Return np.array of features axis

        """
        return self.features

    def get_feature_index(self, feature):
        """ Return int index of feature in features

        """
        return np.indices(self.features.shape)[0][self.features == feature][0]

    def get_time(self, i):
        """ Return datetime at given index in times

        """
        return self.times[i]

    def get_times(self):
        """ Return np.array of time axis

        """
        return self.times

    def get_time_index(self, time):
        """ Return int index of time in times

        """
        return np.indices(self.times.shape)[0][self.times == time][0]

    def set_grid(self, grid):
        """ Set grid to whatever grid was given if it correct shape

        :param grid: Grid of similar shape to replace current grid with
        :type grid: np.array
        """
        assert self.get_grid().shape == grid.shape, "Given grid has incorrect shape"
        self.feature_time_space_grid = grid

    def set_feature_grid(self, feature, grid):
        """ Set grid of time, space at feature to whatever grid was given if it
        correct shape

        :param grid: Grid of similar shape to replace current grid at feature with
        :type grid: np.array
        """
        assert self.get_grid().shape[1:] == grid.shape, "Given feature grid has incorrect shape"
        feature_index = self.get_feature_index(feature)
        self.feature_time_space_grid[feature_index] = grid

    def get_grid(self):
        """ Return np.array of current grid of FeatureTimeSpaceGrid shape
        (n_features, n_time, n_latitude, n_longitude)

        """
        return self.feature_time_space_grid

    def get_grid_nan_converted(self, fill_val=-1):
        """ Return np.array of current grid of FeatureTimeSpaceGrid shape
        (n_features, n_time, n_latitude, n_longitude) with all np.nan
        converted into -1 or whatever fill value is given.

        :param fill_val: Value to replace np.nan with, default -1
        :type: float, optional
        """
        grid_copy = self.feature_time_space_grid.copy()
        where_nan = np.isnan(grid_copy)
        grid_copy[where_nan] = fill_val
        return grid_copy

    def assign_space_grid(self, lat, lon, mesh=True):
        """ Assign cells i and j for every data point spatially based on longitude and
        latitude.

        :param lat: Array of latitudes either unmeshed 1D, singular 1D, or 2D
        :type lat: np.array
        :param lon: Array of longitudes either unmeshed 1D, singular 1D, or 2D
        :type lon: np.array
        :param mesh: Whether or not to mesh 1D grids, default True
        :type mesh: bool, optional
        :return: Masked array of jth row indices, ith column indices of data values in
                 shape of data, masked values are ones that fall outside of
                 grid of Box
        :rtype: ma.array, ma.array
        """
        # If longitude and latitude are 1D arrays mesh to make grid matching values
        if mesh and lon.ndim == 1 and lat.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)

        # Save shape of coordinate arrays
        shape = lon.shape

        # Vectorize assignment function and assign cell placement if in grid or
        # leave unplaced w np.nan if not
        assign_fcn_vectorized = np.vectorize(
            self.box.get_cell_assignment_if_in_grid,
            otypes=[float, float]
        )
        row, col = assign_fcn_vectorized(lat.flatten(), lon.flatten())
        row, col = row.reshape(shape), col.reshape(shape)
        cell_indices = np.dstack((row, col))

        # Mask array where is np.nan (no assignment) and convert to int
        cell_indices = ma.masked_where(np.isnan(cell_indices), cell_indices)
        cell_indices = cell_indices.astype(int)

        return cell_indices

    def populate_cell(self, feature_index, time_index, j, i, value):
        """ Populate cell with value at given feature index, time index, jth row index,
        and ith column index.

        :param feature_index: Index on feature axis
        :type feature_index: int
        :param time_index: Index on time axis at feature axis location
        :type time_index: int
        :param j: Row index at time axis location
        :type j: int
        :param i: Column index at row index location
        :type i: int
        :param value: Value to place at location
        :type value: float
        """
        self.feature_time_space_grid[feature_index][time_index][j][i] = value

    def populate_space_grid(self, feature, time, unique_cell_assignments, data_vals):
        """ Given an array of unique grid cell assignments (j, i) and corresponding data_vals
        on a single axis, populates the given grid at location feature and time

        :param feature: Feature in grid to populate a time in
        :type feature: str
        :param time: Time in grid to populate space grid of
        :type time: datetime
        :param unique_cell_assignments: Array of unique pairs of coordinates to place data_vals at (j, i)
        :type unique_cell_assignments: np.array shape=(data_vals.size, 2)
        :param data_vals: Array of data_vals to populate cells with for each corresponding cell
                          in unique_cell_assignments
        :type data_vals: np.array shape=(unique_cell_assignments, )
        """
        feature_index = self.get_feature_index(feature)
        time_index = self.get_time_index(time)
        for coords, val in zip(unique_cell_assignments, data_vals):
            self.populate_cell(feature_index, time_index, coords[0], coords[1], val)

    def diagnostic_plot(self):
        """ Plot 2D plot for all features and times

        """
        for feature in self.get_features():

            feature_index = self.get_feature_index(feature)

            for time in self.get_times():

                time_index = self.get_time_index(time)
                grid2D = self.get_grid()[feature_index][time_index]

                fig, ax = plt.subplots(figsize=(16.2, 16))
                im = ax.imshow(grid2D)
                ax.set_title(f"{feature} {time}")
                ax.set_xlabel("Cols")
                ax.set_ylabel("Rows")
                plt.colorbar(im)

                plt.show()

    def save(self, save_dir, prefix=''):
        """ Save 4D grid array, features array, time array, and
        corresponding Box and other meta data in .npy and .json
        files describing entirely the current FTSG.
        File name will contain grid start time, grid end time, and
        grid time resolution

        :param save_dir: Directory to save tar.gz into
        :type save_dir: str
        :param prefix: Prefix to add before automatically generated name,
                       (e.g. firework_ or bluesky_), default ''
        :type prefix: str
        """
        # Generate unique name from start time sop time and some prefix
        unique_name = (
            prefix +
            self.datetime_start.strftime('strt%Y%m%dT%H%M%S_') +
            self.datetime_stop.strftime('stop%Y%m%dT%H%M%S_') +
            f"res{self.time_res_h}"
        )

        # Save individual .npy and .json file in temp dir pre compress
        temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = temp_dir.name
        np.save(os.path.join(temp_dir_path, 'features.npy'), self.get_features())
        np.save(os.path.join(temp_dir_path, 'times.npy'), self.get_times())
        np.save(os.path.join(temp_dir_path, 'grid.npy'), self.get_grid())
        with open(os.path.join(temp_dir_path, 'box_args.json'), 'w') as f_json:
            json.dump(self.orig_box_args, f_json, indent=2)
        with open(os.path.join(temp_dir_path, 'time_args.json'), 'w') as f_json:
            time_args = {
                "datetime_start":self.datetime_start.strftime('%Y-%m-%dT%H:%M:%S'),
                "datetime_stop":self.datetime_stop.strftime('%Y-%m-%dT%H:%M:%S'),
                "time_res_h":self.time_res_h
            }
            json.dump(time_args, f_json, indent=2)
        with open(os.path.join(temp_dir_path, 'meta.json'), 'w') as f_json:
            meta_data = {
                "unique file name":unique_name,
                "start (inclusive)":self.datetime_start.strftime('%Y-%m-%dT%H:%M:%S'),
                "stop (exclusive)":self.datetime_stop.strftime('%Y-%m-%dT%H:%M:%S'),
                "time resolution (h)":self.time_res_h,
                "features list":list(self.get_features()),
                "times list":list(self.get_times().astype(str)),
                "grid shape":self.get_grid().shape,
                "grid all nan":bool(np.isnan(self.get_grid()).all())
            }
            json.dump(meta_data, f_json, indent=2)

        # Compress .npy and .json into a .tar.gz with unique_name in save_dir
        with tarfile.open(os.path.join(save_dir, unique_name+'.tar.gz'), 'w:gz') as f_tar:
            files_in_temp = [
                os.path.join(temp_dir_path, f_in_temp) for f_in_temp in os.listdir(temp_dir_path)
            ]
            for f_in_temp in files_in_temp:
                f_tar.add(f_in_temp, arcname=os.path.basename(f_in_temp))

        # Explicity close tempdir
        temp_dir.cleanup()


class TemporaryTimeSpaceGrid:

    def __init__(self, box, times):
        """ Create an instance of a 4D grid with shape
        (n_features, n_time, n_latitude, n_longitude) object acts as a
        wrapper around the 2D theoretical space grid of the given Box

        :param box: Theoretical space grid to use as last 2 dims of space
        :type box: smoke.box.Box
        :param times: Array of times to use for time axis of timespace grid
        :type times: numpy.ndarray
        """
        # Assign box attributes
        self.box = box

        # Create time and empty feature time space grid (time, lat, lon)
        self.times = times
        self.time_space_grid = np.empty(
            (
                self.times.size,
                self.box.get_num_cells(),
                self.box.get_num_cells()
            )
        )
        self.time_space_grid[:] = np.nan

    def get_times(self):
        """ Return np.array of time axis

        """
        return self.times

    def get_time_index(self, time):
        """ Return int index of time in times

        """
        return np.indices(self.times.shape)[0][self.times == time][0]

    def get_grid(self):
        """ Return np.array of current grid of TemporaryTimeSpaceGrid shape
        (n_time, n_latitude, n_longitude)

        """
        return self.time_space_grid

    def set_time_grid(self, time, grid):
        """ Set grid space at time to whatever grid was given if it
        correct shape

        :param grid: Grid of similar shape to replace current grid at time with
        :type grid: np.array
        """
        assert self.get_grid().shape[1:] == grid.shape, "Given feature grid has incorrect shape"
        time_index = self.get_time_index(time)
        self.time_space_grid[time_index] = grid

    def populate_cell(self, time_index, j, i, value):
        """ Populate cell with value at given time index, jth row index,
        and ith column index.

        :param time_index: Index on time axis at feature axis location
        :type time_index: int
        :param j: Row index at time axis location
        :type j: int
        :param i: Column index at row index location
        :type i: int
        """
        self.time_space_grid[time_index][j][i] = value

    def populate_space_grid(self, time, unique_cell_assignments, data_vals):
        """ Given an array of unique grid cell assignments (j, i) and corresponding data_vals
        on a single axis, populates the given grid at location time

        :param time: Time in grid to populate space grid of
        :type time: datetime
        :param unique_cell_assignments: Array of unique pairs of coordinates to place data_vals at (j, i)
        :type unique_cell_assignments: np.array shape=(data_vals.size, 2)
        :param data_vals: Array of data_vals to populate cells with for each corresponding cell
                          in unique_cell_assignments
        :type data_vals: np.array shape=(unique_cell_assignments, )
        """
        time_index = self.get_time_index(time)
        for coords, val in zip(unique_cell_assignments, data_vals):
            self.populate_cell(time_index, coords[0], coords[1], val)


def load_FeatureTimeSpaceGrid(file_path):
    """ Load a previously saved FTSG into the same state as it was when it was
    saved

    :param file_path: Path to saved tar.gz of FeatureTimeSpaceGrid
    :type file_path: str
    :returns: FeatureTimeSpaceGrid in same state as one which was saved
    :rtype: FeatureTimeSpaceGrid
    """
    # Load all files in tar.gz into a temp dir
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = temp_dir.name
    f_tar = tarfile.open(file_path, mode='r')
    f_tar.extractall(temp_dir_path)

    # Create FeatureTimeSpaceGrid from old data
    features = np.load(os.path.join(temp_dir_path, "features.npy"))
    with open(os.path.join(temp_dir_path, "box_args.json")) as f_json:
        box_args = json.load(f_json)
    with open(os.path.join(temp_dir_path, "time_args.json")) as f_json:
        time_args = json.load(f_json)
    new_box = Box(box_args["nw_lat"],
                  box_args["nw_lon"],
                  box_args["sw_lat_est"],
                  box_args["sw_lon_est"],
                  box_args["dist_km"],
                  box_args["dist_res_km"])
    new_ftsg = FeatureTimeSpaceGrid(
        new_box,
        features,
        datetime.strptime(time_args["datetime_start"], '%Y-%m-%dT%H:%M:%S'),
        datetime.strptime(time_args["datetime_stop"], '%Y-%m-%dT%H:%M:%S'),
        time_args["time_res_h"]
    )
    new_ftsg.set_grid(np.load(os.path.join(temp_dir_path, "grid.npy")))

    # Explicity close tempdir
    temp_dir.cleanup()

    return new_ftsg
