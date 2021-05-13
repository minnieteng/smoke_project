import os
import re
import numpy as np
import numpy.ma as ma
import xarray as xr
from datetime import datetime
from abc import ABC, abstractmethod

from smoke.load.datasets import GeographicalDataset
from smoke.load.parsers import *
from smoke.clean.toolset import *
from smoke.box.Box import Box
from smoke.box.FeatureTimeSpaceGrid import *


class BCBox(Box):
    def __init__(self, resolution_km=5):
        """ Create a theoretical 2D space grid in Box, specifically around
        BC

        :param resolution_km: Grid cell length and width in km, default 5 km
        :type resolution_km: int, optional
        """
        super().__init__(
            57.870760, -133.540154, 46.173395, -129.055971, 1250, resolution_km
        )


class GenericCleaner(ABC):
    def __init__(self):
        pass

    def create_featuretimespacegrid(
        self,
        file_dir,
        box,
        data_datetime_start,
        data_datetime_finish,
        grid_datetime_start,
        grid_datetime_stop,
        grid_time_res_h,
    ):
        """ Generates FeatureTimeSpaceGrid from all data between data_datetime_start
        and data_datetime_finish placed into a FeatureTimeSpaceGrid

        :param file_dir: Location of files to search through
        :type file_dir: os.path or str
        :param box: Theoretical space grid to use as last 2 dims of space in ftsg
        :type box: smoke_tools.box.Box
        :param data_datetime_start: Start datetime for data range inclusive
        :type data_datetime_start: datetime.datetime
        :param data_datetime_finish: End datetime for data range inclusive
        :type data_datetime_finish: datetime.datetime
        :param grid_datetime_start: Time for FeatureTimeSpaceGrid to start
        :type grid_datetime_start: datetime.datetime
        :param grid_datetime_stop: Time for FeatureTimeSpaceGrid to end
        :type grid_datetime_stop: datetime.datetime
        :param grid_time_res_h: Resolution to use in between start and stop for time in hours
        :type grid_time_res_h: int
        :return: FeatureTimeSpaceGrid representing dataset with given timeframe
        :rtype: FeatureTimeSpaceGrid
        """
        file_paths = self.get_files(file_dir, data_datetime_start, data_datetime_finish)
        grid = self.convert_files_tofeaturetimespacegrid(
            file_paths,
            box,
            grid_datetime_start,
            grid_datetime_stop,
            grid_time_res_h,
        )
        return grid

    def get_files(self, file_dir, data_datetime_start, data_datetime_finish):
        """ Retrieves a list of all files in file_dir in data date range

        :param file_dir: Location of files to search through
        :type file_dir: os.path or str
        :param data_datetime_start: Start datetime for data range inclusive
        :type data_datetime_start: datetime.datetime
        :param data_datetime_finish: End datetime for data range inclusive
        :type data_datetime_finish: datetime.datetime
        :return: List of absolute file paths in date range
        :rtype: list<os.path>
        """
        # Load files
        all_files = os.listdir(file_dir)

        # Filter files that match file_name_regex
        compiled_file_name_regex = re.compile(self.file_name_regex)
        filtered_files = list(
            filter(lambda f: compiled_file_name_regex.match(f) is not None, all_files)
        )

        # Grab dates from each of the matching using file_name_datetime_regex, file_name_datetime_fmt
        compiled_file_name_datetime_regex = re.compile(self.file_name_datetime_regex)
        dates = list(
            map(
                lambda s: datetime.strptime(
                    compiled_file_name_datetime_regex.search(s).group(0),
                    self.file_name_datetime_fmt,
                ),
                filtered_files,
            )
        )

        # Put into corresponding arrays and select file names with dates in given data range
        file_arr = np.array(list(filtered_files))
        date_arr = np.array(list(dates))
        relevant_files = file_arr[
            (data_datetime_start <= date_arr) & (date_arr <= data_datetime_finish)
        ]

        # Join file_dir to create absolute paths to files
        relevant_files = list(map(lambda f: os.path.join(file_dir, f), relevant_files))

        return relevant_files

    @property
    @abstractmethod
    def expected_features_array(self):
        """ Array of features to expect for dataset

        :type: numpy.array
        """

    @property
    @abstractmethod
    def file_name_regex(self):
        """ Regular expression for matching file name

        :type: str
        """
        ...

    @property
    @abstractmethod
    def file_name_datetime_regex(self):
        """ Regular expression for finding str datetime within file name

        :type: str
        """
        ...

    @property
    @abstractmethod
    def file_name_datetime_fmt(self):
        """ datetime.datetime format for parsing str datetime within file name

        :type: str
        """
        ...

    @abstractmethod
    def convert_files_tofeaturetimespacegrid(
        self,
        file_paths,
        box,
        grid_datetime_start,
        grid_datetime_stop,
        grid_time_res_h,
    ):
        """ Converts all files given, into a FeatureTimeSpaceGrid of given parameters, containing
        dataset's data in the given time range.

        :param file_paths: Files on to use for dataset
        :type file_paths: list<str>
        :param box: Theoretical space grid to use as last 2 dims of space in ftsg
        :type box: smoke_tools.box.Box
        :param grid_datetime_start: Time for FeatureTimeSpaceGrid to start
        :type grid_datetime_start: datetime.datetime
        :param grid_datetime_stop: Time for FeatureTimeSpaceGrid to end
        :type grid_datetime_stop: datetime.datetime
        :param grid_time_res_h: Resolution to use in between start and stop for grid's time in hours
        :type grid_time_res_h: int
        :return: FeatureTimeSpaceGrid with given parameters containing data from files in file_paths
        :rtype: FeatureTimeSpaceGrid
        """
        ...


class GeneralConversionCleaner(GenericCleaner):

    @property
    @abstractmethod
    def expected_features_array(self):
        """ Array of features to expect for dataset

        :type: numpy.array
        """

    @property
    @abstractmethod
    def file_name_regex(self):
        """ Regular expression for matching file name

        :type: str
        """
        ...

    @property
    @abstractmethod
    def file_name_datetime_regex(self):
        """ Regular expression for finding str datetime within file name

        :type: str
        """
        ...

    @property
    @abstractmethod
    def file_name_datetime_fmt(self):
        """ datetime.datetime format for parsing str datetime within file name

        :type: str
        """
        ...

    @property
    @abstractmethod
    def parser(self):
        """ parser to use to get GeographicalDataset from files for dataset

        """
        ...

    @property
    @abstractmethod
    def requires_mesh(self):
        """ boolean whether to mesh lat and lon

        :type: bool
        """
        ...

    def assign_space_each_time(self, time_lat_lon_data, ftsg, requires_mesh):
        """ Create grid assignments for each set of space
        coordinates for each time. Returns arrays of data
        and grid assignments for that data, for each time.

        """
        time_data_assigns = []
        # Assign each time's lat/lon to grid
        for _time, lat, lon, data in time_lat_lon_data:

            if requires_mesh and lon.ndim == 1 and lat.ndim == 1:  # Mesh lon, lat if is necessary
                lon, lat = np.meshgrid(lon, lat)

            # Filter out nan's data, lat, and lon to speed up time
            non_nan_indices = np.logical_not(np.isnan(data.flatten()))
            non_nan_data = data.flatten()[non_nan_indices]
            non_nan_lat = lat.flatten()[non_nan_indices]
            non_nan_lon = lon.flatten()[non_nan_indices]

            # Assign lat and lon coords for each time, take care of mesh
            # above so don't need FTSG to
            non_nan_assigns = ftsg.assign_space_grid(non_nan_lat,
                                                     non_nan_lon,
                                                     mesh=False)

            # Flatten data and assigns and append w time to list
            flat_data = non_nan_data.flatten()
            flat_assigns = non_nan_assigns.flatten().reshape(flat_data.size, 2)
            time_data_assigns.append(
                (
                    _time,
                    flat_data,
                    flat_assigns
                )
            )

        return time_data_assigns

    def group_to_unique_times(self, time_data_assigns):
        """ For the list of (time, data_arr, grid_assigns) group all data_arr and
        grid_assigns together of the same time

        """
        # Extract times, an index array corresponding to times, and a sorted array of unique times
        time_arr = np.array([tup[0] for tup in time_data_assigns])
        time_indices = np.indices(time_arr.shape)[0]
        unique_times = np.sort(np.unique(time_arr))

        # For each unique time, group together all data and all grid assignments
        # in corresponding order together, so there is one data array and corresponding
        # grid assignment array for every time
        time_groupeddata_groupedassigns = []
        for time_ in unique_times:
            relevant_indices = time_indices[time_arr == time_]
            grouped_data = []
            grouped_assigns = []
            for i in relevant_indices:
                dump, data_arr, assign_arr = time_data_assigns[i]
                grouped_data.append(data_arr)
                grouped_assigns.append(assign_arr)
            time_groupeddata_groupedassigns.append(
                (
                    time_,
                    grouped_data,
                    grouped_assigns
                )
            )

        return time_groupeddata_groupedassigns

    def crunch_overlap_each_time(self, time_groupeddata_groupedassigns):
        """ For the list of (time, grouped_data_arr, grouped_grid_assigns) crunch any
        data points at each time with overlapping grid assignments

        Note: This implementation is for sparse grids will be very slow if grid is not
              sparse, alternatively use other object with regular grid that is not
              sparse

        """
        time_cruncheddata_crunchedassigns = []
        cruncher = MeanCellCruncher()
        for time_, grouped_data, grouped_assigns in time_groupeddata_groupedassigns:

            crunched_assigns, crunched_data = cruncher.crunch_data(
                ma.vstack(grouped_assigns),  # assignments are pairs (2D) so vertical stack
                ma.hstack(grouped_data)  # data singular 1D so horizontal stack
            )

            time_cruncheddata_crunchedassigns.append(
                (
                    time_,
                    crunched_data,
                    crunched_assigns
                )
            )

        return time_cruncheddata_crunchedassigns

    def crunch_to_ftsg_times(self, box, ftsg_times, time_bin_size_h,
                             time_cruncheddata_crunchedassigns):
        """ Crunch all gridded data at all unique times in
        time_cruncheddata_crunchedassigns to the time bins
        specified by ftsg_times and time_bin_size_h

        """
        # Populate TemporaryTimeSpaceGrid with crunched data and assigns for each time
        orig_times = np.array([tup[0] for tup in time_cruncheddata_crunchedassigns])
        orig_ttsg = TemporaryTimeSpaceGrid(box, orig_times)
        for time_, cruncheddata, crunchedassigns in time_cruncheddata_crunchedassigns:
            orig_ttsg.populate_space_grid(time_, crunchedassigns, cruncheddata)

        # Create a crunched TemporaryTimeSpaceGrid representative of what we want
        # to crunch time_cruncheddata_crunchedassigns to, which will be
        # TTSG of FTSG, with ftsg_times and box
        result_ttsg = TemporaryTimeSpaceGrid(box, ftsg_times)

        # Using TimeCruncher crunch the original TTSG to the result TTSG
        # and return
        cruncher = AvgTimeBinTimeCruncher()

        return cruncher.crunch_to_result_TTSG(time_bin_size_h, orig_ttsg, result_ttsg)

    def convert_files_tofeaturetimespacegrid(
            self,
            file_paths,
            box,
            grid_datetime_start,
            grid_datetime_stop,
            grid_time_res_h):
        """ Converts all files given, into a FeatureTimeSpaceGrid of given parameters, containing
        dataset's data in the given time range.

        :param file_paths: Files containing data to use for populating grid
        :type file_paths: list<str>
        :param box: Theoretical space grid to use as last 2 dims of space in ftsg
        :type box: smoke_tools.box.Box
        :param grid_datetime_start: Time for FeatureTimeSpaceGrid to start
        :type grid_datetime_start: datetime.datetime
        :param grid_datetime_stop: Time for FeatureTimeSpaceGrid to end
        :type grid_datetime_stop: datetime.datetime
        :param grid_time_res_h: Resolution to use in between start and stop for grid's time in hours
        :type grid_time_res_h: int
        :return: FeatureTimeSpaceGrid with given parameters containing data from files in file_paths
        :rtype: FeatureTimeSpaceGrid
        """

        # Grab GeographicalDatasets of each file using self defined parser
        datasets = list(map(lambda f: self.parser.parse_file(f), file_paths))

        # Create FTSG to place data values in (all files have same features so just use first's)
        ftsg = FeatureTimeSpaceGrid(
            box,
            self.expected_features_array,
            grid_datetime_start,
            grid_datetime_stop,
            grid_time_res_h
        )

        # Iterate over every feature getting and populating a time, row, col grid for each
        for feature in self.expected_features_array:

            # Extract each data array of feature in each dataset
            relevant_feature_data_arrays = [d.get_feature_data_array(feature) for d in datasets]

            # From each data array take out a tuple of it's arrays of time, lat, lon, and data measurements
            time_lat_lon_data = []
            for da in relevant_feature_data_arrays:
                for single_time in da:
                    time_lat_lon_data.append(
                        (
                            single_time['time'].values,
                            single_time['lat'].values,
                            single_time['lon'].values,
                            single_time.values
                        )
                    )

            # For each time assign each lat and lon pair of each data point to a point on the grid
            # getting a time, array of data, and array of corresponding grid assignments to data
            time_data_assigns = self.assign_space_each_time(time_lat_lon_data, ftsg, self.requires_mesh)

            # Group the data and grid assignments of each unique time together
            time_groupeddata_groupedassigns = self.group_to_unique_times(time_data_assigns)

            # Crunch the overlapping grid assignments for each time and filter out bad assigns
            time_cruncheddata_crunchedassigns = self.crunch_overlap_each_time(time_groupeddata_groupedassigns)

            # Crunch to the times of the grid, by taking an average across all space grids grouped before
            # each time on the grid's time axis
            ftsg_time_space_grid = self.crunch_to_ftsg_times(
                box, ftsg.get_times(), grid_time_res_h, time_cruncheddata_crunchedassigns
            )

            # Populate feature with resulting time space grid
            ftsg.set_feature_grid(feature, ftsg_time_space_grid.get_grid())

            return ftsg


class ConsistentGridConversionCleaner(GeneralConversionCleaner):

    def assign_space_each_time(self, time_lat_lon_data, ftsg, requires_mesh):
        """ Create grid assignments for each set of space
        coordinates for each time. Returns arrays of data
        and grid assignments for that data, for each time.

        Note: Assumes that grid is same shape and is used for speeding
        up

        """
        # Only assign anything if there are grids to assign
        if not (len(time_lat_lon_data) == 0):
            # Assign lat and lon coords for just first time since all grids are
            # assumed to be the same (saves time by only doing once)
            _time, lat, lon, data = time_lat_lon_data[0]
            assigns = ftsg.assign_space_grid(lat, lon, requires_mesh)
            flat_assigns = assigns.flatten().reshape(data.size, 2)

        time_data_assigns = []
        # Store time, flat data, and corresponding flat common assignments in list
        for _time, dump1, dump2, data in time_lat_lon_data:
            flat_data = data.flatten()
            time_data_assigns.append(
                (
                    _time,
                    flat_data,
                    flat_assigns
                )
            )

        return time_data_assigns

    def are_same_assigns(self, list_of_assignment_arrays):
        """
        Returns true if all assignment arrays are equivalent entailing an exact same grid
        across all in list.

        """
        bool_tracker = True
        comparison_arr = list_of_assignment_arrays[0]
        for test_arr in list_of_assignment_arrays:
            bool_tracker = bool_tracker and np.equal(comparison_arr.data, test_arr.data).all()
            bool_tracker = bool_tracker and np.equal(comparison_arr.mask, test_arr.mask).all()
        return bool_tracker

    def crunch_overlap_each_time(self, time_groupeddata_groupedassigns):
        """ For the list of (time, grouped_data_arr, grouped_grid_assigns) crunch any
        data point at each time with overlapping grid assignments

        Note: This implementation is for regular grids will not work for non consistent
              grid shapes, need to use different version because non-sparse consistent grids
              take extremely long to crunch, when not taking advantage of the consistency
              assumption

        """
        time_cruncheddata_crunchedassigns = []
        cruncher = MeanCellCruncher()
        for time_, grouped_data, grouped_assigns in time_groupeddata_groupedassigns:

            # Since all in grouped_data, and grouped_assigns have exact same grid, all in grouped_assigns
            # will be exactly the same and thus we can just stack data and take a mean along the stacked
            # axis instead of indivdually taking a mean for each cell, this will reduce the number
            # of required crunching assignments by a huge margin speeding up crunching
            assert self.are_same_assigns(grouped_assigns), "Error: inconsistent grid for data"  # Assert all same grid

            same_grid_precrunch_data = np.mean(np.dstack(grouped_data)[0], axis=1)
            same_grid_precrunch_assigns = grouped_assigns[0]

            crunched_assigns, crunched_data = cruncher.crunch_data(
                same_grid_precrunch_assigns,
                same_grid_precrunch_data
            )

            time_cruncheddata_crunchedassigns.append(
                (
                    time_,
                    crunched_data,
                    crunched_assigns
                )
            )

        return time_cruncheddata_crunchedassigns


class FireworkCleaner(ConsistentGridConversionCleaner):

    file_name_regex = r"^\d\d\d\d\d\d\d\d\d\d_AF_HOURLY_FIRESURFACE.geotiff$"
    file_name_datetime_regex = r"^\d\d\d\d\d\d\d\d\d\d"
    file_name_datetime_fmt = "%Y%m%d%H"
    expected_features_array = np.array(['PM25Forecast'])
    parser = FireworkParser()
    requires_mesh = True


class BlueSkyCleaner(ConsistentGridConversionCleaner):

    file_name_regex = (
        #r"^BSC((00)|(06)|(12)|(18))CA((04)|(12))_\d\d\d\d\d\d\d\d\d\d_dispersion.nc$"
        r"^BSC00CA04_\d\d\d\d\d\d\d\d\d\d_dispersion.nc$"
    )
    file_name_datetime_regex = r"\d\d\d\d\d\d\d\d\d\d"
    file_name_datetime_fmt = "%Y%m%d%H"
    expected_features_array = np.array(['PM25Forecast'])
    parser = BlueSkyParser()
    requires_mesh = True


class MODISAODCleaner(GeneralConversionCleaner):

    file_name_regex = (
        r"^M(Y|O)D04_3K.A\d\d\d\d\d\d\d.\d\d\d\d.061.\d\d\d\d\d\d\d\d\d\d\d\d\d.hdf$"
    )
    file_name_datetime_regex = r"\d\d\d\d\d\d\d.\d\d\d\d"
    file_name_datetime_fmt = "%Y%j.%H%M"
    expected_features_array = np.array(
        [
            "Corrected_Optical_Depth_Land_Solution_3_Land_47",
            "Corrected_Optical_Depth_Land_Solution_3_Land_55",
            "Corrected_Optical_Depth_Land_Solution_3_Land_65",
            "Corrected_Optical_Depth_Land_wav2p1",
            "Mass_Concentration_Land"
        ]
    )
    parser = MODISAODParser()
    requires_mesh = False


class MODISFRPCleaner(GeneralConversionCleaner):
    file_name_regex = r"^split_fire_archive_hour_\d\d\d\d\d\d\d\dT\d\d.hdf$"
    file_name_datetime_regex = r"\d\d\d\d\d\d\d\dT\d\d"
    file_name_datetime_fmt = "%Y%m%dT%H"
    expected_features_array = np.array(['FRP'])
    parser = MODISFRPParser()
    requires_mesh = True
