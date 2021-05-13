import os
import sys
import torch
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone

#from smoke.box.Box import Box
from box.Box import Box
from box.FeatureTimeSpaceGrid import load_FeatureTimeSpaceGrid
from amalgamate.errors.errors import NoValidValuesInGrid, IncompletePredictionSet, NoCorrespondingLabel


class Amalgamator:

    def __init__(self, pm25_labels_folder=None, firework_ftsg_folder=None,
                 bluesky_ftsg_folder=None, modisaod_ftsg_folder=None,
                 modisfrp_ftsg_folder=None, noaa_grid_folder=None):
        """ Create instance of amalgamator which will create pytorch tensors
        from the pm2.5 ground truths and prediction datasets located in the
        given file directories

        """
        self.pm25_labels_folder = pm25_labels_folder

        self.firework_ftsg_folder = firework_ftsg_folder
        self.bluesky_ftsg_folder = bluesky_ftsg_folder
        self.modisaod_ftsg_folder = modisaod_ftsg_folder
        self.modisfrp_ftsg_folder = modisfrp_ftsg_folder
        self.noaa_grid_folder = noaa_grid_folder

    #     self.pm25_arr = None
    #     self.pm25_mask = None
    #
    # def pm25(self, label_time, time_accumulation_res):
    #     # Use label_time and find the n times which are relevant in the past, where
    #     # n is equal to time_accumulation_res
    #
    #     # Round down to the nearest hour, obtaining the year, month, day and hour
    #     start_time = label_time.replace(microsecond=0, second=0, minute=0) - timedelta(hours=time_accumulation_res - 1)
    #     end_time = label_time.replace(microsecond=0, second=0, minute=0) + timedelta(hours=1)
    #     valid_times = np.arange(start_time, end_time, timedelta(hours=1))
    #
    #     # Load the appropriate arrays and average
    #     # To handle the very, very rare case where station recordings vary from hour to hour,
    #     # as evidenced by changing cell values in the PM2.5 grid mask,
    #     # we simply average the masks, and check which values are non-integer
    #     # For those cells which have non-integer mask values, we need to adjust the average PM2.5
    #
    #     # pm25_box = Box(56.956768, -131.38922, 48.541751, -129.580869, 1120, 5)
    #     pm25_box = Box(57.870760, -133.540154, 46.173395, -129.055971, 1250, 5)
    #     pm25_avg_grid = np.zeros((pm25_box.num_cells, pm25_box.num_cells))
    #     pm25_avg_mask = np.zeros((pm25_box.num_cells, pm25_box.num_cells))
    #     for i, t in enumerate(tqdm(valid_times)):
    #         t = datetime.strptime(str(t), "%Y-%m-%dT%H:%M:%S.000000")  # Bugfix datetime64 to datetime
    #         year, month, day, hour = str(t.year), str(t.month), str(t.day), str(t.hour)
    #         load_path = os.path.join(self.pm25_data_folder,
    #                                  year + '_' + month + '_' + day + '_' + hour + '_pm25_labels.npy')
    #         pm25_avg_grid += np.load(load_path)
    #         load_path = os.path.join(self.pm25_data_folder,
    #                                  year + '_' + month + '_' + day + '_' + hour + '_pm25_mask.npy')
    #         pm25_avg_mask += np.load(load_path)
    #
    #     # Average over the time unit
    #     pm25_avg_grid /= time_accumulation_res
    #     pm25_avg_mask /= time_accumulation_res
    #
    #     # Now retrieve the indices which have non-integer mask values
    #     # (i.e. not time_accumulation_res number of measurements)
    #     non_int = np.where(pm25_avg_mask != pm25_avg_mask.round())
    #     for x, y in enumerate(zip(non_int[0], non_int[1])):
    #         pm25_avg_grid[x][y] = pm25_avg_grid[x][y] * time_accumulation_res / (pm25_avg_mask[x][y] * time_accumulation_res)
    #
    #     self.pm25_arr = pm25_avg_grid
    #     self.pm25_mask = pm25_avg_mask

    def check_pm25_PST_file_exists(self, label_time):
        """ Convert label_time to equivalent PST and return true if a pm25
        label npy file of that time exists in the pm25 label's folder.
        label_time will originally be in UTC.

        :param label_time: Desired label time in UTC to check if exists as file
        :type label_time: datetime.datetime
        :return: If label npy file exists or not
        :rtype: bool
        """
        # Set label_time as UTC tzinfo and convert to PST for checking
        utc_label_time = label_time.replace(tzinfo=timezone('UTC'))
        pst_label_time = utc_label_time.astimezone(timezone('US/Pacific'))
        file_name = (
            f"{pst_label_time.year}" + '_' +
            f"{pst_label_time.month}" + '_' +
            f"{pst_label_time.day}" + '_' +
            f"{pst_label_time.hour}" + '_pm25_labels.npy'
        )
        abs_file_path = os.path.join(self.pm25_labels_folder, file_name)

        return os.path.isfile(abs_file_path)


    def _load_ftsg_containing_time(self,
                                   _time,
                                   file_directory,
                                   file_prefix='',
                                   ftsg_time_res_h=1):
        """ Creates a file name of saved FTSG which theoretically would contain
        _time given, then attempts to load it, raising FileNotFoundError if
        the file could not be found and NoValidValuesInGrid is gri loaded is
        all nan.

        """
        # Floor round time to nearest lowest 00h, this will be the endtime
        # of a day's file, use floor since dealing with FTSG time and FTSG
        # times are represented by the time at the end of the bin, so 00
        # label time will occur in FTSG file with that 00 as end time due to this
        if _time.hour == 0:
            day_endtime = datetime(_time.year, _time.month, _time.day)  # Leave at 00 H
        else:
            day_endtime = datetime(_time.year, _time.month, _time.day) + timedelta(days=1)  # Round up then add a day to floor
        day_str_endtime = day_endtime.strftime('%Y%m%dT%H%M%S')
        day_str_starttime = (day_endtime - timedelta(days=1)).strftime('%Y%m%dT%H%M%S')
        file_name = (
            f"{file_prefix}strt{day_str_starttime}_stop{day_str_endtime}_res{ftsg_time_res_h}.tar.gz"
        )

        try:
            # Load FeatureTimeSpaceGrid and return
            ftsg = load_FeatureTimeSpaceGrid(
                os.path.join(
                    file_directory,
                    file_name
                )
            )

            # If is an all nan grid, raise exception stating that it is else,
            # return valid grid
            if np.isnan(ftsg.get_grid()).all():
                raise NoValidValuesInGrid(
                    f'FTSG file name containing {_time} which is {file_name} has no valid data.'
                )

            return ftsg

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f'Generated FTSG file name containing {_time} which is {file_name} does not exist.'
            )

    def _get_closest_previous_release_time(self, _time, daily_release_time_h):
        """ Based on the hour that item is released at daily, generates the
        closest previous release time to _time given.

        """
        # If current _time's time is past release time for that day, means
        # data for that day was released already so just use that
        if _time.hour > daily_release_time_h:
            closest_release_time = datetime(
                _time.year,
                _time.month,
                _time.day,
                daily_release_time_h
            )
        # If not use data from last day
        else:
            previous_day_time = _time-timedelta(days=1)
            closest_release_time = datetime(
                previous_day_time.year,
                previous_day_time.month,
                previous_day_time.day,
                daily_release_time_h
            )

        return closest_release_time

    def firework(self, label_time, file_prefix='firework_', ftsg_time_res_h=1):
        # Load firework FTSG file
        ftsg = self._load_ftsg_containing_time(
            label_time,
            self.firework_ftsg_folder,
            file_prefix,
            ftsg_time_res_h
        )

        # Return array of firework forecast at label_time if is not all nan,
        # raise no NoValidValuesInGrid if is all nan
        time_index = ftsg.get_time_index(label_time)
        if not np.isnan(ftsg.get_grid()[:, time_index]).all():
            return ftsg.get_grid_nan_converted()[:, time_index]
        else:
            raise NoValidValuesInGrid(
                f'FTSG file has no valid data for {label_time}.'
            )

    def bluesky(self, label_time, file_prefix='bluesky_', ftsg_time_res_h=1):
        # Load bluesky FTSG file
        ftsg = self._load_ftsg_containing_time(
            label_time,
            self.bluesky_ftsg_folder,
            file_prefix,
            ftsg_time_res_h
        )

        # Return array of bluesky forecast at label_time if is not all nan,
        # raise no NoValidValuesInGrid if is all nan
        time_index = ftsg.get_time_index(label_time)
        if not np.isnan(ftsg.get_grid()[:, time_index]).all():
            return ftsg.get_grid_nan_converted()[:, time_index]
        else:
            raise NoValidValuesInGrid(
                f'FTSG file has no valid data for {label_time}.'
            )

    def modisaod(self, label_time, modisaod_daily_release_time_h=10, ftsg_time_res_h=6):
        # # Get closest last release time for files
        # last_release_time = self._get_closest_previous_release_time(
        #     label_time,
        #     modisaod_daily_release_time_h
        # )
        #
        # # Load modisaod FTSG file containing last release time
        # ftsg = self._load_ftsg_containing_time(
        #     last_release_time,
        #     self.modisaod_ftsg_folder,
        #     'modisaod_',
        #     ftsg_length_h,
        #     ftsg_time_res_h
        # )

        # Get modisaod FTSG containing label time
        ftsg = self._load_ftsg_containing_time(
            label_time,
            self.modisaod_ftsg_folder,
            'modisaod_',
            ftsg_time_res_h
        )

        # Return time array closest to label time
        before_inc_release_time = ftsg.get_times() <= label_time
        closest_time_index = ftsg.get_time_index(
            ftsg.get_times()[before_inc_release_time][-1]
        )
        return ftsg.get_grid_nan_converted()[:, closest_time_index]

    def modisfrp(self, label_time, modisfrp_daily_release_time_h=10, ftsg_time_res_h=1):
        # # Get closest last release time for files
        # last_release_time = self._get_closest_previous_release_time(
        #     label_time,
        #     modisfrp_daily_release_time_h
        # )
        #
        # # Load modisfrp FTSG file containing last release time
        # ftsg = self._load_ftsg_containing_time(
        #     last_release_time,
        #     self.modisfrp_ftsg_folder,
        #     'modisfrp_',
        #     ftsg_length_h,
        #     ftsg_time_res_h
        # )

        # Get modisfrp FTSG containing label time
        ftsg = self._load_ftsg_containing_time(
            label_time,
            self.modisfrp_ftsg_folder,
            'modisfrp_',
            ftsg_time_res_h
        )

        # Return array of modisfrp at label_time if is not all nan,
        # don't raise no NoValidValuesInGrid if is all nan for modisFRP
        # since no value represents no fire at place
        time_index = ftsg.get_time_index(label_time)
        return ftsg.get_grid_nan_converted()[:, time_index]

    def noaa(self, label_time, noaa_grid_folder):
        label_datetime_str = label_time.strftime('%Y%m%d-%H')
        for file in os.listdir(noaa_grid_folder):
            filename = file.replace(".npy","")
            if filename == label_datetime_str:
                try:
                    arr = np.load(noaa_grid_folder+str(file))
                except FileNotFoundError:
                    raise FileNotFoundError(
                    f'{label_datetime_str} at {label_time} does not exist.'
                    )
                return arr

    def make_pytorch_file_name(self, save_directory, label_time):
        """ Create pytorch tensor file name unique to given label_time

        :param save_directory: Directory to save torch tensor in
        :type save_directory: str
        :param label_time: Time we want to aggregate dataset features around
        :type label_time: datetime.datetime
        :return: File name unique to label_time
        :rtype: str
        """
        pt_fname = label_time.strftime("%Y%m%dT%H%M%S_prediction.pt")
        return os.path.join(save_directory, pt_fname)


    def make_pytorch_tensor(self,
                            label_time,
                            dataset_list,
                            save_directory=''):

        ''' Call each of the dataset functions and amalgamate them together,
        and save into a PyTorch tensor. Each of the datasets included in
        dataset_list should match the __name__ property for one of the dataset
        functions. Moreover, "pm25" MUST be included in this list.

        :param label_time: Time we want to aggregate dataset features around
        :type label_time: datetime.datetime
        :param dataset_list: List of datasets to aggregate over (useful if we
                             for example do not want to include NOAA for
                             whatever reason)
        :type dataset_list: list
        :param save_directory: Directory to save torch tensor in, default saves
                               in current directory which is blank
        :type save_directory: str, optional

        '''
        # NOTE: PyTorch tensors are "channel-first", meaning
        # the feature dimension comes first (i.e. tensor shapes are (n_features, grid_h, grid_w))
        # This should be the shape that is returned by each of the individual dataset functions

        # Check if labels exist for given label time, raising error is not
        if not self.check_pm25_PST_file_exists(label_time):
            raise NoCorrespondingLabel(
                f"No corresponding pm2.5 label found for UTC {label_time}"
            )

        # Append grids of all desired features to one list and then stack
        # them to one array, raise IncompletePredictionSet error if any
        # prediction dataset appending raises NoValidValuesInGrid
        master_arr = []
        try:
            for dataset in dataset_list:
                if dataset == 'firework_closest':
                    master_arr.append(self.firework(label_time,
                                                    file_prefix='firework_closest_'))
                elif dataset == 'firework_2ndclosest':
                    master_arr.append(self.firework(label_time,
                                                    file_prefix='firework_2ndclosest_'))
                elif dataset == 'firework_3rdclosest':
                    master_arr.append(self.firework(label_time,
                                                    file_prefix='firework_3rdclosest_'))
                elif dataset == 'firework_4thclosest':
                    master_arr.append(self.firework(label_time,
                                                    file_prefix='firework_4thclosest_'))
                elif dataset == 'bluesky_closest':
                    master_arr.append(self.bluesky(label_time,
                                                   file_prefix='bluesky_closest_'))
                elif dataset == 'bluesky_2ndclosest':
                    master_arr.append(self.bluesky(label_time,
                                                   file_prefix='bluesky_2ndclosest_'))
                elif dataset == 'modisaod':
                    master_arr.append(self.modisaod(label_time))
                elif dataset == 'modisfrp':
                    master_arr.append(self.modisfrp(label_time))
                elif dataset == 'noaa':
                    master_arr.append(self.noaa(label_time))
                else:
                    raise NotImplementedError

        except NoValidValuesInGrid as e:
            raise IncompletePredictionSet(
                f"Incomplete prediction set with dataset: {dataset} raising:\n{str(e)}"
            )

        except FileNotFoundError as e:
            raise IncompletePredictionSet(
                f"Incomplete prediction set with dataset: {dataset} raising:\n{str(e)}"
            )

        master_arr = np.vstack(master_arr)
        master_tensor = torch.from_numpy(master_arr).type(torch.FloatTensor)

        # Create file name from label_time and save_directory
        pt_file_save_path = self.make_pytorch_file_name(save_directory,
                                                        label_time)

        torch.save(master_tensor, pt_file_save_path)
