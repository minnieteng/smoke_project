from __future__ import division

import argparse
import os
import re
import shutil
import sys
sys.path.append('../smoke_tools')
from datetime import datetime, timedelta
from glob import glob

import geopy.distance as gd
import numpy as np
import pandas as pd
from tqdm import tqdm

from smoke.box.Box import Box


# PM25.csv Fields
# DATE_PST (for ordering observations)
# DATE, TIME (for aggregating observations)
# EMS_ID (for uniquely identifying stations, and aggregation)
# RAW_VALUE (PM2.5 value in ug/m^3)


# bc_air_monitoring_stations.csv
# EMS_ID (for uniquely identifying stations, and aggregation)
# LAT, LONG (for identifying relevant cells within the defined grid)

def custom_pm25_avg(x):
    # Threshold PM2.5 measurements to 500
    # Negative values are discarded when averaging
    x = x[x >= 0]
    x = x.clip(upper=500)

    # If there are no observations, we return NaN as the average
    if len(x) < 1:
        return np.nan

    return x.mean()

def custom_pm25_count(x):
    # Threshold PM2.5 measurements to 500
    # Negative values are discarded when averaging
    x = x[x >= 0]
    x = x.clip(upper=500)

    # If there are no observations, we return NaN as the average
    return len(x)


if __name__ == "__main__":
    # Hardcoding these for now
    times = np.arange(datetime(2018, 1, 1, 1), datetime(2020, 1 , 1, 1), timedelta(hours=1)).astype(datetime)
    stations_path = '/projects/smoke_downloads/pm25/bc_air_monitoring_stations.csv'
    pm25_dir = '/projects/smoke_downloads/pm25/using'
    save_dir = '/projects/pm25_labels'
    # pm25_box = Box(56.956768, -131.38922, 48.541751, -129.580869, 1120, 5)
    pm25_box = Box(57.870760, -133.540154, 46.173395, -129.055971, 1250, 5)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Load bc_air_monitoring_stations.csv
    stations_df = pd.read_csv(stations_path,
                            header=0)
    cols_stations = ['EMS_ID', 'LAT', 'LONG']
    stations_df = stations_df[cols_stations]

    #######################
    # 1. Build the PM2.5 master dataframe

    # Get a list of all pm25 datasets
    all_pm25 = glob(os.path.join(pm25_dir, '2*.csv'))

    # Load the dataframes within the PM2.5 directory and concatenate
    list_of_pm25_dfs = []
    for pm25_path in all_pm25:
        pm25_df = pd.read_csv(pm25_path,
                                header=0)
        cols_pm25 = ['DATE_PST', 'EMS_ID', 'RAW_VALUE', 'INSTRUMENT']
        pm25_df = pm25_df[cols_pm25]

        # Remove negative and NaN values
        # Clip to maintain <500
        pm25_df = pm25_df[pm25_df['RAW_VALUE'] >= 0]
        pm25_df = pm25_df.dropna(subset=['RAW_VALUE'])
        pm25_df['RAW_VALUE'] = pm25_df['RAW_VALUE'].clip(upper=500)

        # Remove BAM measurements
        # For every measurement, if there is another with the same ems_id,
        # prioritize PM25_SHARP5030(i), then PM25_R&P_TEOM, and finally BAM1020
        date_ems = pm25_df.groupby(['DATE_PST', 'EMS_ID'], as_index=True).agg({'INSTRUMENT': 'count'})
        duplicates = date_ems[date_ems['INSTRUMENT'] >= 2]
        duplicates = set(duplicates.index.values)

        grouped_pm25 = pm25_df.groupby(['DATE_PST', 'EMS_ID'], as_index=False)
        # grouped_pm25 = pm25_df.groupby(['DATE_PST', 'EMS_ID'])
        def raw_subset_and_avg(group):
            df = {}
            d = group['DATE_PST'].to_numpy()
            assert(len(set(d))) == 1
            ems = group['EMS_ID'].to_numpy()
            assert(len(set(ems))) == 1
            instru = group['INSTRUMENT'].to_numpy()
            raw = group['RAW_VALUE'].to_numpy()

            if (d[0], ems[0]) in duplicates:
                regex_instru = list(map(lambda x: re.match('PM25_SHARP*', x), instru))
                if regex_instru.count(None) == len(regex_instru):
                    regex_instru = list(map(lambda x: re.match('PM25_T640*', x), instru))
                if regex_instru.count(None) == len(regex_instru):
                    regex_instru = list(map(lambda x: re.match('PM25_*', x), instru))
                sharp_idx = next(i for i, item in enumerate(regex_instru) if item is not None)
                df['RAW_VALUE'] = raw[sharp_idx]
            else:
                try:
                    assert(len(raw) == 1)
                except:
                    raw = [np.mean(raw)]
                df['RAW_VALUE'] = raw[0]

            return pd.Series(df)

        # Create and register a new `tqdm` instance with `pandas`
        # (can use tqdm_gui, optional kwargs, etc.)
        tqdm.pandas()

        # Now you can use `progress_apply` instead of `apply`
        pm25_df = grouped_pm25.progress_apply(raw_subset_and_avg)['RAW_VALUE'].reset_index()

        list_of_pm25_dfs.append(pm25_df)

    # Concatenate the list of DataFrames
    pm25_df = pd.concat(list_of_pm25_dfs)

    # Convert DATE_PST to a datetime object
    pm25_df['DATE_PST'] =  pd.to_datetime(pm25_df['DATE_PST'])

    # Save the master PM2.5 df
    pm25_df.to_csv(os.path.join(pm25_dir, 'master_pm25.csv'), index=False)

    ######################
    # 2. Identify the actual time slots that we want, and the EMS stations we want at those time slots

    # Load the master PM2.5 df
    master_pm25_path = os.path.join(pm25_dir, 'master_pm25.csv')
    pm25_df = pd.read_csv(master_pm25_path,
                            header=0)

    pm25_df_ems_avg = pm25_df.groupby(['EMS_ID'], as_index=False).agg({'RAW_VALUE' : [custom_pm25_avg, custom_pm25_count]})
    pm25_df_ems_avg.reset_index(inplace=True)
    pm25_df_ems_avg.columns = [' '.join(col).strip() for col in pm25_df_ems_avg.columns.values]

    # Drop the NaN values
    pm25_df_ems_avg = pm25_df_ems_avg.dropna()

    # Keep track of which EMS_IDs we had an average for in this time block
    present_ems_ids = pm25_df_ems_avg['EMS_ID'].to_numpy()
    ems_counts = pm25_df_ems_avg['RAW_VALUE custom_pm25_count'].to_numpy()

    # Find top N counts
    N = 10
    ind = np.argpartition(ems_counts, -N)[-N:]

    # Determine the EMS_IDs we want to include
    included_ems_ids = set(present_ems_ids[ind])

    #####################################
    # Find the dates and times where we will have observations with those EMS_IDs
    # i.e. we want stations identified above to have measurements in each observation

    def custom_ems_count(x):
        if included_ems_ids.issubset(set(x.to_numpy())):
            return 1
        else:
            return 0

    # We do this by aggregating DATE_PST
    # For each DATE_PST, we increment the number of measurements if all EMS_IDs are present
    pm25_df_times_count = pm25_df.groupby(['DATE_PST'], as_index=False).agg({'EMS_ID' : custom_ems_count})
    times = pm25_df_times_count[pm25_df_times_count['EMS_ID'] == 1]['DATE_PST'].to_numpy()


    #####################
    # 3. Iterate through array of times defined above and save the appropriate PM2.5 arrays
    # for i, t in enumerate(tqdm(times)):
    for i, t in enumerate(tqdm(times)):
        # Subset the pm25 dataframe to those entries containing the relevant times
        pm25_df_relevant = pm25_df[pm25_df['DATE_PST'] == t]

        # Average over EMS_ID to handle repeated measurements problem
        # Group on EMS_ID and average aggregate RAW_VALUE
        pm25_df_relevant_avg = pm25_df_relevant.groupby(['EMS_ID']).agg({'RAW_VALUE' : custom_pm25_avg})
        pm25_df_relevant_avg.reset_index(inplace=True)

        # print(pm25_df_relevant_avg.head(25))

        # Drop the NaN values
        pm25_df_relevant_avg = pm25_df_relevant_avg.dropna()

        # Keep track of which EMS_IDs we had an average for in this time block
        present_ems_ids = pm25_df_relevant_avg['EMS_ID'].to_numpy()
        station_pm25s = pm25_df_relevant_avg['RAW_VALUE'].to_numpy()

        # Initialize the PM2.5 grid, as well as the grid mask
        # pm25_grid = np.zeros((pm25_box.num_cells, pm25_box.num_cells))
        pm25_grid = dict.fromkeys(included_ems_ids)
        # pm25_mask = np.zeros((pm25_box.num_cells, pm25_box.num_cells))

        # For each EMS_ID we have a measurement for, retrieve its LAT and LONG from the stations dataframe
        # and compute which grid index we should set its RAW_VALUE avg to
        for ems_id, pm25 in zip(present_ems_ids, station_pm25s):
            stat_lat = stations_df[stations_df['EMS_ID'] == ems_id]['LAT'].to_numpy()[0]
            stat_lon = stations_df[stations_df['EMS_ID'] == ems_id]['LONG'].to_numpy()[0]

            # if not pm25_box.is_within(stat_lat, stat_lon):
            #     print('All stations should be within box...')
            #     print(ems_id)
            #     print(stat_lat)
            #     print(stat_lon)
            #     sys.exit(1)

            if ems_id not in included_ems_ids:
                continue

            m, n = pm25_box.get_cell_assignment(stat_lat, stat_lon)
            # # if np.isnan(m) or np.isnan(n):
            # #     print(ems_id)
            # #     print(stat_lat)
            # #     print(stat_lon)
            # #     sys.exit(1)
            # # print(m, n)
            # pm25_grid[m][n] = pm25
            pm25_grid[ems_id] = pm25

            # # The corresponding grid index in the grid mask should be set to 1
            # pm25_mask[m][n] = 1

        # Check that we only added keys that were to be included, and that we added all of them
        assert(pm25_grid.keys() == included_ems_ids)
        pm25_grid = np.array(list(pm25_grid.values()))

        # Save the image and mask at this time block
        t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
        year, month, day, hour = str(t.year), str(t.month), str(t.day), str(t.hour)
        save_path = os.path.join(save_dir, year + '_' + month + '_' + day + '_' + hour + '_pm25_labels.npy')
        np.save(save_path, pm25_grid)
        # save_path = os.path.join(save_dir, year + '_' + month + '_' + day + '_' + hour + '_pm25_mask.npy')
        # np.save(save_path, pm25_mask)
