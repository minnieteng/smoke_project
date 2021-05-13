from __future__ import division

import argparse
import re
import os
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
    # Load 2018 and 2019 data, look for how many stations have measurements
    # exceeding some defined minimum number of measurements
    max_measure = 365 * 2 * 24

    # Hardcoding these for now
    times = np.arange(datetime(2018, 1, 1, 1), datetime(2020, 1 , 1, 1), timedelta(hours=1)).astype(datetime)
    stations_path = './bc_air_monitoring_stations.csv'
    pm25_dir = 'test_pm25_dir'
    save_dir = 'test_pm25_save'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Load bc_air_monitoring_stations.csv
    stations_df = pd.read_csv(stations_path,
                            header=0)
    cols_stations = ['EMS_ID', 'LAT', 'LONG']
    stations_df = stations_df[cols_stations]

    # # Get a list of all pm25 datasets
    # all_pm25 = glob(os.path.join(pm25_dir, '*.csv'))

    # # Load the dataframes within the PM2.5 directory and concatenate
    # list_of_pm25_dfs = []
    # for pm25_path in all_pm25:
    #     pm25_df = pd.read_csv(pm25_path,
    #                             header=0)
    #     cols_pm25 = ['DATE_PST', 'EMS_ID', 'RAW_VALUE', 'INSTRUMENT']
    #     pm25_df = pm25_df[cols_pm25]

    #     # Remove negative and NaN values
    #     # Clip to maintain <500
    #     pm25_df = pm25_df[pm25_df['RAW_VALUE'] >= 0]
    #     pm25_df = pm25_df.dropna(subset=['RAW_VALUE'])
    #     pm25_df['RAW_VALUE'] = pm25_df['RAW_VALUE'].clip(upper=500)

    #     # Remove BAM measurements
    #     # For every measurement, if there is another with the same ems_id,
    #     # prioritize PM25_SHARP5030(i), then PM25_R&P_TEOM, and finally BAM1020
    #     date_ems = pm25_df.groupby(['DATE_PST', 'EMS_ID'], as_index=True).agg({'INSTRUMENT': 'count'})
    #     duplicates = date_ems[date_ems['INSTRUMENT'] >= 2]
    #     duplicates = set(duplicates.index.values)

    #     grouped_pm25 = pm25_df.groupby(['DATE_PST', 'EMS_ID'], as_index=False)
    #     # grouped_pm25 = pm25_df.groupby(['DATE_PST', 'EMS_ID'])
    #     def raw_subset_and_avg(group):
    #         df = {}
    #         d = group['DATE_PST'].to_numpy()
    #         assert(len(set(d))) == 1
    #         ems = group['EMS_ID'].to_numpy()
    #         assert(len(set(ems))) == 1
    #         instru = group['INSTRUMENT'].to_numpy()
    #         raw = group['RAW_VALUE'].to_numpy()

    #         if (d[0], ems[0]) in duplicates:
    #             regex_instru = list(map(lambda x: re.match('PM25_SHARP*', x), instru))
    #             if regex_instru.count(None) == len(regex_instru):
    #                 regex_instru = list(map(lambda x: re.match('PM25_T640*', x), instru))
    #             if regex_instru.count(None) == len(regex_instru):
    #                 regex_instru = list(map(lambda x: re.match('PM25_*', x), instru))
    #             sharp_idx = next(i for i, item in enumerate(regex_instru) if item is not None)
    #             df['RAW_VALUE'] = raw[sharp_idx]
    #         else:
    #             try:
    #                 assert(len(raw) == 1)
    #             except:
    #                 raw = [np.mean(raw)]
    #             df['RAW_VALUE'] = raw[0]

    #         return pd.Series(df)

    #     # Create and register a new `tqdm` instance with `pandas`
    #     # (can use tqdm_gui, optional kwargs, etc.)
    #     tqdm.pandas()

    #     # Now you can use `progress_apply` instead of `apply`
    #     pm25_df = grouped_pm25.progress_apply(raw_subset_and_avg).reset_index()

    #     list_of_pm25_dfs.append(pm25_df)

    # # Concatenate the list of DataFrames
    # pm25_df = pd.concat(list_of_pm25_dfs)

    # # Convert DATE_PST to a datetime object
    # pm25_df['DATE_PST'] =  pd.to_datetime(pm25_df['DATE_PST'])

    # # Save the master PM2.5 df
    # pm25_df.to_csv(os.path.join(pm25_dir, 'master_pm25.csv'), index=False)

    ###########
    # Load the master PM2.5 df
    master_pm25_path = os.path.join(pm25_dir, 'master_pm25.csv')
    pm25_df = pd.read_csv(master_pm25_path,
                            header=0)

    pm25_df_relevant_avg = pm25_df.groupby(['EMS_ID'], as_index=False).agg({'RAW_VALUE' : [custom_pm25_avg, custom_pm25_count]})
    pm25_df_relevant_avg.reset_index(inplace=True)
    pm25_df_relevant_avg.columns = [' '.join(col).strip() for col in pm25_df_relevant_avg.columns.values]

    # # print(pm25_df_relevant_avg.head(25))
    # print(pm25_df_relevant_avg.columns)
    # pm25_df_relevant_avg.columns = [' '.join(col).strip() for col in pm25_df_relevant_avg.columns.values]
    # print(pm25_df_relevant_avg.columns)
    # sys.exit(1)

    # Drop the NaN values
    pm25_df_relevant_avg = pm25_df_relevant_avg.dropna()

    # Keep track of which EMS_IDs we had an average for in this time block
    present_ems_ids = pm25_df_relevant_avg['EMS_ID'].to_numpy()
    ems_counts = pm25_df_relevant_avg['RAW_VALUE custom_pm25_count'].to_numpy()

    # print(min(ems_counts))
    # print(max(ems_counts))

    # Find top N counts
    N = 10
    print(len(set(present_ems_ids)))
    ind = np.argpartition(ems_counts, -N)[-N:]
    print(ems_counts[ind])

    # Determine the EMS_IDs we want to include
    ems_ids = set(present_ems_ids[ind])

    #####################################
    # Find the maximal number of observations we have if
    # we want stations identified above to have measurements in each observation

    def custom_ems_count(x):
        if ems_ids.issubset(set(x.to_numpy())):
            return 1
        else:
            return 0

    # We do this by aggregating DATE_PST
    # For each DATE_PST, we increment the number of measurements if all EMS_IDs are present
    pm25_df_relevant_avg = pm25_df.groupby(['DATE_PST'], as_index=False).agg({'EMS_ID' : custom_ems_count})


    # print(pm25_df_relevant_avg.head(25))
    # sys.exit(1)

    num_obs = sum(pm25_df_relevant_avg['EMS_ID'].to_numpy())
    print(num_obs)
