from __future__ import division

import argparse
import os
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


if __name__ == "__main__":
    # Hardcoding these for now
    times = np.arange(datetime(2018, 1, 1, 1), datetime(2020, 1 , 1, 1), timedelta(hours=1)).astype(datetime)
    stations_path = './bc_air_monitoring_stations.csv'
    pm25_dir = 'test_pm25_dir'
    save_dir = 'test_pm25_save'
    pm25_box = Box(56.956768, -131.38922, 48.541751, -129.580869, 1120, 5)

    os.makedirs(save_dir, exist_ok=True)

    # Load bc_air_monitoring_stations.csv
    stations_df = pd.read_csv(stations_path,
                            header=0)
    cols_stations = ['EMS_ID', 'LAT', 'LONG']
    stations_df = stations_df[cols_stations]

    # Get a list of all pm25 datasets
    all_pm25 = glob(os.path.join(pm25_dir, '*.csv'))

    # Load the dataframes within the PM2.5 directory and concatenate
    list_of_pm25_dfs = []
    for pm25_path in all_pm25:
        pm25_df = pd.read_csv(pm25_path,
                                header=0)
        cols_pm25 = ['DATE_PST', 'DATE', 'TIME', 'EMS_ID', 'RAW_VALUE']
        pm25_df = pm25_df[cols_pm25]

        # Remove negative and NaN values
        # Clip to maintain <500
        pm25_df = pm25_df[pm25_df['RAW_VALUE'] >= 0]
        pm25_df = pm25_df.dropna(subset=['RAW_VALUE'])
        pm25_df['RAW_VALUE'] = pm25_df['RAW_VALUE'].clip(upper=500)

        list_of_pm25_dfs.append(pm25_df)

    # Concatenate the list of DataFrames
    pm25_df = pd.concat(list_of_pm25_dfs)

    # Convert DATE_PST to a datetime object
    pm25_df['DATE_PST'] =  pd.to_datetime(pm25_df['DATE_PST'])

    # Iterate through array of times defined above
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

        # TODO
        # We do not save arrays which have less than 10% measurements available throughout the year

        # Initialize the PM2.5 grid, as well as the grid mask
        pm25_grid = np.zeros((pm25_box.num_cells, pm25_box.num_cells))
        pm25_mask = np.zeros((pm25_box.num_cells, pm25_box.num_cells))

        # For each EMS_ID we have a measurement for, retrieve its LAT and LONG from the stations dataframe
        # and compute which grid index we should set its RAW_VALUE avg to
        for ems_id, pm25 in zip(present_ems_ids, station_pm25s):
            stat_lat = stations_df[stations_df['EMS_ID'] == ems_id]['LAT'].to_numpy()[0]
            stat_lon = stations_df[stations_df['EMS_ID'] == ems_id]['LONG'].to_numpy()[0]

            if not pm25_box.is_within(stat_lat, stat_lon):
                print('All stations should be within box...')
                sys.exit(1)

            m, n = pm25_box.get_cell_assignment(stat_lat, stat_lon)
            pm25_grid[m][n] = pm25
            # The corresponding grid index in the grid mask should be set to 1
            pm25_mask[m][n] = 1

        # Save the image and mask at this time block
        year, month, day, hour = str(t.year), str(t.month), str(t.day), str(t.hour)
        save_path = os.path.join(save_dir, year + '_' + month + '_' + day + '_' + hour + '_pm25_labels.npy')
        np.save(save_path, pm25_grid)
        save_path = os.path.join(save_dir, year + '_' + month + '_' + day + '_' + hour + '_pm25_mask.npy')
        np.save(save_path, pm25_mask)
