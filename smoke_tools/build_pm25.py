from __future__ import division

import argparse
import os
import sys
sys.path.append('../smoke_tools')
from datetime import datetime, timedelta
from glob import glob

import geopy.distance as gd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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


def build_pm25(pt_file_save_path, label_time, time_accumulation_res, path_to_pm25_arrays):
    # Use label_time and find the n times which are relevant in the past, where
    # n is equal to time_accumulation_res

    # Round down to the nearest hour, obtaining the year, month, day and hour
    start_time = label_time.replace(microsecond=0, second=0, minute=0) - timedelta(hours=time_accumulation_res - 1)
    end_time = label_time.replace(microsecond=0, second=0, minute=0) + timedelta(hours=1)
    valid_times = np.arange(start_time, end_time, timedelta(hours=1))

    # Load the appropriate arrays and average
    # To handle the very, very rare case where station recordings vary from hour to hour,
    # as evidenced by changing cell values in the PM2.5 grid mask,
    # we simply average the masks, and check which values are non-integer
    # For those cells which have non-integer mask values, we need to adjust the average PM2.5

    # pm25_box = Box(56.956768, -131.38922, 48.541751, -129.580869, 1120, 5)
    pm25_box = Box(57.870760, -133.540154, 46.173395, -129.055971, 1250, 5)
    pm25_avg_grid = np.zeros((pm25_box.num_cells, pm25_box.num_cells))
    pm25_avg_mask = np.zeros((pm25_box.num_cells, pm25_box.num_cells))
    for i, t in enumerate(tqdm(valid_times)):
        t = pd.to_datetime(t)
        year, month, day, hour = str(t.year), str(t.month), str(t.day), str(t.hour)
        load_path = os.path.join(path_to_pm25_arrays,
                                year + '_' + month + '_' + day + '_' + hour + '_pm25_labels.npy')
        pm25_avg_grid += np.load(load_path)
        load_path = os.path.join(path_to_pm25_arrays,
                                year + '_' + month + '_' + day + '_' + hour + '_pm25_mask.npy')
        pm25_avg_mask += np.load(load_path)

    # Average over the time unit
    pm25_avg_grid /= time_accumulation_res
    pm25_avg_mask /= time_accumulation_res

    # Now retrieve the indices which have non-integer mask values
    # (i.e. not time_accumulation_res number of measurements)
    non_int = np.where(pm25_avg_mask != pm25_avg_mask.round())
    for x, y in enumerate(zip(non_int[0], non_int[1])):
        pm25_avg_grid[x][y] = pm25_avg_grid[x][y] * time_accumulation_res / (pm25_avg_mask[x][y] * time_accumulation_res)

    return pm25_avg_grid, pm25_avg_mask


if __name__ == "__main__":
    # Hardcoded for now
    pt_file_save_path = 'pm25_tensors'
    label_time = datetime(2019, 7, 15, 12)
    time_accumulation_res = 4
    path_to_pm25_arrays = 'test_pm25_save'

    pm25_avg_grid, pm25_avg_mask = build_pm25(pt_file_save_path, label_time, time_accumulation_res, path_to_pm25_arrays)
    np.save('pm25_avg_test_grid.npy', pm25_avg_grid)
    np.save('pm25_avg_test_mask.npy', pm25_avg_mask)

    # Plot heatmaps
    plt.imshow(pm25_avg_grid, cmap='hot', interpolation='nearest')
    plt.savefig('grid.png')
    plt.close()
    plt.imshow(pm25_avg_mask, cmap='hot', interpolation='nearest')
    plt.savefig('mask.png')
    plt.close()
