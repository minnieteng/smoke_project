"""
i assume what you'd want to do it just use np load to load your grid
and create logic to find from label_time the specific .npy file you want
and just find file, load it, then return the array
"""
import datetime
import numpy as np
import os

noaa_grid_folder = "C:/temp/2018-test/"
label_time = datetime.datetime.strptime("2018-06-29 23:00:00", '%Y-%m-%d %H:%M:%S')
label_time_str = label_time.strftime('%Y-%m-%d %H:%M:%S')

def noaa(label_time, noaa_grid_folder):
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
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    