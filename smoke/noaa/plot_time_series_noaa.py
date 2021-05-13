#plot time series data
from pandas import read_csv
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
from pylab import *
from collections import Counter

plotf = read_csv(r"C:\Users\melxt\Git\smoke\smoke_tools\noaa_times.csv")
#convert to datetime object
plotf['start_epoch_round'] = pd.to_datetime(plotf['start_epoch_round'])
plotf['end_epoch_round'] = pd.to_datetime(plotf['end_epoch_round'])
#get times
plotf['starttime'] = plotf.start_epoch_round
plotf['endtime'] = plotf.end_epoch_round
#get times as Series
starts = plotf.start_epoch_round
ends = plotf.end_epoch_round
#convert times to lists
starts_list = starts.values.tolist()
ends_list = ends.values.tolist()

#count occurences for each hour
time_results = [0] * 24
for start, end in zip(starts_list, ends_list):
    while start <= end:
        start_dt = dt.datetime.utcfromtimestamp(start)
        time_results[start_dt.hour] +=1
        start += 3600

print(time_results)
        
#plotf['time_results'] = plotf.append([time_results])
        
#plotf.to_csv('noaa_times_count.csv')














