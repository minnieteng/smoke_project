import os
import math
import time
import geohash
import geojson
from geojson import MultiLineString
from shapely import geometry
import shapefile
import numpy
import datetime as dt
import pandas as pd
import logging

logger = logging.getLogger(__name__)
source_shape_file_path = "C:/temp/2018/"
threshold = 60*60
cols = ['start', 'end','start_epoch_round','end_epoch_round','start_epoch_round_dt','end_epoch_round_dt']
times = []

for root,dirs,files in os.walk(source_shape_file_path):
    for file in files:
        with open(os.path.join(root,file),"r") as auto:
            if file.endswith(".shp"):
                try:
                    filename = file.replace(".shp","")
                    shape=shapefile.Reader(source_shape_file_path+filename+"/"+file)
                    for r in shape.iterRecords():
                        start_time = dt.datetime.strptime(r[1], '%Y%j %H%M')
                        end_time = dt.datetime.strptime(r[2], '%Y%j %H%M')
                        epoch_s = dt.datetime.timestamp(dt.datetime.strptime(r[1], '%Y%j %H%M'))
                        epoch_e = dt.datetime.timestamp(dt.datetime.strptime(r[2], '%Y%j %H%M'))
                        # sometimes start is later than end time, we'll assume the earlier time is start
                        epoch_end_round = round(max(epoch_s,epoch_e) / threshold) * threshold
                        epoch_start_round = round(min(epoch_s,epoch_e) / threshold) * threshold
                        epoch_end_round_dt = dt.datetime.utcfromtimestamp(3600 * ((max(epoch_s,epoch_e) + 1800) // 3600))
                        epoch_start_round_dt = dt.datetime.utcfromtimestamp(3600 * ((min(epoch_s,epoch_e) + 1800) // 3600))
                        times.append([start_time,end_time,epoch_start_round,epoch_end_round,epoch_start_round_dt,epoch_end_round_dt])
                    break
                except:
                    logger.error('failed to parse file:'+source_shape_file_path+filename+"/")
                continue


df = pd.DataFrame(times, columns=cols)
df.to_csv('noaa_times.csv')
         