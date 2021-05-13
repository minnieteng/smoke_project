import os
import logging

import unittest
import numpy as np
import xarray as xr

from smoke.split.frp_splitter import partition_frp_to_hour


def testLoadedFile(file_path, lats, lons, times, check_time_lat_lon_val_list):

    bool_track = True
    ds = xr.open_dataset(file_path)
    for j in lats:
        bool_track = bool_track and (j in ds['lat'].values)
    for i in lons:
        bool_track = bool_track and (i in ds['lon'].values)
    for t in times:
        bool_track = bool_track and (t in ds['time'].values)
    for _time, lat, lon, val in check_time_lat_lon_val_list:
        time_i = np.indices((len(times), ))[0][ds['time'].values == _time][0]
        lat_i = np.indices((len(lats), ))[0][ds['lat'].values == lat][0]
        lon_i = np.indices((len(lons), ))[0][ds['lon'].values == lon][0]
        bool_track = bool_track and (ds['FRP'].values[time_i][lat_i][lon_i] == val)
    bool_track = bool_track and (np.sum(np.logical_not(np.isnan(ds['FRP'].values))) == len(check_time_lat_lon_val_list))

    return bool_track


class TestPartitionFRPToHour(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger(__name__)
        f_to_remove = [os.path.join('testfiles', f) for f in os.listdir('testfiles') if 'split' in f]
        for f in f_to_remove:
            os.remove(f)

    def testSplitFiles(self):
        partition_frp_to_hour('testfiles/paired_down_fire_archive.shp',
                              'testfiles',
                               self.logger)
        self.assertTrue(
            testLoadedFile('testfiles/split_fire_archive_hour_20180411T19.hdf',
                           [48.9294, 46.9842, 48.9254, 46.9871, 46.9745],
                           [-119.6593, -119.1413, -119.6667, -119.1570, -119.1452],
                           [np.datetime64('2018-04-11T19:19')],
                           [(np.datetime64('2018-04-11T19:19'), 48.9294, -119.6593, 30.6),
                            (np.datetime64('2018-04-11T19:19'), 46.9842, -119.1413, 27.2),
                            (np.datetime64('2018-04-11T19:19'), 48.9254, -119.6667, 14.5),
                            (np.datetime64('2018-04-11T19:19'), 46.9871, -119.157, 11.1),
                            (np.datetime64('2018-04-11T19:19'), 46.9745, -119.1452, 14.8)]))
        self.assertTrue(
            testLoadedFile('testfiles/split_fire_archive_hour_20180411T21.hdf',
                           [46.2905],
                           [-118.0352],
                           [np.datetime64('2018-04-11T21:03')],
                           [(np.datetime64('2018-04-11T21:03'), 46.2905, -118.0352, 10.3)]))
        self.assertTrue(
            testLoadedFile('testfiles/split_fire_archive_hour_20180413T19.hdf',
                           [53.393],
                           [-117.3498],
                           [np.datetime64('2018-04-13T19:06')],
                           [(np.datetime64('2018-04-13T19:06'), 53.393, -117.3498, 8.0)]))
        self.assertTrue(
            testLoadedFile('testfiles/split_fire_archive_hour_20180414T06.hdf',
                           [52.8425],
                           [-124.2321],
                           [np.datetime64('2018-04-14T06:18')],
                           [(np.datetime64('2018-04-14T06:18'), 52.8425, -124.2321, 11.8)]))
        self.assertTrue(
            testLoadedFile('testfiles/split_fire_archive_hour_20180414T19.hdf',
                           [44.4888],
                           [-108.4439],
                           [np.datetime64('2018-04-14T19:55')],
                           [(np.datetime64('2018-04-14T19:55'), 44.4888, -108.4439, 6.7)]))
        self.assertTrue(
            testLoadedFile('testfiles/split_fire_archive_hour_20180415T18.hdf',
                           [57.007],
                           [-111.4856],
                           [np.datetime64('2018-04-15T18:52')],
                           [(np.datetime64('2018-04-15T18:52'), 57.007, -111.4856, 7.4)]))
        self.assertTrue(
            testLoadedFile('testfiles/split_fire_archive_hour_20180416T18.hdf',
                           [44.6946, 44.433, 44.4312],
                           [-108.819, -108.373, -108.3646],
                           [np.datetime64('2018-04-16T18:00')],
                           [(np.datetime64('2018-04-16T18:00'), 44.6946, -108.819, 29.9),
                            (np.datetime64('2018-04-16T18:00'), 44.433, -108.373, 6.2),
                            (np.datetime64('2018-04-16T18:00'), 44.4312, -108.3646, 9.4)]))
        self.assertTrue(
            testLoadedFile('testfiles/split_fire_archive_hour_20180417T18.hdf',
                           [49.8624, 44.5438],
                           [-112.7635, -120.125],
                           [np.datetime64('2018-04-17T18:42'), np.datetime64('2018-04-17T18:44')],
                           [(np.datetime64('2018-04-17T18:42'), 49.8624, -112.7635, 8.7),
                            (np.datetime64('2018-04-17T18:44'), 44.5438, -120.125, 22.0)]))
        self.assertTrue(
            testLoadedFile('testfiles/split_fire_archive_hour_20180418T05.hdf',
                           [52.0857, 52.8502],
                           [-122.0589, -124.2433],
                           [np.datetime64('2018-04-18T05:53')],
                           [(np.datetime64('2018-04-18T05:53'), 52.0857, -122.0589, 62.5),
                            (np.datetime64('2018-04-18T05:53'), 52.8502, -124.2433, 15.0)]))
        self.assertTrue(
            testLoadedFile('testfiles/split_fire_archive_hour_20180418T17.hdf',
                           [44.3882],
                           [-108.2486],
                           [np.datetime64('2018-04-18T17:48')],
                           [(np.datetime64('2018-04-18T17:48'), 44.3882, -108.2486, 40.4)]))
        self.assertTrue(
            testLoadedFile('testfiles/split_fire_archive_hour_20180418T19.hdf',
                           [47.7122, 47.7052],
                           [-110.7246, -110.7367],
                           [np.datetime64('2018-04-18T19:25')],
                           [(np.datetime64('2018-04-18T19:25'), 47.7122, -110.7246, 365.7),
                            (np.datetime64('2018-04-18T19:25'), 47.7052, -110.7367, 1054.2)]))


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
