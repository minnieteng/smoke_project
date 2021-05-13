import os
import json
import tarfile
import tempfile

import unittest
import numpy as np
from datetime import datetime

from smoke.box.Box import Box
from smoke.box.FeatureTimeSpaceGrid import FeatureTimeSpaceGrid, load_FeatureTimeSpaceGrid


def check_meta(file_path, unique_name, datetime_start, datetime_stop, time_res_h,
               features, times, grid_shape, all_nan):
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = temp_dir.name
    f_tar = tarfile.open(file_path, mode='r')
    f_tar.extractall(temp_dir_path)
    with open(os.path.join(temp_dir_path, 'meta.json'), 'r') as f_json:
        bool_track = True
        meta_data = json.load(f_json)
        bool_track = bool_track and (meta_data["unique file name"] == unique_name)
        bool_track = bool_track and (meta_data["start (inclusive)"] == datetime_start.strftime('%Y-%m-%dT%H:%M:%S'))
        bool_track = bool_track and (meta_data["stop (exclusive)"] == datetime_stop.strftime('%Y-%m-%dT%H:%M:%S'))
        bool_track = bool_track and (meta_data["time resolution (h)"] == time_res_h)
        bool_track = bool_track and (meta_data["features list"] == features)
        bool_track = bool_track and (meta_data["times list"] == times)
        bool_track = bool_track and (meta_data["grid shape"] == grid_shape)
        bool_track = bool_track and (meta_data["grid all nan"] == all_nan)
    temp_dir.cleanup()
    return bool_track


class testSaveLoadFTSG(unittest.TestCase):

    def setUp(self):

        if os.path.isfile('testfiles/prefix1test_strt20200630T000000_stop20200701T000000_res1.tar.gz'):
            os.remove('testfiles/prefix1test_strt20200630T000000_stop20200701T000000_res1.tar.gz')

        if os.path.isfile('testfiles/prefix2test_strt20200525T000000_stop20200527T000000_res6.tar.gz'):
            os.remove('testfiles/prefix2test_strt20200525T000000_stop20200527T000000_res6.tar.gz')

        box = Box(
            57.870760, -133.540154, 46.173395, -129.055971, 1250, 5
        )
        self.FTSG_res1 =  FeatureTimeSpaceGrid(
            box,
            np.array(['feat1', 'feat2']),
            datetime(2020, 6, 30),
            datetime(2020, 7, 1),
            1
        )
        self.FTSG_res1.populate_cell(0, 0, 0, 0, 1)
        self.FTSG_d2_res6 =  FeatureTimeSpaceGrid(
            box,
            np.array(['feat1', 'feat2', 'feat3']),
            datetime(2020, 5, 25),
            datetime(2020, 5, 27),
            6
        )

    def testSaveLoad(self):
        self.FTSG_res1.save('testfiles', 'prefix1test_')
        self.FTSG_d2_res6.save('testfiles', 'prefix2test_')
        self.assertTrue(os.path.isfile('testfiles/prefix1test_strt20200630T000000_stop20200701T000000_res1.tar.gz'))
        self.assertTrue(os.path.isfile('testfiles/prefix2test_strt20200525T000000_stop20200527T000000_res6.tar.gz'))
        self.assertTrue(check_meta('testfiles/prefix1test_strt20200630T000000_stop20200701T000000_res1.tar.gz',
                        'prefix1test_strt20200630T000000_stop20200701T000000_res1',
                        datetime(2020, 6, 30), datetime(2020, 7, 1), 1,
                        ['feat1', 'feat2'],
                        ['2020-06-30T01:00:00.000000',
                         '2020-06-30T02:00:00.000000',
                         '2020-06-30T03:00:00.000000',
                         '2020-06-30T04:00:00.000000',
                         '2020-06-30T05:00:00.000000',
                         '2020-06-30T06:00:00.000000',
                         '2020-06-30T07:00:00.000000',
                         '2020-06-30T08:00:00.000000',
                         '2020-06-30T09:00:00.000000',
                         '2020-06-30T10:00:00.000000',
                         '2020-06-30T11:00:00.000000',
                         '2020-06-30T12:00:00.000000',
                         '2020-06-30T13:00:00.000000',
                         '2020-06-30T14:00:00.000000',
                         '2020-06-30T15:00:00.000000',
                         '2020-06-30T16:00:00.000000',
                         '2020-06-30T17:00:00.000000',
                         '2020-06-30T18:00:00.000000',
                         '2020-06-30T19:00:00.000000',
                         '2020-06-30T20:00:00.000000',
                         '2020-06-30T21:00:00.000000',
                         '2020-06-30T22:00:00.000000',
                         '2020-06-30T23:00:00.000000',
                         '2020-07-01T00:00:00.000000'],
                        [2, 24, 250, 250], False))
        self.assertTrue(check_meta('testfiles/prefix2test_strt20200525T000000_stop20200527T000000_res6.tar.gz',
                        'prefix2test_strt20200525T000000_stop20200527T000000_res6',
                        datetime(2020, 5, 25), datetime(2020, 5, 27), 6,
                        ['feat1', 'feat2', 'feat3'],
                        ['2020-05-25T06:00:00.000000',
                         '2020-05-25T12:00:00.000000',
                         '2020-05-25T18:00:00.000000',
                         '2020-05-26T00:00:00.000000',
                         '2020-05-26T06:00:00.000000',
                         '2020-05-26T12:00:00.000000',
                         '2020-05-26T18:00:00.000000',
                         '2020-05-27T00:00:00.000000'],
                        [3, 8, 250, 250], True))
        first_load = load_FeatureTimeSpaceGrid('testfiles/prefix1test_strt20200630T000000_stop20200701T000000_res1.tar.gz')
        self.assertTrue((first_load.get_features() == self.FTSG_res1.get_features()).all())
        self.assertTrue((first_load.get_times() == self.FTSG_res1.get_times()).all())
        self.assertTrue((first_load.get_grid_nan_converted() == self.FTSG_res1.get_grid_nan_converted()).all())
        second_load = load_FeatureTimeSpaceGrid('testfiles/prefix2test_strt20200525T000000_stop20200527T000000_res6.tar.gz')
        self.assertTrue((second_load.get_features() == self.FTSG_d2_res6.get_features()).all())
        self.assertTrue((second_load.get_times() == self.FTSG_d2_res6.get_times()).all())
        self.assertTrue((second_load.get_grid_nan_converted() == self.FTSG_d2_res6.get_grid_nan_converted()).all())

if __name__ == "__main__":
     unittest.main(argv=["first-arg-is-ignored"], exit=False)
