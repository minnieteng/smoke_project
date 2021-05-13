import unittest
import numpy as np

from smoke.box.Box import Box
from smoke.box.FeatureTimeSpaceGrid import TemporaryTimeSpaceGrid
from smoke.clean.toolset import AvgTimeBinTimeCruncher

class testAvgTimeBinTimeCruncher(unittest.TestCase):

    def setUp(self):

        self.cruncher = AvgTimeBinTimeCruncher()

        box = Box(
            57.870760, -133.540154, 46.173395, -129.055971, 1250, 5
        )

        self.target_ttsg = TemporaryTimeSpaceGrid(
            box,
            np.arange(np.datetime64('2020-06-30T06'),
                      np.datetime64('2020-07-01T06'),
                      np.timedelta64(6, 'h')))

        # Create data
        no_times = np.array([])
        in_first_bin = np.array([np.datetime64('2020-06-30T05')])
        one_all_bins = np.array([np.datetime64('2020-06-30T05'),
                                 np.datetime64('2020-06-30T11'),
                                 np.datetime64('2020-06-30T17'),
                                 np.datetime64('2020-06-30T23')])
        two_odd_bins = np.array([np.datetime64('2020-06-30T05'),
                                 np.datetime64('2020-06-30T11'),
                                 np.datetime64('2020-06-30T11'),
                                 np.datetime64('2020-06-30T17'),
                                 np.datetime64('2020-06-30T22'),
                                 np.datetime64('2020-06-30T23')])

        empty_space_grid = np.empty((box.get_num_cells(), box.get_num_cells()))
        empty_space_grid[:] = np.nan

        self.initial_ttsg_none = TemporaryTimeSpaceGrid(
            box,
            no_times)
        self.initial_ttsg_none_result = np.vstack((empty_space_grid[None, :],
                                                   empty_space_grid[None, :],
                                                   empty_space_grid[None, :],
                                                   empty_space_grid[None, :]))

        self.initial_ttsg_one = TemporaryTimeSpaceGrid(
            box,
            in_first_bin
        )
        self.initial_ttsg_one.populate_cell(0, 0, 0, 40)
        self.initial_ttsg_one.populate_cell(0, 50, 40, 35)
        one_space_grid = empty_space_grid.copy()
        one_space_grid[0][0] = 40.
        one_space_grid[50][40] = 35.
        self.initial_ttsg_one_result = np.vstack((one_space_grid[None, :],
                                                  empty_space_grid[None, :],
                                                  empty_space_grid[None, :],
                                                  empty_space_grid[None, :]))

        self.initial_ttsg_all = TemporaryTimeSpaceGrid(
           box,
           one_all_bins
        )
        self.initial_ttsg_all.populate_cell(0, 3, 4, 40)
        self.initial_ttsg_all.populate_cell(0, 47, 36, 25)
        self.initial_ttsg_all.populate_cell(1, 12, 98, 17)
        self.initial_ttsg_all.populate_cell(2, 0, 0, 7000)
        self.initial_ttsg_all.populate_cell(2, 47, 36, 432)
        self.initial_ttsg_all.populate_cell(3, 3, 4, 0)
        self.initial_ttsg_all.populate_cell(3, 47, 36, 32)
        one_all1 = empty_space_grid.copy()
        one_all1[3][4] = 40.
        one_all1[47][36] = 25.
        one_all2 = empty_space_grid.copy()
        one_all2[12][98] = 17.
        one_all3 = empty_space_grid.copy()
        one_all3[0][0] = 7000.
        one_all3[47][36] = 432.
        one_all4 = empty_space_grid.copy()
        one_all4[3][4] = 0.
        one_all4[47][36] = 32.
        self.initial_ttsg_all_result = np.vstack((one_all1[None, :],
                                                  one_all2[None, :],
                                                  one_all3[None, :],
                                                  one_all4[None, :]))

        self.initial_ttsg_dbl = TemporaryTimeSpaceGrid(
           box,
           two_odd_bins
        )
        self.initial_ttsg_dbl.populate_cell(0, 3, 4, 40)
        self.initial_ttsg_dbl.populate_cell(0, 47, 36, 25)
        self.initial_ttsg_dbl.populate_cell(1, 12, 98, 17)
        self.initial_ttsg_dbl.populate_cell(1, 0, 0, 232)
        self.initial_ttsg_dbl.populate_cell(2, 12, 98, 53)
        self.initial_ttsg_dbl.populate_cell(2, 0, 0, 232)
        self.initial_ttsg_dbl.populate_cell(3, 0, 0, 7000)
        self.initial_ttsg_dbl.populate_cell(3, 47, 36, 432)
        self.initial_ttsg_dbl.populate_cell(4, 3, 4, 0)
        self.initial_ttsg_dbl.populate_cell(4, 47, 36, 32)
        self.initial_ttsg_dbl.populate_cell(5, 3, 4, 100)
        self.initial_ttsg_dbl.populate_cell(5, 47, 36, 1000)
        self.initial_ttsg_dbl.populate_cell(5, 100, 100, 23)
        one_double1 = empty_space_grid.copy()
        one_double1[3][4] = 40.
        one_double1[47][36] = 25.
        one_double2 = empty_space_grid.copy()
        one_double2[0][0] = 232
        one_double2[12][98] = (17+53)/2
        one_double3 = empty_space_grid.copy()
        one_double3[0][0] = 7000.
        one_double3[47][36] = 432.
        one_double4 = empty_space_grid.copy()
        one_double4[3][4] = 50
        one_double4[47][36] = 516
        one_double4[100][100] = 23
        self.initial_ttsg_dbl_result = np.vstack((one_double1[None, :],
                                                  one_double2[None, :],
                                                  one_double3[None, :],
                                                  one_double4[None, :]))


    def testNoTimes(self):
        result_ttsg = self.cruncher.crunch_to_result_TTSG(
            6,
            self.initial_ttsg_none,
            self.target_ttsg)
        self.assertTrue(((result_ttsg.get_grid() == self.initial_ttsg_none_result) |
                         (np.isnan(result_ttsg.get_grid()) & np.isnan(self.initial_ttsg_none_result))).all())

    def testOneTimeFirstBin(self):
        result_ttsg = self.cruncher.crunch_to_result_TTSG(
            6,
            self.initial_ttsg_one,
            self.target_ttsg)
        self.assertTrue(((result_ttsg.get_grid() == self.initial_ttsg_one_result) |
                         (np.isnan(result_ttsg.get_grid()) & np.isnan(self.initial_ttsg_one_result))).all())

    def testOneEachTimeBin(self):
        result_ttsg = self.cruncher.crunch_to_result_TTSG(
            6,
            self.initial_ttsg_all,
            self.target_ttsg)
        self.assertTrue(((result_ttsg.get_grid() == self.initial_ttsg_all_result) |
                         (np.isnan(result_ttsg.get_grid()) & np.isnan(self.initial_ttsg_all_result))).all())

    def testDoubleTwoBins(self):
        result_ttsg = self.cruncher.crunch_to_result_TTSG(
            6,
            self.initial_ttsg_dbl,
            self.target_ttsg)
        self.assertTrue(((result_ttsg.get_grid() == self.initial_ttsg_dbl_result) |
                         (np.isnan(result_ttsg.get_grid()) & np.isnan(self.initial_ttsg_dbl_result))).all())


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
