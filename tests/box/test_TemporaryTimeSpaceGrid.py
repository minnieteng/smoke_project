import unittest
import numpy as np
from datetime import datetime, timedelta
from smoke.box.FeatureTimeSpaceGrid import TemporaryTimeSpaceGrid
from smoke.clean.cleaners import Box

class testTimeSpaceGrid(unittest.TestCase):

    def setUp(self):
        box = Box(
            57.870760, -133.540154, 46.173395, -129.055971, 1250, 5
        )
        self.grid = TemporaryTimeSpaceGrid(
            box,
            np.arange(datetime(2020, 1, 1, 6),
                      datetime(2020, 1, 2, 6),
                      timedelta(hours=6)
            )
        )

    def testConstructor(self):
        grid = self.grid
        self.assertTrue((grid.get_times() == np.array([datetime(2020, 1, 1, 6),
                                                       datetime(2020, 1, 1, 12),
                                                       datetime(2020, 1, 1, 18),
                                                       datetime(2020, 1, 2, 0)])).all())
        self.assertEqual(grid.get_time_index(datetime(2020, 1, 1, 6)), 0)
        self.assertEqual(grid.get_time_index(datetime(2020, 1, 1, 12)), 1)
        self.assertEqual(grid.get_time_index(datetime(2020, 1, 1, 18)), 2)
        self.assertEqual(grid.get_time_index(datetime(2020, 1, 2, 0)), 3)
        self.assertEqual(grid.get_grid().shape, (4, 250, 250))

    def testPopulateSpaceGrid(self):
        self.grid.populate_space_grid(datetime(2020, 1, 1, 6),
                                      np.array([[0, 0], [249, 249]]),
                                      np.array([12, 14]))
        check_grid = np.empty((4, 250, 250))
        check_grid[:] = np.nan
        check_grid[0][0][0] = 12
        check_grid[0][249][249] = 14
        self.assertTrue((np.nansum(check_grid) > 0) and (np.nansum(self.grid.get_grid()) > 0))
        self.assertTrue(np.logical_or((self.grid.get_grid() == check_grid),
                                      np.logical_and(np.isnan(self.grid.get_grid()),
                                                     np.isnan(check_grid))).all())
        self.grid.populate_space_grid(datetime(2020, 1, 1, 6),
                                      np.array([[0, 249], [249, 0]]),
                                      np.array([13, 15]))
        check_grid[0][0][249] = 13
        check_grid[0][249][0] = 15
        self.assertTrue(np.logical_or((self.grid.get_grid() == check_grid),
                                      np.logical_and(np.isnan(self.grid.get_grid()),
                                                     np.isnan(check_grid))).all())
        self.grid.populate_space_grid(datetime(2020, 1, 1, 6),
                                      np.array([[1, 50], [43, 12]]),
                                      np.array([42, 300]))
        check_grid[0][1][50] = 42
        check_grid[0][43][12] = 300
        self.grid.populate_space_grid(datetime(2020, 1, 1, 12),
                                      np.array([[100, 100], [200, 100]]),
                                      np.array([1234, 4321]))
        check_grid[1][100][100] = 1234
        check_grid[1][200][100] = 4321
        self.assertTrue(np.logical_or((self.grid.get_grid() == check_grid),
                                      np.logical_and(np.isnan(self.grid.get_grid()),
                                                     np.isnan(check_grid))).all())
        self.grid.populate_space_grid(datetime(2020, 1, 1, 6),
                                      np.array([[0, 0]]),
                                      np.array([4001]))
        check_grid[0][0][0] = 4001
        self.assertTrue(np.logical_or((self.grid.get_grid() == check_grid),
                                      np.logical_and(np.isnan(self.grid.get_grid()),
                                                     np.isnan(check_grid))).all())

    def testSetTimeGrid(self):
        self.assertEqual(np.nansum(self.grid.get_grid()), 0)
        self.grid.set_time_grid(datetime(2020, 1, 1, 6), np.ones((250, 250)))
        self.assertEqual(np.nansum(self.grid.get_grid()), 250*250)
        self.grid.set_time_grid(datetime(2020, 1, 1, 12), np.ones((250, 250)))
        self.assertEqual(np.nansum(self.grid.get_grid()), 2*250*250)
        test_grid = np.zeros((250, 250))
        test_grid[20][20] = 47
        test_grid[4][2] = 53
        self.grid.set_time_grid(datetime(2020, 1, 1, 6), test_grid)
        self.assertEqual(self.grid.get_grid()[0][20][20], 47)
        self.assertEqual(self.grid.get_grid()[0][4][2], 53)
        self.assertEqual(np.nansum(self.grid.get_grid()), 250*250+47+53)

    def testSetGridFail(self):
        try:
            self.grid.set_time_grid(datetime(2020, 1, 1, 12), np.array([1,2,3]))
            self.fail('Should raise AssertionError')
        except AssertionError:
            pass


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
