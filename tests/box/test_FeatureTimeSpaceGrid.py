import unittest
import numpy as np
from datetime import datetime
from smoke.box.FeatureTimeSpaceGrid import FeatureTimeSpaceGrid
from smoke.clean.cleaners import Box

class testFeatureTimeSpaceGrid(unittest.TestCase):

    def setUp(self):
        box = Box(
            57.870760, -133.540154, 46.173395, -129.055971, 1250, 5
        )
        self.grid = FeatureTimeSpaceGrid(
            box,
            np.array(['x1', 'x2', 'x3']),
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            6
        )

    def testConstructor(self):
        grid = self.grid
        self.assertEqual(grid.get_feature(0), 'x1')
        self.assertEqual(grid.get_feature(1), 'x2')
        self.assertEqual(grid.get_feature(2), 'x3')
        self.assertTrue((grid.get_features() == np.array(['x1', 'x2', 'x3'])).all())
        self.assertEqual(grid.get_feature_index('x1'), 0)
        self.assertEqual(grid.get_feature_index('x2'), 1)
        self.assertEqual(grid.get_feature_index('x3'), 2)
        self.assertEqual(grid.get_time(0), datetime(2020, 1, 1, 6))
        self.assertEqual(grid.get_time(1), datetime(2020, 1, 1, 12))
        self.assertEqual(grid.get_time(2), datetime(2020, 1, 1, 18))
        self.assertEqual(grid.get_time(3), datetime(2020, 1, 2, 0))
        self.assertTrue((grid.get_times() == np.array([datetime(2020, 1, 1, 6),
                                                       datetime(2020, 1, 1, 12),
                                                       datetime(2020, 1, 1, 18),
                                                       datetime(2020, 1, 2, 0)])).all())
        self.assertEqual(grid.get_time_index(datetime(2020, 1, 1, 6)), 0)
        self.assertEqual(grid.get_time_index(datetime(2020, 1, 1, 12)), 1)
        self.assertEqual(grid.get_time_index(datetime(2020, 1, 1, 18)), 2)
        self.assertEqual(grid.get_time_index(datetime(2020, 1, 2, 0)), 3)
        self.assertEqual(grid.get_grid().shape, (3, 4, 250, 250))

    def testAssignSpaceGrid1DMaskProperly(self):
        assigns = self.grid.assign_space_grid(np.array([30, 40, 55, 70, 80]), np.array([-160, -150, -120, -90, -80]))
        self.assertEqual(assigns.shape, (5, 5, 2))
        self.assertTrue((assigns.mask == np.array([[[True, True],[True, True],[True, True],[True, True],[True, True]],
                                                   [[True, True],[True, True],[True, True],[True, True],[True, True]],
                                                   [[True, True],[True, True],[False, False],[True, True],[True, True]],
                                                   [[True, True],[True, True],[True, True],[True, True],[True, True]],
                                                   [[True, True],[True, True],[True, True],[True, True],[True, True]]])).all())

    def testAssignSpaceGrid2DMaskProperly(self):
        lat2d, lon2d = np.meshgrid(np.array([30, 40, 55, 70, 80]), np.array([-160, -150, -120, -90, -80]))
        assigns = self.grid.assign_space_grid(lat2d, lon2d)
        self.assertEqual(assigns.shape, (5, 5, 2))
        self.assertTrue((assigns.mask == np.array([[[True, True],[True, True],[True, True],[True, True],[True, True]],
                                                   [[True, True],[True, True],[True, True],[True, True],[True, True]],
                                                   [[True, True],[True, True],[False, False],[True, True],[True, True]],
                                                   [[True, True],[True, True],[True, True],[True, True],[True, True]],
                                                   [[True, True],[True, True],[True, True],[True, True],[True, True]]])).all())

    def testPopulateSpaceGrid(self):
        self.grid.populate_space_grid('x1', datetime(2020, 1, 1, 6),
                                      np.array([[0, 0], [249, 249]]),
                                      np.array([12, 14]))
        check_grid = np.empty((3, 4, 250, 250))
        check_grid[:] = np.nan
        check_grid[0][0][0][0] = 12
        check_grid[0][0][249][249] = 14
        self.assertTrue((np.nansum(check_grid) > 0) and (np.nansum(self.grid.get_grid()) > 0))
        self.assertTrue(np.logical_or((self.grid.get_grid() == check_grid),
                                      np.logical_and(np.isnan(self.grid.get_grid()),
                                                     np.isnan(check_grid))).all())
        self.grid.populate_space_grid('x1', datetime(2020, 1, 1, 6),
                                      np.array([[0, 249], [249, 0]]),
                                      np.array([13, 15]))
        check_grid[0][0][0][249] = 13
        check_grid[0][0][249][0] = 15
        self.assertTrue(np.logical_or((self.grid.get_grid() == check_grid),
                                      np.logical_and(np.isnan(self.grid.get_grid()),
                                                     np.isnan(check_grid))).all())
        self.grid.populate_space_grid('x3', datetime(2020, 1, 1, 6),
                                      np.array([[1, 50], [43, 12]]),
                                      np.array([42, 300]))
        check_grid[2][0][1][50] = 42
        check_grid[2][0][43][12] = 300
        self.grid.populate_space_grid('x2', datetime(2020, 1, 1, 12),
                                      np.array([[100, 100], [200, 100]]),
                                      np.array([1234, 4321]))
        check_grid[1][1][100][100] = 1234
        check_grid[1][1][200][100] = 4321
        self.assertTrue(np.logical_or((self.grid.get_grid() == check_grid),
                                      np.logical_and(np.isnan(self.grid.get_grid()),
                                                     np.isnan(check_grid))).all())
        self.grid.populate_space_grid('x1', datetime(2020, 1, 1, 6),
                                      np.array([[0, 0]]),
                                      np.array([4001]))
        check_grid[0][0][0][0] = 4001
        self.assertTrue(np.logical_or((self.grid.get_grid() == check_grid),
                                      np.logical_and(np.isnan(self.grid.get_grid()),
                                                     np.isnan(check_grid))).all())

    def testGetSetGrid(self):
        self.assertEqual(np.nansum(self.grid.get_grid()), 0)
        self.grid.set_grid(np.ones((3, 4, 250, 250)))
        self.assertEqual(np.nansum(self.grid.get_grid()), 3*4*250*250)

    def testSetFeatureGrid(self):
        self.assertEqual(np.nansum(self.grid.get_grid()), 0)
        self.grid.set_feature_grid('x1', np.ones((4, 250, 250)))
        self.assertEqual(np.nansum(self.grid.get_grid()), 4*250*250)
        self.grid.set_feature_grid('x2', np.ones((4, 250, 250)))
        self.assertEqual(np.nansum(self.grid.get_grid()), 2*4*250*250)
        self.grid.set_feature_grid('x2', 2*np.ones((4, 250, 250)))
        self.assertEqual(np.nansum(self.grid.get_grid()), 3*4*250*250)
        test_grid = np.zeros((4, 250, 250))
        test_grid[0][20][20] = 47
        test_grid[3][4][2] = 53
        self.grid.set_feature_grid('x1', test_grid)
        self.assertEqual(self.grid.get_grid()[0][0][20][20], 47)
        self.assertEqual(self.grid.get_grid()[0][3][4][2], 53)
        self.assertEqual(np.nansum(self.grid.get_grid()), 2*4*250*250+47+53)

    def testGetGridNanConverted(self):
        self.assertTrue(np.isnan(self.grid.get_grid()).all())
        self.assertTrue(np.logical_not(np.isnan(self.grid.get_grid_nan_converted(-1))).all())
        self.assertEqual(np.nansum(self.grid.get_grid_nan_converted(-1)), -3*4*250*250)
        self.assertTrue(np.logical_not(np.isnan(self.grid.get_grid_nan_converted(0))).all())
        self.assertEqual(np.nansum(self.grid.get_grid_nan_converted(0)), 0)
        self.grid.populate_space_grid('x1', datetime(2020, 1, 1, 6),
                                      np.array([[0, 0]]),
                                      np.array([4001]))
        self.assertEqual(np.nansum(self.grid.get_grid_nan_converted(0)), 4001)

    def testSetGridFail(self):
        try:
            self.grid.set_grid(np.array([1,2,3]))
            self.fail('Should raise AssertionError')
        except AssertionError:
            pass

    def testSetFeatureGridFail(self):
        try:
            self.grid.set_feature_grid('x2', np.array([1,2,3]))
            self.fail('Should raise AssertionError')
        except AssertionError:
            pass

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
