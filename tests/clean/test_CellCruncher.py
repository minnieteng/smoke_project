import unittest as ut
import numpy as np
import numpy.ma as ma
from smoke.clean.toolset import MeanCellCruncher, SumCellCruncher

class testCellCruncher(ut.TestCase):

    def setUp(self):
        arr12345 = np.array([[1, 2, 3, 4, 5],
                             [1, 2, 3, 4, 5],
                             [1, 2, 3, 4, 5],
                             [1, 2, 3, 4, 5],
                             [1, 2, 3, 4, 5]])
        arr1to25 = np.array([[1, 2, 3, 4, 5],
                             [6, 7, 8, 9, 10],
                             [11, 12, 13, 14, 15],
                             [16, 17, 18, 19, 20],
                             [21, 22, 23, 24, 25]])
        self.data_allsame = arr12345
        self.cell_allsame = ma.array(np.dstack((np.ones((5, 5)),
                                                np.ones((5, 5)))),
                                     mask=np.dstack((np.zeros((5,5)).astype(bool),
                                                     np.zeros((5,5)).astype(bool))),
                                     dtype=int)
        self.result_allsame_mean = (np.array([[1, 1]]), np.array([3]))
        self.result_allsame_sum = (np.array([[1, 1]]), np.array([75]))
        self.data_alldiff = arr12345
        self.cell_alldiff = ma.array(np.dstack((arr1to25,arr1to25)),
                                     mask=np.dstack((np.zeros((5,5)).astype(bool),
                                                     np.zeros((5,5)).astype(bool))),
                                     dtype=int)
        self.result_alldiff_mean_sum = (np.dstack((arr1to25,arr1to25)).flatten().reshape(25, 2),
                                        arr12345.flatten())
        self.data_alldiff_somemasked = arr12345
        self.cell_alldiff_somemasked = ma.array(
            np.dstack((arr1to25,arr1to25)),
            mask=np.array([[[False, False],[False, False],[False, False],[False, False],[False, False]],
                           [[False, False],[True, True],[True, True],[True, True],[True, True]],
                           [[False, False],[False, False],[False, False],[False, False],[False, False]],
                           [[False, False],[True, True],[True, True],[True, True],[True, True]],
                           [[False, False],[False, False],[False, False],[False, False],[False, False]]]),
            dtype=int)
        self.result_alldiff_somemasked_mean_sum = (np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[11,11],
                                                             [12,12], [13,13], [14,14], [15,15], [16,16],
                                                             [21,21], [22,22], [23,23], [24,24], [25,25]]),
                                                   np.array([1,2,3,4,5,1,1,2,3,4,5,1,1,2,3,4,5]),
                                                   )
        self.data_mixed = arr12345
        self.cell_mixed = ma.array(
            np.array([[[5,5],[5,6],[7,7],[8,8],[9,9]],
                      [[2,3],[2,3],[2,4],[2,4],[2,4]],
                      [[1,0],[1,1],[1,2],[1,3],[1,4]],
                      [[0,0],[0,0],[0,0],[0,0],[0,0]],
                      [[0,0],[0,0],[0,1],[0,1],[0,2]]]),
            mask=np.array([[[False,False],[False,False],[True,True],[True,True],[True,True]],
                           [[False,False],[False,False],[False,False],[True,True],[False,False]],
                           [[False,False],[False,False],[False,False],[False,False],[False,False]],
                           [[False,False],[False,False],[False,False],[True,True],[True,True]],
                           [[False,False],[False,False],[False,False],[False,False],[False,False]]]),
            dtype=int)
        self.result_mixed_mean = (np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[1,3],[1,4],[2,3],[2,4],[5,5],[5,6]]),
                                  np.array([1.8,3.5,5,1,2,3,4,5,1.5,4,1,2]))
        self.result_mixed_sum = (np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[1,3],[1,4],[2,3],[2,4],[5,5],[5,6]]),
                                 np.array([9,7,5,1,2,3,4,5,3,8,1,2]))
        self.data_allnan = np.empty((5, 5))
        self.data_allnan[:] = np.nan
        self.results_allnan_allsame = (np.array([[1, 1]]), np.array([np.nan]))
        self.results_allnan_alldiff = (self.cell_alldiff.data.copy().flatten().reshape(25, 2),
                                       self.data_allnan.copy().flatten())

    def testMeanCellCruncher(self):
        cruncher = MeanCellCruncher()
        crunched_allsame = cruncher.crunch_data(self.cell_allsame, self.data_allsame)
        self.assertTrue(np.all(np.sort(crunched_allsame[0], axis=0) == np.sort(self.result_allsame_mean[0], axis=0)))
        self.assertTrue(np.all(np.sort(crunched_allsame[1]) == np.sort(self.result_allsame_mean[1])))
        crunched_alldiff = cruncher.crunch_data(self.cell_alldiff, self.data_alldiff)
        self.assertTrue(np.all(np.sort(crunched_alldiff[0], axis=0) == np.sort(self.result_alldiff_mean_sum[0], axis=0)))
        self.assertTrue(np.all(np.sort(crunched_alldiff[1]) == np.sort(self.result_alldiff_mean_sum[1])))
        crunched_alldiff_somemasked = cruncher.crunch_data(self.cell_alldiff_somemasked, self.data_alldiff_somemasked)
        self.assertTrue(np.all(np.sort(crunched_alldiff_somemasked[0], axis=0) == np.sort(self.result_alldiff_somemasked_mean_sum[0], axis=0)))
        self.assertTrue(np.all(np.sort(crunched_alldiff_somemasked[1]) == np.sort(self.result_alldiff_somemasked_mean_sum[1])))
        crunched_mixed = cruncher.crunch_data(self.cell_mixed, self.data_mixed)
        self.assertTrue(np.all(np.sort(crunched_mixed[0], axis=0) == np.sort(self.result_mixed_mean[0], axis=0)))
        self.assertTrue(np.all(np.sort(crunched_mixed[1]) == np.sort(self.result_mixed_mean[1])))
        crunched_allnan_allsame = cruncher.crunch_data(self.cell_allsame, self.data_allnan)
        self.assertTrue(np.all(np.sort(crunched_allnan_allsame[0], axis=0) == np.sort(self.results_allnan_allsame[0], axis=0)))
        self.assertTrue(np.all(np.isnan(crunched_allnan_allsame[1]) & np.isnan(self.results_allnan_allsame[1])))
        crunched_allnan_alldiff = cruncher.crunch_data(self.cell_alldiff, self.data_allnan)
        self.assertTrue(np.all(np.sort(crunched_allnan_alldiff[0], axis=0) == np.sort(self.results_allnan_alldiff[0], axis=0)))
        self.assertTrue(np.all(np.isnan(crunched_allnan_alldiff[1]) & np.isnan(self.results_allnan_alldiff[1])))

    def testSumCellCruncher(self):
        cruncher = SumCellCruncher()
        crunched_allsame = cruncher.crunch_data(self.cell_allsame, self.data_allsame)
        self.assertTrue(np.all(np.sort(crunched_allsame[0], axis=0) == np.sort(self.result_allsame_sum[0], axis=0)))
        self.assertTrue(np.all(np.sort(crunched_allsame[1]) == np.sort(self.result_allsame_sum[1])))
        crunched_alldiff = cruncher.crunch_data(self.cell_alldiff, self.data_alldiff)
        self.assertTrue(np.all(np.sort(crunched_alldiff[0], axis=0) == np.sort(self.result_alldiff_mean_sum[0], axis=0)))
        self.assertTrue(np.all(np.sort(crunched_alldiff[1]) == np.sort(self.result_alldiff_mean_sum[1])))
        crunched_alldiff_somemasked = cruncher.crunch_data(self.cell_alldiff_somemasked, self.data_alldiff_somemasked)
        self.assertTrue(np.all(np.sort(crunched_alldiff_somemasked[0], axis=0) == np.sort(self.result_alldiff_somemasked_mean_sum[0], axis=0)))
        self.assertTrue(np.all(np.sort(crunched_alldiff_somemasked[1]) == np.sort(self.result_alldiff_somemasked_mean_sum[1])))
        crunched_mixed = cruncher.crunch_data(self.cell_mixed, self.data_mixed)
        self.assertTrue(np.all(np.sort(crunched_mixed[0], axis=0) == np.sort(self.result_mixed_sum[0], axis=0)))
        self.assertTrue(np.all(np.sort(crunched_mixed[1]) == np.sort(self.result_mixed_sum[1])))
        crunched_allnan_allsame = cruncher.crunch_data(self.cell_allsame, self.data_allnan)
        self.assertTrue(np.all(np.sort(crunched_allnan_allsame[0], axis=0) == np.sort(self.results_allnan_allsame[0], axis=0)))
        self.assertTrue(np.all(np.isnan(crunched_allnan_allsame[1]) & np.isnan(self.results_allnan_allsame[1])))
        crunched_allnan_alldiff = cruncher.crunch_data(self.cell_alldiff, self.data_allnan)
        self.assertTrue(np.all(np.sort(crunched_allnan_alldiff[0], axis=0) == np.sort(self.results_allnan_alldiff[0], axis=0)))
        self.assertTrue(np.all(np.isnan(crunched_allnan_alldiff[1]) & np.isnan(self.results_allnan_alldiff[1])))


if __name__ == "__main__":
    ut.main(argv=["first-arg-is-ignored"], exit=False)
