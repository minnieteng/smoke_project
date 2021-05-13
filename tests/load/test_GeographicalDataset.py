import unittest as ut
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from smoke.load.datasets import GeographicalDataset
from smoke.load.errors.errors import *

class testGeographicalDataset(ut.TestCase):
    def setUp(self):
        self.dataArr1 = xr.DataArray(
            np.array(
                [[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[0, 1, 2], [3, 4, 5], [6, 7, 8]]]
            ),
            dims=["time", "lat", "lon"],
            coords={"time": [0, 1], "lat": [0, 1, 2], "lon": [0, 1, 2]}
        )
        self.dataArr2 = xr.DataArray(
            np.array(
                [[[2, 1, 0], [5, 4, 3], [8, 7, 6]], [[2, 1, 0], [5, 4, 3], [8, 7, 6]]]
            ),
            dims=["time", "lat", "lon"],
            coords={"time": [0, 1], "lat": [0, 1, 2], "lon": [0, 1, 2]}
        )
        self.dataSet = xr.Dataset({"da1": self.dataArr1, "da2": self.dataArr2})

    def testConstructor(self):
        test = GeographicalDataset(self.dataSet)
        self.assertTrue(np.all(test.get_features() == np.array(["da1", "da2"])))
        self.assertTrue(np.all(test.get_feature_data_array("da1") == self.dataArr1))
        self.assertTrue(np.all(test.get_feature_data_array("da2") == self.dataArr2))
        self.assertTrue(np.all(test.get_latitudes() == np.array([0, 1, 2])))
        self.assertTrue(np.all(test.get_longitudes() == np.array([0, 1, 2])))
        self.assertTrue(np.all(test.get_times() == np.array([0, 1])))

    def testMetaData(self):
        test = GeographicalDataset(self.dataSet)
        metadata = test.get_meta()
        self.assertTrue(metadata["lat_shape"] == (3,))
        self.assertTrue(metadata["lat_size"] == 3)
        self.assertTrue(metadata["lat_min"] == 0)
        self.assertTrue(metadata["lat_max"] == 2)
        self.assertTrue(metadata["lon_shape"] == (3,))
        self.assertTrue(metadata["lon_size"] == 3)
        self.assertTrue(metadata["lon_min"] == 0)
        self.assertTrue(metadata["lon_max"] == 2)
        self.assertTrue(metadata["time_shape"] == (2,))
        self.assertTrue(metadata["time_size"] == 2)
        self.assertTrue(metadata["time_min"] == 0)
        self.assertTrue(metadata["time_max"] == 1)
        self.assertTrue(metadata["features_size"] == 2)
        self.assertTrue(metadata["features_names"] == ["da1", "da2"])

    def testPlot(self):
        # Just make sure it doesn't raise any errors
        pass  # Commented out so pop ups don't happen, uncomment to test plots
        # test = GeographicalDataset(self.dataSet)
        # test.plot_feature_time_index('da1', 0)
        # test.plot_feature_time_index('da1', 1)
        # test.plot_feature_time_index('da2', 0)

    def testWrongTypeError(self):
        try:
            GeographicalDataset(np.array([1, 2, 3]))
            self.fail("Needs to throw WrongTypeError")
        except WrongTypeError as s:
            self.assertTrue(
                "data given is of type <class 'numpy.ndarray'> not xr.Dataset" == str(s)
            )

    def testDimensionErrorTime(self):
        dataSet = xr.Dataset(
            {
                "a": xr.DataArray(
                    np.array([[[0]]]),
                    dims=["lime", "lat", "lon"],
                    coords={"lime": [0], "lat": [0], "lon": [0]},
                )
            }
        )
        try:
            GeographicalDataset(dataSet)
            self.fail("Needs to throw DimensionError")
        except DimensionError as s:
            self.assertTrue("dimensions of data given were" in str(s))
            self.assertTrue("lime" in str(s))
            self.assertTrue("lat" in str(s))
            self.assertTrue("lon" in str(s))

    def testDimensionErrorLat(self):
        dataSet = xr.Dataset(
            {
                "a": xr.DataArray(
                    np.array([[[0]]]),
                    dims=["time", "bat", "lon"],
                    coords={"time": [0], "bat": [0], "lon": [0]},
                )
            }
        )
        try:
            GeographicalDataset(dataSet)
            self.fail("Needs to throw DimensionError")
        except DimensionError as s:
            self.assertTrue("bat" in str(s))

    def testDimensionErrorLon(self):
        dataSet = xr.Dataset(
            {
                "a": xr.DataArray(
                    np.array([[[0]]]),
                    dims=["time", "lat", "aon"],
                    coords={"time": [0], "lat": [0], "aon": [0]},
                )
            }
        )
        try:
            GeographicalDataset(dataSet)
            self.fail("Needs to throw DimensionError")
        except DimensionError as s:
            self.assertTrue("aon" in str(s))

    def testFeatureNotInDataset(self):
        dataSet = GeographicalDataset(self.dataSet)
        try:
            dataSet.get_feature_data_array("not a name in features")
            self.fail("Needs to throw FeatureNotInDataset")
        except FeatureNotInDataset as s:
            self.assertTrue(
                "feature not a name in features not in dataset features of ['da1' 'da2']"
                == str(s)
            )


if __name__ == "__main__":
    ut.main(argv=["first-arg-is-ignored"], exit=False)
