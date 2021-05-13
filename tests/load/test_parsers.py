import unittest as ut
import numpy as np
import xarray as xr

from smoke.load.parsers import *


def _array_equal_nan(arr1, arr2):
    return np.all((arr1 == arr2) | (np.isnan(arr1) & np.isnan(arr2)))


class TestParsers(ut.TestCase):

    def setUp(self):
        self.firework_test_file = "testfiles/2020032412_AF_HOURLY_FIRESURFACE.geotiff"
        self.bluesky_test_file = "testfiles/bluesky_dispersion.nc"
        self.old_modisaod_test_file = (
            "testfiles/reduced_MOD04_3K_A2019164_1800_061_2019168190900.hdf"
        )
        self.modisaod_test_file = (
            "testfiles/MOD04_3K_A2019164_1800_061_2019168190900.hdf"
        )
        self.split_modisfrp_test_file = (
            "testfiles/split_fire_archive_hour_20180411T19.hdf"
        )

    def testFireworkParser(self):
        manual_test_data = xr.open_rasterio(self.firework_test_file)
        firework_parser = FireworkParser()
        firework_dataset = firework_parser.parse_file(self.firework_test_file)
        self.assertTrue(
            np.all(firework_dataset.get_longitudes() == manual_test_data["x"].values)
        )
        self.assertTrue(
            np.all(firework_dataset.get_latitudes() == manual_test_data["y"].values)
        )
        self.assertTrue(
            firework_dataset.get_times().size == manual_test_data["band"].values.size
        )
        self.assertTrue(firework_dataset.get_features() == np.array(["PM25Forecast"]))
        self.assertTrue(
            np.all(
                firework_dataset.get_feature_data_array("PM25Forecast").values
                == manual_test_data.values
            )
        )

    def testBlueskyParser(self):
        manual_test_data = xr.open_dataset(self.bluesky_test_file)
        bluesky_parser = BlueSkyParser()
        bluesky_dataset = bluesky_parser.parse_file(self.bluesky_test_file)
        self.assertTrue(
            manual_test_data["ROW"].size == bluesky_dataset.get_latitudes().size
        )
        self.assertTrue(
            manual_test_data["COL"].size == bluesky_dataset.get_longitudes().size
        )
        self.assertTrue(
            manual_test_data["TSTEP"].size == bluesky_dataset.get_times().size
        )
        self.assertTrue(bluesky_dataset.get_features() == np.array(["PM25Forecast"]))
        self.assertTrue(
            np.all(
                bluesky_dataset.get_feature_data_array("PM25Forecast")
                == manual_test_data["PM25"].squeeze().values
            )
        )

    def testMODISAODParser(self):
        manual_test_data = xr.open_dataset(self.old_modisaod_test_file)
        modisaod_parser = MODISAODParser()
        modisaod_dataset = modisaod_parser.parse_file(self.modisaod_test_file)
        self.assertTrue(
            np.all(modisaod_dataset.get_latitudes() == manual_test_data["Latitude"])
        )
        self.assertTrue(
            np.all(modisaod_dataset.get_longitudes() == manual_test_data["Longitude"])
        )
        test_time = (
            manual_test_data.attrs[
                "CoreMetadata.INVENTORYMETADATA.RANGEDATETIME.RANGEBEGINNINGDATE.VALUE"
            ] + 'T' +
            manual_test_data.attrs[
                "CoreMetadata.INVENTORYMETADATA.RANGEDATETIME.RANGEBEGINNINGTIME.VALUE"
            ]
        )
        self.assertTrue(
            list(map(lambda x: str(x)[:19], modisaod_dataset.get_times()))
            == [test_time[:19]]
        )
        self.assertTrue(
            set(modisaod_dataset.get_features())
            == {
                "Corrected_Optical_Depth_Land_Solution_3_Land_55",
                "Corrected_Optical_Depth_Land_wav2p1",
                "Corrected_Optical_Depth_Land_Solution_3_Land_47",
                "Mass_Concentration_Land",
                "Corrected_Optical_Depth_Land_Solution_3_Land_65",
            }
        )
        self.assertTrue(
            _array_equal_nan(
                modisaod_dataset.get_feature_data_array(
                    "Corrected_Optical_Depth_Land_Solution_3_Land_47"
                )
                .squeeze()
                .values,
                manual_test_data["Corrected_Optical_Depth_Land"]
                .isel({"Solution_3_Land": 0})
                .values,
            )
        )
        self.assertTrue(
            _array_equal_nan(
                modisaod_dataset.get_feature_data_array(
                    "Corrected_Optical_Depth_Land_Solution_3_Land_55"
                )
                .squeeze()
                .values,
                manual_test_data["Corrected_Optical_Depth_Land"]
                .isel({"Solution_3_Land": 1})
                .values,
            )
        )
        self.assertTrue(
            _array_equal_nan(
                modisaod_dataset.get_feature_data_array(
                    "Corrected_Optical_Depth_Land_Solution_3_Land_65"
                )
                .squeeze()
                .values,
                manual_test_data["Corrected_Optical_Depth_Land"]
                .isel({"Solution_3_Land": 2})
                .values,
            )
        )
        self.assertTrue(
            _array_equal_nan(
                modisaod_dataset.get_feature_data_array(
                    "Corrected_Optical_Depth_Land_wav2p1"
                )
                .squeeze()
                .values,
                manual_test_data["Corrected_Optical_Depth_Land_wav2p1"].values,
            )
        )
        self.assertTrue(
            _array_equal_nan(
                modisaod_dataset.get_feature_data_array("Mass_Concentration_Land")
                .squeeze()
                .values,
                manual_test_data["Mass_Concentration_Land"].values,
            )
        )


    def testMODISFRPParser(self):
        manual_test_data = xr.open_dataset(self.split_modisfrp_test_file)
        modisFRP_parser = MODISFRPParser()
        modisFRP_dataset = modisFRP_parser.parse_file(self.split_modisfrp_test_file)
        self.assertTrue((manual_test_data['time'].values == modisFRP_dataset.get_times()).all())
        self.assertTrue((manual_test_data['lat'].values == modisFRP_dataset.get_latitudes()).all())
        self.assertTrue((manual_test_data['lon'].values == modisFRP_dataset.get_longitudes()).all())
        self.assertTrue(_array_equal_nan(manual_test_data['FRP'].values,
                                         modisFRP_dataset.get_feature_data_array('FRP').values))


if __name__ == "__main__":
    ut.main(argv=["first-arg-is-ignored"], exit=False)
