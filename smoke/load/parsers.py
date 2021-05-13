import os
import pvl
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from smoke.load.datasets import GeographicalDataset


class GenericParser(ABC):

    def __init__(self):
        pass

    def parse_file(self, file_path):
        """ Parses a raw data file, returning a geographical
        dataset of data

        :param file_path: path to raw data file
        :type file_path: str
        """
        data_set = self.convert_raw_to_dataset(file_path)
        return GeographicalDataset(data_set)

    @abstractmethod
    def convert_raw_to_dataset(self, file_path):
        """ Abstract method to override, should convert a raw file at file_path
        to one usable in GeographicalDataset object

        :param file_path: path to raw data file
        :type file_path: str
        :returns: xarray dataset with specs for GeographicalDataset
        :rtype: xr.Dataset
        """
        pass


class FireworkParser(GenericParser):

    def convert_raw_to_dataset(self, file_path):
        """ Override of abstract method specific for loading and returning data from firework
        geotiff files as of file format including and previous too 2020/06/29

        :param file_path: path to raw firework data file, file name must have date as first 10 digits in form
                          %Y%m%d%H
        :type file_path: str
        :returns: xarray dataset with specs for GeographicalDataset
        :rtype: xr.Dataset
        """
        data_array = xr.open_rasterio(file_path)
        data_array.name = file_path

        # Replace names in dataset with lat, lon, time
        new_data_array = data_array.rename({"x": "lon", "y": "lat", "band": "time"})

        # Get time from name in datetime form %Y%m%d%H (not in meta data) and create time array
        int_to_parse = os.path.split(new_data_array.name)[1][:10]
        assert int_to_parse.isdigit()  # assert str int is parsable
        start_time = datetime.strptime(int_to_parse, "%Y%m%d%H")
        time_coords = np.array(
            list(
                map(
                    lambda x: start_time + timedelta(hours=x),
                    range(new_data_array["time"].size),
                )
            )
        )

        # Replace time array with datetimes under info saying each band is an hour forward forcast
        new_data_array["time"] = time_coords

        # Add to dataset as forecast of PM 2.5
        ds = xr.Dataset({"PM25Forecast": new_data_array})

        return ds


class BlueSkyParser(GenericParser):

    def convert_raw_to_dataset(self, file_path):
        """ Override of abstract method specific for loading and returning data from bluesky
        netcdf4 files as of file format including and previous too 2020/06/29

        :param file_path: path to raw bluesky data file
        :type file_path: str
        :returns: xarray dataset with specs for GeographicalDataset
        :rtype: xr.Dataset
        """
        # Load raw data
        data = xr.open_dataset(file_path)

        # Create longitude coordinates from center for better accuracy
        indices_right_from_center = np.arange(int(data.attrs["NCOLS"] / 2)) + 1.0
        indices_left_from_center = np.flipud(indices_right_from_center) * -1.0
        x_indices_from_center = np.append(
            np.append(indices_left_from_center, np.array([0.0])), indices_right_from_center,
        )
        x_coords = x_indices_from_center * data.attrs["XCELL"] + data.attrs["XCENT"]

        # Create latitude coordinates from center for better accuracy
        indices_up_from_center = np.arange(int(data.attrs["NROWS"] / 2)) + 1.0
        indices_down_from_center = np.flipud(indices_up_from_center) * -1.0
        y_indices_from_center = np.append(
            np.append(indices_down_from_center, np.array([0.0])), indices_up_from_center
        )
        y_coords = y_indices_from_center * data.attrs["YCELL"] + data.attrs["YCENT"]
        y_coords = np.flipud(y_coords)  # Flip latitudes so highest latitudes are
                                        # lowest row indices (highest row position)
                                        # and lowest latitudes are highest row
                                        # indices (lowest row position) to match
                                        # geography

        # Convert TFLAG data variable into flat array of timestamps assuming UTC since across canada
        time_coords = np.array(
            list(
                map(
                    lambda x: datetime.strptime(
                        "{}{:06d}".format(x[0], x[1]), "%Y%j%H%M%S"
                    ),
                    data["TFLAG"].squeeze().values,
                )
            )
        )

        # Create new dataset with dims only of lat, lon, and time
        new_data = data.squeeze()
        new_data = new_data.rename({"ROW": "lat", "COL": "lon", "TSTEP": "time"})

        # Create new xarray Dataset of PM 2.5 with earlier derived coords
        new_data_array = xr.DataArray(
            new_data["PM25"],
            coords={"lon": x_coords, "lat": y_coords, "time": time_coords},
        )
        new_data_set = xr.Dataset({"PM25Forecast": new_data_array})

        return new_data_set


class MODISAODParser(GenericParser):

    def convert_raw_to_dataset(self, file_path):
        """ Override of abstract method specific for loading and returning data from modis
        aod files as of file format including and previous too 2020/08/12

        :param file_path: path to raw modis aod data file
        :type file_path: str
        :returns: xarray dataset with specs for GeographicalDataset
        :rtype: xr.Dataset
        """
        # Load raw data
        data = xr.open_dataset(file_path, engine="pynio")
        data = data.assign_coords({'lat':data['Latitude'], 'lon':data['Longitude']})  # Assign lat, lon as coordinates not attrs

        # Seperate out 3 COD Land arrays as well as CODLW2P1 and MCL
        CODL47 = (
            data["Corrected_Optical_Depth_Land"]
            .isel({"Solution_3_Land_mod04": 0})
        )
        CODL55 = (
            data["Corrected_Optical_Depth_Land"]
            .isel({"Solution_3_Land_mod04": 1})
        )
        CODL65 = (
            data["Corrected_Optical_Depth_Land"]
            .isel({"Solution_3_Land_mod04": 2})
        )
        CODLW2P1 = data["Corrected_Optical_Depth_Land_wav2p1"]
        MCL = data["Mass_Concentration_Land"]

        # Store in a new data set renaming to standard
        new_data_set = xr.Dataset(
            {
                "Corrected_Optical_Depth_Land_Solution_3_Land_47": CODL47,
                "Corrected_Optical_Depth_Land_Solution_3_Land_55": CODL55,
                "Corrected_Optical_Depth_Land_Solution_3_Land_65": CODL65,
                "Corrected_Optical_Depth_Land_wav2p1": CODLW2P1,
                "Mass_Concentration_Land": MCL,
            }
        )

        # Add an additional dimension of time, as all occur at same on from metadata
        pvl_meta_data = pvl.loads(data.attrs['CoreMetadata_0'])
        time = datetime.strptime(
            (
                pvl_meta_data['INVENTORYMETADATA']['RANGEDATETIME']['RANGEBEGINNINGDATE']['VALUE'] +
                pvl_meta_data['INVENTORYMETADATA']['RANGEDATETIME']['RANGEBEGINNINGTIME']['VALUE']
            ),
            "%Y-%m-%d%H:%M:%S.%f",
        )
        new_data_set = new_data_set.assign_coords({"time": time})
        new_data_set = new_data_set.expand_dims("time")

        return new_data_set


class MODISFRPParser(GenericParser):

    def convert_raw_to_dataset(self, file_path):
        """ Override of abstract method specific for loading and returning data
        from split modis frp files (after using smoke/split/frp_splitter.py) as
        of file format including and previous too 2020/08/10.

        :param file_path: path to split modis frp data file
        :type file_path: str
        :returns: xarray dataset with specs for GeographicalDataset
        :rtype: xr.Dataset
        """
        ds = xr.open_dataset(file_path)

        return ds
