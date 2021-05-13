"""
Functions to transform MODIS Fire Radiative Power shapefiles to netCDF files
"""
import datetime
import logging
import os
import sys
import geopandas as gpd
import xarray as xr
import click
import smoke.utils.utilities as utilities
import sparse
import numpy as np
from pathlib import Path

from smoke.load.parsers import *


logging.getLogger(__name__).addHandler(logging.NullHandler())


def convert_time(date, time):
    _format = "%Y-%m-%d %H%M"
    datetime_str = datetime.datetime.strptime(f"{date} {time}", _format)
    return datetime_str


def convert_shp(shp_file, output_directory):
    ds = shp_to_xr(shp_file)
    name = os.path.join(output_directory, Path(shp_file).stem)
    utilities.mkdir(output_directory)
    os.mkdir(output_directory)
    ds.to_netcdf(
        name, mode="w", format="NETCDF4",
    )


def shp_to_xr(shp_file):
    f = gpd.read_file(shp_file)
    ds = xr.Dataset.from_dataframe(f)

    times = []
    for date, time in zip(ds["ACQ_DATE"].values, ds["ACQ_TIME"].values):
        times.append(convert_time(date, time))

    FRP = ds.FRP.values
    lats = ds.LATITUDE.values
    lons = ds.LONGITUDE.values
    size = np.shape(FRP)[0]
    indices = [i for i in range(size)]
    arr = sparse.COO([indices, indices, indices], FRP, shape=(size, size, size))

    da = xr.DataArray(
        arr,
        coords={
            "time": times,
            "lat": lats,
            "lon": lons
        },
        dims=["time", "lat", "lon"],
    )
    ds = xr.Dataset({"FRP": da})
    return ds


def split_frp_hdf_to_xr(split_frp_hdf_file):
    """ Loading and returning data from split frp hdf files created by
    smoke/split/frp_splitter.py as of file format
    including and previous to 2020/06/29

    :param split_frp_hdf_file: path to split frp hdf file
    :type split_frp_hdf_file: str
    :returns: xarray dataset with specs for GeographicalDataset
    :rtype: xr.Dataset
    """
    parser = MODISFRPParser()

    return parser.parse_file(split_frp_hdf_file)


@click.command(help=convert_shp.__doc__)
@click.argument("shp_file", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path(writable=True))
@click.option(
    "-v",
    "--verbosity",
    default="WARNING",
    show_default=True,
    type=click.Choice(("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")),
    help="""
        Choose the logging level. Defaults to WARNING.
        WARNING, ERROR, and CRITICAL will only report when Murphy's law kicks in
    """,
)
def cli(shp_file, output_directory, logging_level):
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    convert_shp(shp_file, output_directory)
