"""
Functions to transform Smoke Plume files to netCDF files
"""

import logging
import os
import sys
import xarray as xr
import click
import smoke.utils.utilities as utilities
import geopandas as gpd
import numpy as np
from pathlib import Path


logging.getLogger(__name__).addHandler(logging.NullHandler())


def convert_smoke_plume(smoke_plume_file, output_directory):
    ds = smoke_plume_to_xr(smoke_plume_file)
    name = os.path.join(output_directory, Path(smoke_plume_file).stem)
    utilities.mkdir(output_directory)
    os.mkdir(output_directory)
    ds.to_netcdf(
        name, mode="w", format="NETCDF4",
    )


def smoke_plume_to_xr(smoke_plume_file):
    """ loading and returning data from NOAA smoke plume polygon shapefiles

    :param file_path: path to raw NOAA file, file name contains date of data collection in the form of %Y%m%d
    e.g. "hms_smoke20191009.shp"
    :type file_path: str
    :returns: xarray dataset with specs for GeographicalDataset
    :rtype: xr.Dataset
    """
    gdf = gpd.read_file(smoke_plume_file)

    # define start/end times in the form of %Y%m%d %h%m example: 2018220 1202
    start_time = gdf.Start
    end_time = gdf.End
    data_array = np.append(start_time, end_time)

    # get lon, lat
    lon_lat = gdf["geometry"]

    # append variables in dataset with lat, lon, start/end time of record
    new_data_array = np.append(data_array, lon_lat)

    # Add to dataset
    output_data_set = xr.Dataset({"SmokePlumeBinary": new_data_array})

    return output_data_set


@click.command(help=convert_smoke_plume.__doc__)
@click.argument("smoke_plume_file", type=click.Path(exists=True))
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
def cli(smoke_plume_file, output_directory, logging_level):
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    convert_smoke_plume(smoke_plume_file, output_directory)
