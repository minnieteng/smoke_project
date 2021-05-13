"""
Functions to transform MODIS Aerosol Optical Depth files to netCDF files
"""
import datetime
import logging
import os
import sys
import xarray as xr
import click
import smoke.utils.utilities as utilities
from pathlib import Path

from smoke.load.parsers import *


logging.getLogger(__name__).addHandler(logging.NullHandler())


def convert_modis_aod(modis_aod_file, output_directory):
    ds = modis_aod_to_xr(modis_aod_file)
    name = os.path.join(output_directory, Path(modis_aod_file).stem)
    utilities.mkdir(output_directory)
    os.mkdir(output_directory)
    ds.to_netcdf(
        name, mode="w", format="NETCDF4",
    )


def modis_aod_to_xr(modis_aod_file):
    """loading and returning data from modis Aerosol Optical Depth
    files as of file format including and previous too 2020/06/30

    :param modis_aod_file: path to raw modis aod data file
    :type modis_aod_file: str
    :returns: xarray dataset with specs for GeographicalDataset
    :rtype: xr.Dataset
    """
    parser = MODISAODParser()

    return parser.parse_file(modis_aod_file)


@click.command(help=convert_modis_aod.__doc__)
@click.argument("modis_aod_file", type=click.Path(exists=True))
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
def cli(modis_aod_file, output_directory, logging_level):
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    convert_modis_aod(modis_aod_file, output_directory)
