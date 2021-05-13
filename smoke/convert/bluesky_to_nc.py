"""
Functions to transform Bluesky model output files to netCDF files
"""
import datetime
import logging
import os
import sys
import numpy as np
import xarray as xr
import click
import smoke.utils.utilities as utilities
from pathlib import Path

from smoke.load.parsers import *


logging.getLogger(__name__).addHandler(logging.NullHandler())


def convert_bluesky(bluesky_file, output_directory):
    ds = bluesky_to_xr(bluesky_file)
    name = os.path.join(output_directory, Path(bluesky_file).stem)
    utilities.mkdir(output_directory)
    os.mkdir(output_directory)
    ds.to_netcdf(
        name, mode="w", format="NETCDF4",
    )


def bluesky_to_xr(file_path):
    """ Loading and returning data from bluesky
    netcdf4 files as of file format including and previous too 2020/06/29

    :param file_path: path to raw bluesky data file
    :type file_path: str
    :returns: xarray dataset with specs for GeographicalDataset
    :rtype: xr.Dataset
    """
    parser = BlueSkyParser()

    return parser.parse_file(file_path)


@click.command(help=convert_bluesky.__doc__)
@click.argument("bluesky_file", type=click.Path(exists=True))
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
def cli(bluesky_file, output_directory, logging_level):
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    convert_bluesky(bluesky_file, output_directory)
