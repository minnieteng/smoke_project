"""
Functions to transform firework model files to netCDF files
"""
import logging
import os
import sys
import numpy as np
import datetime
import xarray as xr
import click
import smoke.utils.utilities as utilities
from pathlib import Path

from smoke.load.parsers import *


logging.getLogger(__name__).addHandler(logging.NullHandler())


def convert_firework(firework_file, output_directory):
    ds = firework_to_xr(firework_file)
    name = os.path.join(output_directory, Path(firework_file).stem)
    utilities.mkdir(output_directory)
    os.mkdir(output_directory)
    ds.to_netcdf(
        name, mode="w", format="NETCDF4",
    )


def firework_to_xr(firework_file):
    """ Loading and returning data from firework
    geotiff files as of file format including and previous to 2020/06/29

    :param firework_file: path to raw firework data file, file name must have date as first 10 digits in form
                      %Y%m%d%H
    :type firework_file: str
    :returns: xarray dataset with specs for GeographicalDataset
    :rtype: xr.Dataset
    """
    parser = FireworkParser()

    return parser.parse_file(firework_file)


@click.command(help=convert_firework.__doc__)
@click.argument("firework_file", type=click.Path(exists=True))
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
def cli(firework_file, output_directory, logging_level):
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    convert_firework(firework_file, output_directory)
