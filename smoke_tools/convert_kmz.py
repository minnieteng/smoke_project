"""Functions to transform Bluesky US `.kmz` files into:
1) netCDF files for PM 2.5 data
2) CSVs for fire information data
"""
import datetime
import errno
import logging
import os
import sys
import tempfile
from zipfile import ZipFile

import click
import numpy as np
import pandas as pd
import xarray as xr
import xmltodict
from PIL import Image

logging.getLogger(__name__).addHandler(logging.NullHandler())


def convert_kmz(kmz_file, output_directory):
    """

    :param kmz_file:
    :param output_directory:
    :return:
    """
    with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
        temp_dir_path = os.path(temp_dir)
        kmz = ZipFile(kmz_file, "r")
        kmz.extractall(temp_dir_path)
        with open(os.path.join(temp_dir_path, "doc.kml")) as fd:
            doc = xmltodict.parse(fd.read())
            fire_information, pm25 = doc["kml"]["Document"]["Folder"]
            pm25_to_netcdf(pm25, output_directory, temp_dir_path)
            fire_info_to_csv(fire_information, output_directory)


def pm25_to_netcdf(pm25, output_directory, temp_dir_path):
    """

    :param pm25:
    :param output_directory:
    :param temp_dir_path:
    :return:
    """
    for folder in pm25["Folder"]:
        times = []
        img_arrs = []
        coords = []
        dims_known = False
        for entry in folder["GroundOverlay"]:
            img = os.path.join(temp_dir_path, entry["Icon"]["href"])
            img_arr = img_to_array(img)
            img_arr = np.moveaxis(img_arr, -1, 0)
            if not dims_known:
                rgba, x, y = map(np.arange, img_arr.shape)
                dims_known = True
            timestamp = np.datetime64(entry["TimeSpan"]["begin"])
            times.append(timestamp)
            coords.append(entry["LatLonBox"])
            img_arrs.append(img_arr)
        da = xr.DataArray(
            img_arrs, coords=[times, rgba, x, y], dims=["time", "rgba", "x", "y"]
        )
        ds = xr.Dataset({folder["name"]: da})
        name = os.path.join(output_directory, folder["name"].replace(" ", "") + ".nc")
        ds.to_netcdf(
            name, mode="w", format="NETCDF4",
        )
        del (ds, da, times, img_arrs, coords, img_arr)


def img_to_array(img_path):
    image = Image.open(img_path)
    return np.asarray(image)


def fire_info_to_csv(fire_information, output_directory):
    """

    :param fire_information:
    :param output_directory:
    :return:
    """
    main_df = None
    for fire_event in fire_information["Folder"]:
        name = fire_event["name"]
        fire_event_dict = {
            "name": None,
            "date": [],
            "info": [],
            "lat": [],
            "lon": [],
            "altitude": [],
        }
        for entry in fire_event["Placemark"]["description"]["html"]["body"]["div"][
            "div"
        ][1]["div"][1]["div"]:
            str_info = entry["#text"].split(": ")
            date = str_info[0].split(", ")
            month, day = date[1].split(" ")
            date = month[0:3] + " " + day + " " + date[2]
            date = convert_time(date)
            fire_event_dict["date"].append(date)
            fire_event_dict["info"].append(str_info[1])
            lat, lon, alt = fire_event["Placemark"]["Point"]["coordinates"].split(",")
            fire_event_dict["lat"].append(lat)
            fire_event_dict["lon"].append(lon)
            fire_event_dict["altitude"].append(alt)
        names = [name for i in range(len(fire_event_dict["date"]))]
        fire_event_dict["name"] = names
        df = pd.DataFrame.from_dict(fire_event_dict)
        if main_df is None:
            main_df = df
        else:
            main_df = main_df.append(df)
    main_df.to_csv(os.path.join(output_directory, "fire_information.csv"))


def convert_time(date_time):
    _format = "%b %d %Y"
    datetime_str = datetime.datetime.strptime(date_time, _format)
    return datetime_str


def mkdir(dirname):
    """ Check that a directory can be made then make it

    :param dirname: Directory to create
    :type dirname: str or os.path
    :return: True if successful
    :rtype: bool
    """
    if not os.path.exists(os.path.dirname(dirname)):
        try:
            dirname = os.path.dirname(dirname)
            os.makedirs(dirname)
            logging.info(f"convert_kmz.mkdir created directory {dirname}")
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                logging.exception(exc.errno)
                return False
    return True


@click.command(help=convert_kmz.__doc__)
@click.argument("kmz_file", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path(writable=True))
@click.option(
    "-v",
    "--verbosity",
    default="warning",
    show_default=True,
    type=click.Choice(("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")),
    help="""
        Choose the logging level. Defaults to WARNING. 
        WARNING, ERROR, and CRITICAL will only report when Murphy's law kicks in
    """,
)
def cli(kmz_file, output_directory, logging_level):
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    convert_kmz(kmz_file, output_directory)
