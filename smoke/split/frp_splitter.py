import os
import click
import logging
import numpy as np
import geopandas as gpd
import xarray as xr
from datetime import datetime, timedelta


def convert_time(date, time):
    _format = "%Y-%m-%d %H%M"
    datetime_str = datetime.strptime(f"{date} {time}", _format)
    return datetime_str


def partition_frp_to_hour(shp_file, save_dir, logger):
    """ Partitions a MODIS FRP fire archive file into correctly dimensioned .hdf
    files of time, lat, and lon, with each file containing data of a single unique hour
    in the dataset. Hour is chosen as for ten years its at most 87600 files and
    by minute was too many files, and by day was too large individual files, so hour
    is a compromise.

    :param shp_file: Absolute file path to frp archive .shp file to partition
    :type shp_file: str
    :param save_dir: Directory to save seperated .hdf files into
    :type save_dir: str
    :param logger: Logger to output to
    :type logger: logging.Logger
    """
    # Load data into an xarray for ease in use
    f = gpd.read_file(shp_file)
    ds = xr.Dataset.from_dataframe(f)
    del f

    # Get times for each data entry and extract unique hours to group on
    times = []
    for date, time in zip(ds["ACQ_DATE"].values, ds["ACQ_TIME"].values):
        times.append(convert_time(date, time))
    hours = np.unique(np.array(list(map(lambda x: np.datetime64(f"{x.year}-{x.month:02d}-{x.day:02d}T{x.hour:02d}"), times))))

    # Pull FRP, lat, lon, and earlier times into np.array's
    ds_FRP = ds.FRP.values
    ds_lats = ds.LATITUDE.values
    ds_lons = ds.LONGITUDE.values
    ds_times = np.array(times).astype(np.datetime64)

    logger.info('Starting split by hour')

    # Group files on each hour, and populate a full 2D grid of np.nan, then put save out into proper xarray Dataset
    for _hour in hours:

        # Find FRPs lons and lats during day
        indices_in_day = (_hour <= ds_times) & (ds_times < _hour+np.timedelta64(1,'h'))
        ordered_frp_in_day = ds_FRP[indices_in_day]
        ordered_lon_in_day = ds_lons[indices_in_day]
        ordered_lat_in_day = ds_lats[indices_in_day]
        ordered_times_in_day = ds_times[indices_in_day]

        # Ensure both space axis have unique entries by adding very small random noise if any repeat
        # Latitudes have ten thousandth precision so add noise rounded to nearest hundred thousandth
        while (np.unique(ordered_lon_in_day, return_counts=True)[1] > 1).any():
            ordered_lon_in_day += np.round(np.random.random_sample(ordered_lon_in_day.size)*0.0001, 5)
        while (np.unique(ordered_lat_in_day, return_counts=True)[1] > 1).any():
            ordered_lat_in_day += np.round(np.random.random_sample(ordered_lat_in_day.size)*0.0001, 5)

        # Make 1D perpendicular axes of lon, lat, time and corresponding indices array, for use in populating grid
        time_axis = np.sort(list(set(list(ordered_times_in_day))))
        time_indices = np.indices((time_axis.size,))[0]
        lon_axis = np.sort(ordered_lon_in_day)
        lon_indices = np.indices((lon_axis.size,))[0]

        # Flip latitude axis so largest latitude corresponds to lowest row indices (top of array) and
        # smallest latitude corresponds to highest row indices (bottom of array) matching geography
        lat_axis = np.flipud(np.sort(ordered_lat_in_day))
        lat_indices = np.indices((lat_axis.size,))[0]

        # Make grid of np.nan to populate, shape is (time, lat, lon)
        grid_2D = np.empty((time_axis.size, lat_axis.size, lon_axis.size))
        grid_2D[:] = np.nan

        # Find indices in grid each frp measurement's lon, lat, and time is
        x_assignments = [lon_indices[lon_axis == lon][0] for lon in ordered_lon_in_day]
        y_assignments = [lat_indices[lat_axis == lat][0] for lat in ordered_lat_in_day]
        time_assignments = [time_indices[time_axis == _time][0] for _time in ordered_times_in_day]
        frp_data = list(ordered_frp_in_day)

        # Assign frps to earlier found spots in grid
        for data, _time, y, x in zip(frp_data, time_assignments, y_assignments, x_assignments):
            grid_2D[_time][y][x] = data

        # Load grid into xarray DataSet with feature name FRP
        da = xr.DataArray(
            grid_2D,
            dims=["time", "lat", "lon"],
            coords={
                "time": time_axis,
                "lat": lat_axis,
                "lon": lon_axis
            }
        )
        new_ds = xr.Dataset({"FRP": da})

        # Create .hdf file name unique to hour and in save_dir, and save to that path
        new_hdf_save = os.path.join(save_dir, f"split_fire_archive_hour_{str(_hour).replace('-', '')}.hdf")
        new_ds.to_netcdf(new_hdf_save)

        logger.info(f"Created: split_fire_archive_hour_{str(_hour).replace('-', '')}.hdf")

    logger.info('Finished splitting')


@click.command(help="Partitions frp archive shp file into hdf files of each hour with correct dimension implementation of time, lat, lon")
@click.argument("shp_file", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path(writable=True))
@click.option(
    "--logging-level",
    default="INFO",
    show_default=True,
    type=click.Choice(("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")),
    help="Select logging level from DEBUG, INFO, WARNING, ERROR, CRITICAL, default INFO."
)
def cli(shp_file, output_directory, logging_level):

    # Set logging
    logging.basicConfig(
        level=logging_level,
        format="[%(asctime)s] %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    partition_frp_to_hour(shp_file, output_directory, logger)


if __name__ == "__main__":
    cli()
