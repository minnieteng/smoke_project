#!/usr/bin/env python
# coding: utf-8

import click
import logging
import os
import yaml
import numpy as np
from datetime import datetime, timedelta
from multiprocessing import Pool

from smoke.clean.cleaners import *

def use_cleaner_to_save_day_FTSG(day_to_find_data_for,
                                 buffer_time_h,
                                 time_limit_h,
                                 grid_time_res_h,
                                 box,
                                 cleaner,
                                 file_directory,
                                 output_directory,
                                 file_prefix=''):
    """ Saves data on the day_to_find_data_for by using cleaner to create a FTSG for that day,
    using files in file_directory between some buffer_time_h before day_to_find_data_for at 00:00:00,
    and up to time_limit_h hours before that time. Saves resulting FTSG in output_directory

    :param day_to_find_data_for: Day to create FTSG for
    :type day_to_find_data_for: numpy.datetime64
    :param buffer_time_h: Hours before day_to_find_data_for that data for day shouldn't be considered
                          as realistically would not be out yet
    :type buffer_time_h: int
    :param time_limit_h: Hours before buffer_time_h to consider data within
    :type time_limit_h: int
    :param grid_time_res_h: Time resolution for grid in hours
    :type grid_time_res_h: int
    :param box: Theoretical space grid to use as for space assignment
    :type box: smoke_tools.box.Box
    :param cleaner: Cleaner to use to create FTSG
    :type cleaner: smoke.clean.cleaners.GenericCleaner
    :param file_directory: Path to directory containing raw data files
    :type file_directory: str
    :param output_directory: Path to directory to save output FTSG
    :type output_directory: str
    :param file_prefix: Prefix to add to output FTSG file to differentiate ones of that type, default ''
    :param file_prefix: str, optional
    """
    logger = logging.getLogger(__name__)
    data_timerange_end = day_to_find_data_for-timedelta(hours=buffer_time_h)
    logger.info(f"Running cleaner to create FTSG with axis range {day_to_find_data_for} to {day_to_find_data_for+timedelta(days=1)} based on files released from {data_timerange_end-timedelta(hours=time_limit_h)} to {data_timerange_end}")
    day_FTSG = cleaner.create_featuretimespacegrid(file_directory,
                                                   box,
                                                   data_timerange_end-timedelta(hours=time_limit_h),
                                                   data_timerange_end,
                                                   day_to_find_data_for,
                                                   day_to_find_data_for+timedelta(days=1),
                                                   grid_time_res_h)
    day_FTSG.save(output_directory, file_prefix)


@click.command(
    help = (
        """ Runs cleaners with run = True across all files in time range given
        in config, producing a saved FeatureTimeSpaceGrid (FTSG) for each day
        in that time range. Each FTSG uses data from a data window to
        released files specified also in the config.
        """
    )
)
@click.argument('path_to_config')
@click.option(
    "--logging-level",
    default="INFO",
    show_default=True,
    type=click.Choice(("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")),
    help="Select logging level from DEBUG, INFO, WARNING, ERROR, CRITICAL, default INFO."
)
def main(path_to_config, logging_level):
    # Set logging
    logging.basicConfig(
        level=logging_level,
        format="[%(asctime)s] %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    # Load config yaml from given location
    with open(path_to_config, 'r') as f:
        loaded_yaml = yaml.safe_load(f)
    logger.info(f"Loaded yaml config at {path_to_config}")

    # Create BCBox to assign all space to
    bc_box = BCBox(loaded_yaml.get('grid_res_km'))
    logger.info(f"Generated space grid with {loaded_yaml.get('grid_res_km')} km resolution")

    # Create datetime objects for all days in time range
    time_config = loaded_yaml.get('timerange')
    date_range = list(
        map(
            lambda x: datetime.strptime(
                str(x),
                '%Y-%m-%d'
            ),
            list(
                np.arange(
                    np.datetime64(time_config.get('start')),
                    np.datetime64(time_config.get('stop')),
                    np.timedelta64(1, 'D')
                )
            )
        )
    )
    logger.info(f"Running cleaners to create saved daily FTSG\'s in range {date_range[0].strftime('%Y-%m-%d')} to {date_range[-1].strftime('%Y-%m-%d')}")

    # Create threads for multiprocessing
    pool = Pool(processes=loaded_yaml.get('threads'))

    # Firework
    fw_config = loaded_yaml.get('firework')
    if fw_config.get('run'):
        logger.info("Starting firework cleaner run over date range")
        buffer_before_grid_h = fw_config.get('time_we_at_stand_before_grid_h')
        for fw_sub_config in [fw_config.get('first_closest'),
                              fw_config.get('second_closest'),
                              fw_config.get('third_closest'),
                              fw_config.get('fourth_closest')]:
            if fw_sub_config.get('run'):
                args = []
                for day in date_range:
                    args.append((
                        day,
                        buffer_before_grid_h+fw_sub_config.get('data_window_end_n_hours_before_standing'),
                        fw_sub_config.get('data_window_size_h'),
                        fw_config.get('grid_time_res_h'),
                        bc_box,
                        FireworkCleaner(),
                        fw_config.get('file_directory'),
                        fw_config.get('output_directory'),
                        fw_sub_config.get('file_prefix')
                    ))
                pool.starmap(use_cleaner_to_save_day_FTSG, args)
        logger.info("Finished firework cleaner run over date range")

    # Refresh bc_box for RAM sake
    del bc_box
    bc_box = BCBox(loaded_yaml.get('grid_res_km'))

    # Bluesky Canada
    bs_config = loaded_yaml.get('bluesky')
    if bs_config.get('run'):
        logger.info("Starting bluesky cleaner run over date range")
        buffer_before_grid_h = bs_config.get('time_we_at_stand_before_grid_h')
        for bs_sub_config in [bs_config.get('first_closest'),
                              bs_config.get('second_closest')]:
            if bs_sub_config.get('run'):
                args = []
                for day in date_range:
                    args.append((
                        day,
                        buffer_before_grid_h+bs_sub_config.get('data_window_end_n_hours_before_standing'),
                        bs_sub_config.get('data_window_size_h'),
                        bs_config.get('grid_time_res_h'),
                        bc_box,
                        BlueSkyCleaner(),
                        bs_config.get('file_directory'),
                        bs_config.get('output_directory'),
                        bs_sub_config.get('file_prefix')
                    ))
                pool.starmap(use_cleaner_to_save_day_FTSG, args)
        logger.info("Finished bluesky cleaner run over date range")

    # Refresh bc_box for RAM sake
    del bc_box
    bc_box = BCBox(loaded_yaml.get('grid_res_km'))

    # MODIS AOD
    ma_config = loaded_yaml.get('modisAOD')
    if ma_config.get('run'):
        logger.info("Starting modis AOD cleaner run over date range")
        args = []
        for day in date_range:
            args.append((
                day,
                # Physical Measurement so just consider entire day's data plus
                # time_res_h hours previous to that
                -24,
                24+ma_config.get('grid_time_res_h'),
                ma_config.get('grid_time_res_h'),
                bc_box,
                MODISAODCleaner(),
                ma_config.get('file_directory'),
                ma_config.get('output_directory'),
                'modisaod_'
            ))
        pool.starmap(use_cleaner_to_save_day_FTSG, args)
        logger.info("Finished modis AOD cleaner run over date range")

    # Refresh bc_box for RAM sake
    del bc_box
    bc_box = BCBox(loaded_yaml.get('grid_res_km'))

    # MODIS FRP
    mf_config = loaded_yaml.get('modisFRP')
    if mf_config.get('run'):
        logger.info("Starting modis FRP cleaner run over date range")
        args = []
        for day in date_range:
            args.append((
                day,
                # Physical Measurement so just consider entire day's data plus
                # time_res_h hours previous to that
                -24,
                24+mf_config.get('grid_time_res_h'),
                mf_config.get('grid_time_res_h'),
                bc_box,
                MODISFRPCleaner(),
                mf_config.get('file_directory'),
                mf_config.get('output_directory'),
                'modisfrp_'
            ))
        pool.starmap(use_cleaner_to_save_day_FTSG, args)
        logger.info("Finished modis FRP cleaner run over date range")


if __name__ == "__main__":
    main()
