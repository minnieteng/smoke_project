import click
import logging
import os
import tarfile
import tempfile
import yaml

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.colors import Normalize
from datetime import datetime, timedelta

from smoke.box.FeatureTimeSpaceGrid import load_FeatureTimeSpaceGrid
from smoke.clean.cleaners import *
from smoke.load.parsers import *


def update_ax(i, axs, times, xr_dataarrays, ftsg, time_res_h, feature_index, box_cells):

    time_to_plot = times[i]

    # Get global min and global max and creation consistent normalization
    grid_2D_for_plot = []
    for da in xr_dataarrays:
        space_grids = da.values[(time_to_plot-np.timedelta64(time_res_h, 'h') < da['time'].values) &
                                (da['time'].values <= time_to_plot)]
        space_grid = np.nanmean(space_grids, axis=0)
        grid_2D_for_plot.append(space_grid)
    global_min = np.nanmin(grid_2D_for_plot)
    global_max = np.nanmax(grid_2D_for_plot)
    global_min = np.nanmin([np.nanmin(ftsg.get_grid_nan_converted(0)), global_min])
    global_max = np.nanmax([np.nanmax(ftsg.get_grid_nan_converted(0)), global_max])
    global_norm = Normalize(global_min, global_max)

    # For data array plot mean of data in that time bin
    for im, da, grid_2D in zip(axs[:-1], xr_dataarrays, grid_2D_for_plot):
        im.plot(ftsg.box.poly.exterior.xy[1], ftsg.box.poly.exterior.xy[0], color='yellow', linestyle='solid', linewidth=2, zorder=2)
        im.contourf(da['lon'].values, da['lat'].values, grid_2D, zorder=1, norm=global_norm)

    im = axs[-1]
    im.contourf(np.arange(1, box_cells+1, 1),
                np.arange(1, box_cells+1, 1),
                ftsg.get_grid_nan_converted(0)[feature_index][ftsg.get_times() == time_to_plot][0], zorder=1, norm=global_norm)

    return


def generate_visual_animation_test(ftsg_to_test,
                                   output_dir,
                                   original_data_dir,
                                   data_window_end_from_FTSG_start_h,
                                   data_window_size_h,
                                   parser,
                                   cleaner,
                                   feature_index):
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = temp_dir.name
    f_tar = tarfile.open(ftsg_to_test, mode='r')
    f_tar.extractall(temp_dir_path)
    with open(os.path.join(temp_dir_path, 'meta.json'), 'r') as f_json:
        meta_data = json.load(f_json)

    # Load data to find everything
    loaded_ftsg = load_FeatureTimeSpaceGrid(ftsg_to_test)
    time_res_h = meta_data['time resolution (h)']
    unique_name = meta_data['unique file name']
    times = loaded_ftsg.get_times()
    ftsg_start_time = times[0] - np.timedelta64(time_res_h, 'h')
    feature = loaded_ftsg.get_features()[feature_index]

    # Find all files used to make FTSG and load GeographicalDataset of each
    data_files = cleaner.get_files(
        original_data_dir,
        datetime.strptime(str(ftsg_start_time), '%Y-%m-%dT%H:%M:%S.000000')-timedelta(hours=data_window_end_from_FTSG_start_h+data_window_size_h),
        datetime.strptime(str(ftsg_start_time), '%Y-%m-%dT%H:%M:%S.000000')-timedelta(hours=data_window_end_from_FTSG_start_h)
    )
    das = [parser.parse_file(f).get_feature_data_array(feature) for f in data_files]

    # Only make animation if any data_files were loaded
    if (len(data_files) > 0):
        fig, axs = plt.subplots(1, len(das)+1, figsize=(16*len(das)+1, 8))
        # Format all data plots
        for ax, file_name, da in zip(axs[:-1], data_files, das):
            ax.set_title(file_name)
            ax.set_xlim(da['lon'].min(), da['lon'].max())
            ax.set_ylim(da['lat'].min(), da['lat'].max())
        # Format FTSG plot
        axs[-1].set_title('FTSG')
        axs[-1].set_xlim(1, loaded_ftsg.box.get_num_cells())
        axs[-1].set_ylim(1, loaded_ftsg.box.get_num_cells())
        axs[-1].invert_yaxis()
        fig.tight_layout()
        finished_anim = anim.FuncAnimation(fig,
                                           update_ax,
                                           frames=len(times),
                                           fargs=(axs, times, das, loaded_ftsg, time_res_h, feature_index, loaded_ftsg.box.get_num_cells()))
        finished_anim.save(os.path.join(output_dir, unique_name+'_test.mp4'), fps=2)
    else:
        raise Exception('Skipping from lack of data files')


@click.command(
    help=""" Generate an animation through all times in all files of files at
    each point in time for diagnostics. Makes a file for each FTSG given by
    recreating data window used to make FTSG and displaying files which should
    be populating each time bin for each time bin in FTSG.

    """
)
@click.argument("config_path")
def main(config_path):
    # Set logging
    logging.basicConfig(
        level='INFO',
        format="[%(asctime)s] %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    # Load config yaml from given location
    with open(config_path, 'r') as f:
        loaded_yaml = yaml.safe_load(f)
    logger.info(f"Loaded yaml config at {config_path}")

    # Select which cleaner and parser to use
    cur_yaml = loaded_yaml['currently_testing']
    if cur_yaml['firework']:
        parser = FireworkParser()
        cleaner = FireworkCleaner()
    elif cur_yaml['bluesky']:
        parser = BlueSkyParser()
        cleaner = BlueSkyCleaner()
    elif cur_yaml['modisAOD']:
        parser = MODISAODParser()
        cleaner = MODISAODCleaner()
    elif cur_yaml['modisFRP']:
        parser = MODISFRPParser()
        cleaner = MODISFRPleaner()
    else:
        assert False
    logger.info(f"Testing {parser} and {cleaner}")

    # Load FTSGs to test
    ftsgs = [os.path.join(loaded_yaml['ftsg_test_directory'], f) for f in os.listdir(loaded_yaml['ftsg_test_directory']) if (('tar.gz' in f) and (loaded_yaml['ftsg_keyword'] in f))]

    # Create animation for each ftsg
    for f in ftsgs:
        try:
            generate_visual_animation_test(
                f,
                loaded_yaml['output_directory'],
                loaded_yaml['original_data_directory'],
                loaded_yaml['data_window_end_from_FTSG_start_h'],
                loaded_yaml['data_window_size_h'],
                parser,
                cleaner,
                loaded_yaml['feature_index'])
            logger.info(f"Completed animation for {f}")
        catch Exception as e:
            if 'Skipping from lack of data files' in str(e):
                logger.info(f"Skipping {f} as lack data files for it")
            else:
                raise e


if __name__ == "__main__":
    main()
