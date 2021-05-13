import click
import logging
import yaml
import numpy as np
from datetime import datetime, timedelta

from smoke.amalgamate.Amalgamator import Amalgamator
from smoke.amalgamate.errors.errors import IncompletePredictionSet, NoCorrespondingLabel


@click.command(
    help = (
        """ Runs amalgamator across all PM2.5 label times specified in config by
        datetime_start_ISO8601, datetime_stop_ISO8601, and time_resolution_h
        time range specifications. Creates a pytorch tensor for each of those
        times with concatenated PM2.5 ground truths and prediction data.
        """
    )
)
@click.argument('path_to_config')
@click.argument('output_directory')
@click.option(
    "--logging-level",
    default="INFO",
    show_default=True,
    type=click.Choice(("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")),
    help="Select logging level from DEBUG, INFO, WARNING, ERROR, CRITICAL, default INFO."
)
def main(path_to_config, output_directory, logging_level):
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

    # Create array of all times to create tensors for
    tensor_label_times = np.arange(
        datetime.strptime(loaded_yaml['datetime_start_ISO8601'], '%Y-%m-%dT%H:%M:%S'),
        datetime.strptime(loaded_yaml['datetime_stop_ISO8601'], '%Y-%m-%dT%H:%M:%S'),
        timedelta(hours=loaded_yaml['time_resolution_h'])
    )
    tensor_label_times = [datetime.strptime(str(d), '%Y-%m-%dT%H:%M:%S.000000') for d in tensor_label_times]
    logger.info(f"Creating tensors from {tensor_label_times[0]} to {tensor_label_times[-1]} every {loaded_yaml['time_resolution_h']}h")

    # From config load which prediction datasets to use for pytorch tensor
    # creation
    tensor_datasets = []
    if loaded_yaml['firework']:
        tensor_datasets.append("firework_closest")
    if loaded_yaml['firework_2ndclosest']:
        tensor_datasets.append("firework_2ndclosest")
    if loaded_yaml['firework_3rdclosest']:
        tensor_datasets.append("firework_3rdclosest")
    if loaded_yaml['firework_4thclosest']:
        tensor_datasets.append("firework_4thclosest")
    if loaded_yaml['bluesky']:
        tensor_datasets.append("bluesky_closest")
    if loaded_yaml['bluesky_2ndclosest']:
        tensor_datasets.append("bluesky_2ndclosest")
    if loaded_yaml['modisAOD']:
        tensor_datasets.append("modisaod")
    if loaded_yaml['modisFRP']:
        tensor_datasets.append("modisfrp")
    if loaded_yaml['NOAA']:
        tensor_datasets.append("noaa")
    logger.info(f"Datasets included in tensor: {tensor_datasets}")

    # Create amalgamator with file directory specifications in config
    amalgamator = Amalgamator(
        pm25_labels_folder = loaded_yaml['pm25_labels_folder'],
        firework_ftsg_folder = loaded_yaml['firework_ftsg_folder'],
        bluesky_ftsg_folder = loaded_yaml['bluesky_ftsg_folder'],
        modisaod_ftsg_folder = loaded_yaml['modisaod_ftsg_folder'],
        modisfrp_ftsg_folder = loaded_yaml['modisfrp_ftsg_folder'],
        noaa_grid_folder = loaded_yaml['noaa_grid_folder']
    )

    # Create tensor for every label time in tensor
    logger.info("Starting tensor creation over time range")

    for label_time in tensor_label_times:
        try:
            amalgamator.make_pytorch_tensor(
                label_time,
                tensor_datasets,
                save_directory=output_directory
            )
            logger.info(f"Successfully created tensor for time: {label_time}")

        except IncompletePredictionSet as e:
            logger.info(f"Skipping {label_time}:\n{str(e)}")

        except NoCorrespondingLabel as e:
            logger.info(f"Skipping {label_time}:\n{str(e)}")

    logger.info("Finished tensor creation over time range")


if __name__ == "__main__":
    main()
