import sys
from sim4ad.util import parse_args, load_dataset
from extract_observation_action import ExtractObservationAction
import json
from sim4ad.path_utils import get_config_path
from loguru import logger


def load_clustering(file_path: str):
    # Open the JSON file for reading
    with open(file_path, 'r') as file:
        # Load its content and make a new Python dictionary out of it
        clustering = json.load(file)

    return clustering


def main():
    args = parse_args()

    # creat two dicts, one for training and one for testing
    data_splits = ['valid']
    driving_styles = {0: 'Cautious', 1: 'Normal', 2: 'Aggressive', -1: ''}
    driving_style = driving_styles[args.driving_style_idx]
    if driving_style != '':
        logger.info(f'{driving_style} reward weights are loaded')
    else:
        logger.info('No reward weights are loaded.')

    for split in data_splits:
        data = load_dataset(get_config_path(args.map), [split])

        episodes = data.scenario.episodes

        # extract observations and actions
        extractor = ExtractObservationAction(split, args.map, episodes, driving_style)
        extractor.extract_demonstrations()


if __name__ == '__main__':
    sys.exit(main())
