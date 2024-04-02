import sys
from sim4ad.util import parse_args, load_dataset
from extract_observation_action import ExtractObservationAction
import json
from sim4ad.path_utils import get_config_path


def load_clustering(file_path: str):
    # Open the JSON file for reading
    with open(file_path, 'r') as file:
        # Load its content and make a new Python dictionary out of it
        clustering = json.load(file)

    return clustering


def main():
    args = parse_args()
    data_splits = ['valid']
    # creat two dicts, one for training and one for testing
    for split in data_splits:
        data = load_dataset(get_config_path(args.map), [split])

        episodes = data.scenario.episodes

        if not args.clustering:
            # when using no cluster, the entire dataset will be used as a whole
            clustering = None
        else:
            # load clustering result JSON file
            clustering = load_clustering(f"scenarios/configs/{args.map}_drivingStyle.json")

        # extract observations and actions
        extractor = ExtractObservationAction(split, args.map, episodes, clustering)
        extractor.extract_demonstrations()


if __name__ == '__main__':
    sys.exit(main())
