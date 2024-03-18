from loguru import logger
import pickle
from typing import List

from sim4ad.irlenv.irl import IRL
from sim4ad.util import parse_args
from sim4ad.path_utils import get_config_path
from sim4ad.data.data_loaders import DatasetDataLoader


def load_dataset(config_path: str = None, evaluation_data: List[str] = None):
    """Loading clustered trajectories"""
    data_loader = DatasetDataLoader(config_path, evaluation_data)
    data_loader.load()

    return data_loader


def main():
    """The code can only train on one episode"""
    args = parse_args()
    train_data = load_dataset(get_config_path(args.map), ['train'])

    irl_instance = IRL(episode=train_data.scenario.episodes[0], multiprocessing=False, num_processes=12,
                       save_buffer=False, save_training_log=False)
    # compute features
    irl_instance.get_simulated_features()

    # normalize features
    irl_instance.normalize_features()

    # Run MaxEnt IRL, sequential optimization, avoid using multiprocessing
    irl_instance.maxent_irl()

    if irl_instance.save_training_log:
        logger.info('Saved training log.')
        with open("training_log.pkl", "wb") as file:
            pickle.dump(irl_instance.training_log, file)


if __name__ == "__main__":
    main()
