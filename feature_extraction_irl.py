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
    args = parse_args()
    dataset = load_dataset(get_config_path(args.map))
    episode = dataset.scenario.episodes[args.episode_idx]
    logger.info(f'Computing {episode.config.recording_id}')
    irl_instance = IRL(episode=episode, multiprocessing=False, num_processes=12,
                       save_buffer=False, save_training_log=False)
    # compute features
    irl_instance.get_simulated_features()

    # normalize features
    irl_instance.normalize_features()


if __name__ == "__main__":
    main()
