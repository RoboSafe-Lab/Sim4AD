from loguru import logger
import pickle

from sim4ad.irlenv.irl import IRL
from sim4ad.util import parse_args
from sim4ad.path_utils import get_config_path
from sim4ad.data import ScenarioConfig


def load_pkl(file_name: str):
    try:
        with open(file_name + '_buffer.pkl', 'rb') as file:
            # Load the data from the file
            buffer = pickle.load(file)
        return buffer
    except FileNotFoundError:
        # Return None if the file does not exist
        return None


def main():
    """The code is used to train irl reward weights based on the saved features"""
    args = parse_args()
    config = ScenarioConfig.load(get_config_path(args.map))
    irl_instance = IRL(multiprocessing=False, num_processes=12,
                       save_buffer=False, save_training_log=True)

    for episode in config.episodes:
        episode_id = episode.config.recording_id

        if args.driving_style == '':
            # load the buffered features
            buffer = load_pkl(episode_id)
        else:
            buffer = load_pkl(args.driving_style + '_' + episode_id)
        # file not existing
        if buffer is None:
            continue
        logger.info(f'Loading {args.driving_style} {episode_id} for training.')

        # Run MaxEnt IRL, sequential optimization, avoid using multiprocessing
        irl_instance.maxent_irl(buffer=buffer)

    if irl_instance.save_training_log:
        logger.info('Saved training log.')
        with open(args.driving_style + "training_log.pkl", "wb") as file:
            pickle.dump(irl_instance.training_log, file)


if __name__ == "__main__":
    main()
