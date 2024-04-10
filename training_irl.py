from loguru import logger
import pickle

from sim4ad.irlenv.irl import IRL
from sim4ad.util import parse_args, load_dataset
from sim4ad.path_utils import get_config_path


def load_pkl(file_name: str):
    with open(file_name + '_buffer.pkl', 'rb') as file:
        # Load the data from the file
        buffer = pickle.load(file)

    return buffer


def main():
    """The code is used to train irl reward weights based on the saved features"""
    args = parse_args()
    train_data = load_dataset(get_config_path(args.map), ['train'])
    irl_instance = IRL(multiprocessing=False, num_processes=12,
                       save_buffer=False, save_training_log=True)

    for episode in train_data.scenario.episodes:
        episode_id = episode.config.recording_id

        # load the buffered features
        buffer = load_pkl(episode_id)
        logger.info(f'Loading {episode_id} for training.')

        # Run MaxEnt IRL, sequential optimization, avoid using multiprocessing
        irl_instance.maxent_irl(buffer=buffer)

        if irl_instance.save_training_log:
            logger.info('Saved training log.')
            with open("training_log.pkl", "wb") as file:
                pickle.dump(irl_instance.training_log, file)


if __name__ == "__main__":
    main()
