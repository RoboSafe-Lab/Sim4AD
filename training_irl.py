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

    driving_styles = {0: 'Cautious', 1: 'Normal', 2: 'Aggressive', -1: 'General'}

    new_buffer = []
    for episode in config.episodes:
        episode_id = episode.recording_id

        # load all buffered features
        if args.driving_style_idx < 0:
            logger.info(f'Loading {episode_id} for training.')
            for key, value in driving_styles.items():
                if key >= 0:
                    buffer = load_pkl(value + '_' + episode_id)
                    if buffer is None:
                        continue
                    # If new_buffer is empty, simply assign buffer to it
                    if not new_buffer:
                        new_buffer = buffer.copy()
                    else:
                        # Otherwise, concatenate the current buffer to each corresponding sublist in new_buffer
                        new_buffer = [nb + b for nb, b in zip(new_buffer, buffer)]

        # load buffered features for a specific driving style
        else:
            buffer = load_pkl(driving_styles[args.driving_style_idx] + '_' + episode_id)
            # file not existing
            if buffer is None:
                continue
            logger.info(f'Loading {driving_styles[args.driving_style_idx]} {episode_id} for training.')
            # If new_buffer is empty, simply assign buffer to it
            if not new_buffer:
                new_buffer = buffer.copy()
            else:
                # Otherwise, concatenate the current buffer to each corresponding sublist in new_buffer
                new_buffer = [nb + b for nb, b in zip(new_buffer, buffer)]

    # Run MaxEnt IRL, sequential optimization, avoid using multiprocessing
    irl_instance.maxent_irl(buffer=new_buffer)

    if irl_instance.save_training_log:
        logger.info('Saved training log.')
        with open(driving_styles[args.driving_style_idx] + "training_log.pkl", "wb") as file:
            pickle.dump(irl_instance.training_log, file)


if __name__ == "__main__":
    main()
