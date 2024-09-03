from loguru import logger
import pickle
import numpy as np

from sim4ad.irlenv.irl import IRL
from sim4ad.util import parse_args
from sim4ad.path_utils import get_config_path
from sim4ad.data import ScenarioConfig
from sim4ad.path_utils import write_common_property

def load_pkl(file_name: str):
    try:
        with open(file_name + '_buffer.pkl', 'rb') as file:
            # Load the data from the file
            buffer = pickle.load(file)
        return buffer
    except FileNotFoundError:
        # Return None if the file does not exist
        return None

def creat_new_buffer(driving_style, episode_id):
    new_buffer = []
    buffer = load_pkl(driving_style + '_' + episode_id)
    if buffer is None:
        return None
    # If new_buffer is empty, simply assign buffer to it
    if not new_buffer:
        new_buffer = buffer.copy()
    else:
        # Otherwise, concatenate the current buffer to each corresponding sublist in new_buffer
        new_buffer = [nb + b for nb, b in zip(new_buffer, buffer)]

    return new_buffer

def get_mean_std(buffer_scenes):
    """ Get the mean and std of features"""
    trajectories = []
    # get the mean and std for z-score normalization
    for scene in buffer_scenes:
        for trajectory in scene:
            trajectories.append(trajectory[2])
    trajectories = np.array(trajectories)
    mean = np.mean(trajectories, axis=0)
    std = np.std(trajectories, axis=0)
    write_common_property('IRL_MEAN', mean.tolist())
    write_common_property('IRL_STD', std.tolist())
    return mean, std

def z_score_normalization(new_buffer, mean=None, std=None):
    # normalize features, new_buffer[1] is simulated feature
    new_buffer[0] = (new_buffer[0] - mean) / (std + 1e-8)
    new_buffer[0] = new_buffer[0].tolist()
    for scene in new_buffer[1]:
        for i, trajectory in enumerate(scene):
            normalized_trajectory = (trajectory[2] - mean) / (std + 1e-8)
            scene[i] = (trajectory[0], trajectory[1], normalized_trajectory, trajectory[3])

def main():
    """The code is used to train irl reward weights based on the saved features"""
    args = parse_args()
    config = ScenarioConfig.load(get_config_path(args.map))
    irl_instance = IRL(multiprocessing=False, num_processes=12,
                       save_buffer=False, save_training_log=True)

    # get mean and std for normalization
    new_buffer = []
    for episode in config.episodes:
        episode_id = episode.recording_id

        # load all buffered features
        if args.driving_style == 'All':
            logger.info(f'Loading all episodes for training.')
            for driving_style in ['Aggressive', 'Normal', 'Cautious']:
                new_buffer = creat_new_buffer(driving_style, episode_id)
                if new_buffer is None:
                    continue
        # load buffered features for a specific driving style
        else:
            logger.info(f'Loading {args.driving_style} {episode_id} for training.')
            new_buffer = creat_new_buffer(args.driving_style, episode_id)
            if new_buffer is None:
                continue


    # Run MaxEnt IRL, sequential optimization, avoid using multiprocessing
    irl_instance.maxent_irl(buffer=new_buffer)

    if irl_instance.save_training_log:
        logger.info('Saved training log.')
        with open(args.driving_style + "training_log.pkl", "wb") as file:
            pickle.dump(irl_instance.training_log, file)


if __name__ == "__main__":
    main()
