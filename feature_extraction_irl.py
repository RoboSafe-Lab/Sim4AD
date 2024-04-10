from loguru import logger
from preprocessing import load_clustering

from sim4ad.irlenv.irl import IRL
from sim4ad.path_utils import get_config_path
from sim4ad.util import parse_args, load_dataset
from sim4ad.data.episode import Episode


def episode_split(episode, clustering):
    """Split the episode according to the three different driving styles"""
    episode_splits = {
        'Aggressive': Episode(config=episode.config, agents={}, frames=episode.frames, statWorld=episode.statWorld,
                              opendrive_map=episode.map_file),
        'Normal': Episode(config=episode.config, agents={}, frames=episode.frames, statWorld=episode.statWorld,
                          opendrive_map=episode.map_file),
        'Cautious': Episode(config=episode.config, agents={}, frames=episode.frames, statWorld=episode.statWorld,
                            opendrive_map=episode.map_file)
    }
    episode_id = episode.config.recording_id
    for aid, agent in episode.agents.items():
        driving_style = clustering[episode_id + '/' + aid]
        episode_splits[driving_style].agents[aid] = agent

    return episode_splits


def feature_extraction_irl(episode, driving_style=''):
    """Extracting the features for IRL in the dataset"""
    irl_instance = IRL(episode=episode, multiprocessing=False, num_processes=12,
                       save_buffer=False, save_training_log=False)
    # compute features
    irl_instance.get_simulated_features()

    # save buffered features
    irl_instance.save_buffer_data(driving_style)


def main():
    args = parse_args()
    dataset = load_dataset(get_config_path(args.map), ['valid'])
    episode = dataset.scenario.episodes[args.episode_idx]

    # whether using clustered data or the entire data without clustering
    if not args.clustering:
        # when using no cluster, the entire dataset will be used as a whole
        logger.info(f'Computing {episode.config.recording_id} as a whole.')
        feature_extraction_irl(episode)
    else:
        # load clustering result JSON file
        clustering = load_clustering(f"scenarios/configs/{args.map}_drivingStyle.json")
        # split the episode
        episode_splits = episode_split(episode, clustering)
        for driving_style, episode in episode_splits.items():
            logger.info(f'Computing {episode.config.recording_id} with {driving_style} cluster.')
            feature_extraction_irl(episode, driving_style)


if __name__ == "__main__":
    main()
