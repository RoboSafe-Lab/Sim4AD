import json
import abc
from loguru import logger
from typing import List, Dict

from sim4ad.data.episode import EpisodeConfig, EpisodeLoader, Episode
from sim4ad.opendrive.map import Map

"""
Code based on: https://github.com/uoe-agents/IGP2/blob/ogrit/igp2/data/scenario.py
"""


class ScenarioConfig:
    """Metadata about a scenario used for goal recognition"""

    def __init__(self, config_dict):
        self.config_dict = config_dict

    @classmethod
    def load(cls, file_path):
        """Loads the scenario metadata into from a json file
        Args:
            file_path (str): path to the file to load
        Returns:
            ScenarioConfig: metadata about the scenario
        """
        with open(file_path) as f:
            scenario_meta_dict = json.load(f)
        return cls(scenario_meta_dict)

    @property
    def name(self) -> str:
        """Name of the scenario"""
        return self.config_dict.get('name')

    @property
    def opendrive_file(self) -> str:
        """ Path to the *.xodr file specifying the OpenDrive map"""
        return self.config_dict.get('opendrive_file')

    @property
    def data_format(self) -> str:
        """Format in which the data is stored"""
        return self.config_dict.get('data_format')

    @property
    def data_root(self) -> str:
        """ Path to directory in which the data is stored"""
        return self.config_dict.get('data_root')

    @property
    def episodes(self) -> List[EpisodeConfig]:
        """list of dict: Configuration for all episodes for this scenario"""
        return [EpisodeConfig(c) for c in self.config_dict.get('episodes')]

    @property
    def background_image(self) -> str:
        """Path to background image"""
        return self.config_dict.get('background_image')

    @property
    def background_px_to_meter(self) -> float:
        """ Pixels per meter in background image"""
        return self.config_dict.get('background_px_to_meter')

    @property
    def scale_down_factor(self) -> int:
        """ Scale down factor for visualisation"""
        return self.config_dict.get('scale_down_factor')

    @property
    def reachable_pairs(self) -> List[List[List[float]]]:
        """ Pairs of points, where the second point should be reachable from the first
           Can be used for validating maps"""
        return self.config_dict.get('reachable_pairs')

    @property
    def dataset_split(self) -> Dict[str, List[int]]:
        """ Get the which data split each episode belongs to """
        return self.config_dict.get('dataset_split', None)

    @property
    def agent_types(self) -> List[str]:
        """ Gets which types of agents to keep from the data set """
        return self.config_dict.get("agent_types", None)

    @property
    def scaling_factor(self) -> float:
        """ Constant factor to account for mismatch in the scale of the recordings and the size of the map """
        return self.config_dict.get("scaling_factor", None)


class Scenario(abc.ABC):
    """ Represents an arbitrary driving scenario with interactions broken to episodes. """

    def __init__(self, config: ScenarioConfig):
        """ Initialize new Scenario based on the given ScenarioConfig and read map data from config. """
        self.config = config
        self._episodes = None
        self._opendrive_map = None
        self._loader = EpisodeLoader.get_loader(self.config)
        # self.load_map()

    def load_map(self):
        if self.config.opendrive_file:
            self._opendrive_map = Map.parse_from_opendrive(self.config.opendrive_file)
        else:
            raise ValueError(f"OpenDrive map was not specified!")

    @property
    def opendrive_map(self) -> Map:
        """ Return the OpenDrive Map of the Scenario. """
        return self._opendrive_map

    @property
    def episodes(self) -> List[Episode]:
        """ Retrieve a list of loaded Episodes. """
        return self._episodes

    @property
    def loader(self) -> EpisodeLoader:
        """ The EpisodeLoader of the Scenario. """
        return self._loader

    @classmethod
    def load(cls, file_path: str, split: List[str] = None):
        """ Initialise a new Scenario from the given config file.
        Args:
            file_path: Path to the file defining the scenario
            split: The data set splits to load as given by indices. If None, load all.
        Returns:
            A new Scenario instance
        """
        raise NotImplementedError


class DatasetScenario(Scenario):
    @classmethod
    def load(cls, file_path: str, split: List[str] = None):
        config = ScenarioConfig.load(file_path)
        scenario = cls(config)
        scenario.load_episodes(split)
        return scenario

    def load_episodes(self, split: List[str] = None) -> List[Episode]:
        """ Load all/the specified Episodes as given in the ScenarioConfig. Store episodes in field episode """
        if split is not None:
            indices = []
            for s in split:
                indices.extend(self.config.dataset_split[s])
            to_load = [conf for i, conf in enumerate(sorted(self.config.episodes, key=lambda x: x.recording_id))
                       if i in indices]
        else:
            to_load = sorted(self.config.episodes, key=lambda x: x.recording_id)

        logger.info(f"Loading {len(to_load)} episode(self).")
        episodes = []
        for idx, config in enumerate(to_load):
            logger.info(f"Loading Episode {idx + 1}/{len(to_load)}")
            episode = self._loader.load(config,
                                        agent_types=self.config.agent_types,
                                        scale=self.config.scaling_factor)
            episodes.append(episode)

        self._episodes = episodes
        return episodes

    def load_episode(self, episode_id) -> Episode:
        """ Load specific Episode with the given ID. Does not append episode to member field episode. """
        return self._loader.load(self.config.episodes[episode_id])
