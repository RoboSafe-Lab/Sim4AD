import abc
import logging
import os
import numpy as np
from typing import List, Dict, Set

from openautomatumdronedata.dataset import droneDataset
from sim4ad.agentstate import AgentState, AgentMetadata

logger = logging.getLogger(__name__)


class EpisodeConfig:
    """ Metadata about an episode """

    def __init__(self, config):
        self.config = config

    @property
    def recording_id(self) -> str:
        """ Unique ID identifying the episode"""
        return self.config.get('recording_id')


class EpisodeMetadata:
    def __init__(self, config):
        self.config = config

    @property
    def max_speed(self) -> float:
        """ The speed limit at the episode location. """
        return self.config.get("speedLimit")

    @property
    def frame_rate(self) -> int:
        """ Frame rate of the episode recording. """
        return int(self.config.get("frameRate"))


class EpisodeLoader(abc.ABC):
    """ Abstract class that every EpisodeLoader should represent. Also keeps track of registered subclasses. """
    EPISODE_LOADERS = {}  # Each EpisodeLoader can register its own class as loader here

    def __init__(self, scenario_config):
        self.scenario_config = scenario_config

    def load(self, config: EpisodeConfig, road_map=None, **kwargs):
        raise NotImplementedError()

    @classmethod
    def register_loader(cls, loader_name: str, loader):
        if not issubclass(loader, cls):
            raise ValueError(f"Given loader {loader} is not an EpisodeLoader!")
        if loader_name not in cls.EPISODE_LOADERS:
            cls.EPISODE_LOADERS[loader_name] = loader
        else:
            logger.warning(f"Loader {loader} with name {loader_name} already registered!")

    @classmethod
    def get_loader(cls, scenario_config: "ScenarioConfig") -> "EpisodeLoader":
        """ Get the episode loader as specified within the ScenarioConfig

        Args:
            scenario_config: The scenario configuration

        Returns:
            The corresponding EpisodeLoader
        """
        loader = cls.EPISODE_LOADERS[scenario_config.data_format]
        if loader is None:
            raise ValueError('Invalid data format')
        return loader(scenario_config)


class Frame:
    """ A snapshot of time in the data set"""

    def __init__(self, time: float, dead_ids: Set[int] = None):
        """ Create a new frame.

        Args:
            time: Time of the frame recording
            dead_ids: These agents are treated as dead
        """
        self.time = time
        self.dead_ids = dead_ids if dead_ids is not None else set()
        self._agents = {}

    @property
    def all_agents(self) -> Dict[int, AgentState]:
        return self._agents

    @property
    def agents(self) -> Dict[int, AgentState]:
        return {k: v for k, v in self._agents.items() if k not in self.dead_ids}

    def add_agent_state(self, agent_id: int, state: AgentState):
        """ Add a new agent with its specified state.

        Args:
            agent_id: The ID of the Agent whose state is being recorded
            state: The state of the Agent
        """
        if agent_id not in self._agents:
            self._agents[agent_id] = state
        else:
            logger.warning(f"Agent {agent_id} already in Frame. Adding state skipped!")


class Episode:
    """ An episode that is represented with a collection of Agents and their corresponding frames. """

    def __init__(self, config: EpisodeConfig, agents, frames, statWorld, opendrive_map):
        self.config = config
        self.agents = agents
        self.frames = frames
        self.statWorld = statWorld
        self.map_file = opendrive_map

    def __repr__(self):
        return f"Episode {self.config.recording_id}; {len(self.agents)} agents"

    def __iter__(self):
        self.t = 0
        return self


class DatasetEpisodeLoader(EpisodeLoader):
    def load(self, config: EpisodeConfig, agent_types: List[str] = None, scale: float = None):
        path_to_dataset_folder = os.path.join(self.scenario_config.data_root, config.recording_id)

        dataset = droneDataset(path_to_dataset_folder)
        dynWorld = dataset.dynWorld
        statWorld = dataset.statWorld
        dynObjectList = dynWorld.get_list_of_dynamic_objects()
        agents = {}

        # search for the minimum time
        min_time = np.inf
        for dynObj in dynObjectList:
            if dynObj.get_first_time() < min_time:
                min_time = dynObj.get_first_time()
        time_vec = np.arange(min_time, dynWorld.maxTime, dynWorld.delta_t)
        time_vec = np.append(time_vec, dynWorld.maxTime)
        frames = [Frame(t) for t in time_vec]

        for dynObj in dynObjectList:
            agent_id = dynObj.UUID
            agents[agent_id] = dynObj
            agent_meta = self._agent_meta_from_track_meta(dynObj)

            for idx, time in enumerate(dynObj.time):
                state = self._state_from_tracks(dynObj, idx, agent_meta)
                differences = np.abs(time_vec - time)
                closest_index = np.argmin(differences)
                frames[closest_index].add_agent_state(agent_id, state)
                frames[closest_index].time = time

        self.xodr_ = path_to_dataset_folder + '/staticWorld.xodr'
        return Episode(config, agents, frames, statWorld, self.xodr_)

    @staticmethod
    def _state_from_tracks(dynObj, idx, metadata: AgentMetadata = None):
        time = dynObj.time[idx]
        heading = dynObj.psi_vec[idx]
        position = np.array([dynObj.x_vec[idx], dynObj.y_vec[idx]])
        velocity = np.array([dynObj.vx_vec[idx], dynObj.vy_vec[idx]])
        acceleration = np.array([dynObj.ax_vec[idx], dynObj.ay_vec[idx]])

        return AgentState(time, position, velocity, acceleration, heading, metadata)

    @staticmethod
    def _agent_meta_from_track_meta(dynObj):
        return AgentMetadata(width=dynObj.width,
                             length=dynObj.length,
                             agent_type=dynObj.type,
                             initial_time=dynObj.time[0],
                             final_time=dynObj.time[-1])
