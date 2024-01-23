from dataclasses import dataclass
from openautomatumdronedata.dataset import droneDataset

@dataclass
class Feature4Training:
    rel_dx_front_ego: float
    rel_dy_front_ego: float
    vx_front_ego: float
    vy_front_ego: float
    ax_front_ego: float
    ay_front_ego: float
    psi_front_ego: float

class ExtractObservationAction:
    """
    Extract observation and action from the clustered data
    """

    def __init__(self, episodes, clustered_dataframe):
        """
        Args:
            episodes: the episodes from the original dataset
            clustered_dataframe: clustered trajectories
        """
        self._clustered_dataframe = clustered_dataframe
        self._episodes = episodes
        self._agent_feature = {'time': [], 'rel_dx': [], 'rel_dy': [], 'vx': [], 'vy': [], 'ax': [], 'ay': [], 'psi': [],
                               'distance_left_lane_marking': [], 'distance_right_lane_marking': []}

    def extract_observation(self):
        """Extract observations"""
        dynWorld = droneDataset.dynWorld

        grouped_cluster = self._clustered_dataframe.groupby('label')
        # we train one driving style as an example
        driving_style_a = grouped_cluster.get_group(0)

        for episode in self._episodes:
            matching_rows = driving_style_a.iloc[:, 0] == episode.config.recording_id
            if matching_rows.any():
                agent_ids = driving_style_a[matching_rows]['agent_id']
                for agent_id in agent_ids:
                    agent = episode.agents[agent_id]

                    for inx, t in enumerate(agent.time):
                        surrounding_agents = agent.object_relation_dict_list[inx]
                        dynObjectList = dynWorld.get_list_of_dynamic_objects_for_specific_time(t)

                        for surrounding_agent_name, surrounding_agent in surrounding_agents.items():
                            if surrounding_agent is not None:
                                pass
                        agent_feature = Feature4Training(rel_dx_front=surrounding_agents, )

                    agent.lat_dist_dict_vec
            pass
        pass


