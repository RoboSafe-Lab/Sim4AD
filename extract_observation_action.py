from dataclasses import dataclass

from typing import Optional


@dataclass
class SurroundingAgentFeature:
    rel_dx: Optional[float] = None
    rel_dy: Optional[float] = None
    vx: Optional[float] = None
    vy: Optional[float] = None
    ax: Optional[float] = None
    ay: Optional[float] = None
    psi: Optional[float] = None


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

    def extract_observation(self):
        """Extract observations"""

        grouped_cluster = self._clustered_dataframe.groupby('label')
        # we train one driving style as an example
        driving_style_a = grouped_cluster.get_group(0)

        observations = []
        for episode in self._episodes:
            episode_id = episode.config.recording_id
            matching_rows = driving_style_a.iloc[:, 0] == episode_id
            if matching_rows.any():
                agent_ids = driving_style_a[matching_rows]['agent_id']
                for agent_id in agent_ids:
                    agent = episode.agents[agent_id]

                    ego_agent_feature = {'time': agent.time, 'vx': agent.vx_vec, 'vy': agent.vy_vec, 'ax': agent.ax_vec,
                                         'ay': agent.ay_vec, 'psi': agent.psi_vec, 'aid': agent_id, 'eid': episode_id,
                                         'distance_left_lane_marking': agent.distance_left_lane_marking,
                                         'distance_right_lane_marking': agent.distance_right_lane_marking,
                                         'surroundings': []}
                    for inx, t in enumerate(agent.time):
                        surrounding_agents_features = {}
                        # get surrounding agent's information
                        surrounding_agents = agent.object_relation_dict_list[inx]
                        for surrounding_agent_relation, surrounding_agent_id in surrounding_agents.items():
                            surrounding_agent_feature = SurroundingAgentFeature()
                            if surrounding_agent_id is not None:
                                surrounding_agent = episode.agents[surrounding_agent_id]
                                # lat_dist_dict_vec, long_dist_dict_vec are none, so should be recalculated
                                long_distance, lat_distance = agent.get_lat_and_long(t, surrounding_agent)
                                surrounding_agent_inx = surrounding_agent.next_index_of_specific_time(t)
                                surrounding_agent_feature.rel_dx = long_distance
                                surrounding_agent_feature.rel_dy = lat_distance
                                surrounding_agent_feature.vx = surrounding_agent.vx_vec[surrounding_agent_inx]
                                surrounding_agent_feature.vy = surrounding_agent.vy_vec[surrounding_agent_inx]
                                surrounding_agent_feature.ax = surrounding_agent.ax_vec[surrounding_agent_inx]
                                surrounding_agent_feature.ay = surrounding_agent.ay_vec[surrounding_agent_inx]
                                surrounding_agent_feature.psi = surrounding_agent.psi_vec[surrounding_agent_inx]

                            surrounding_agents_features[surrounding_agent_relation] = surrounding_agent_feature

                        ego_agent_feature['surroundings'].append(surrounding_agents_features)

                    observations.append(ego_agent_feature)

    def extract_action(self):
        """Extract action here"""
        pass
