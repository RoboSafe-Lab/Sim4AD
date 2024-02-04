import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import rotate, translate
from typing import Tuple

from sim4ad.irlenv.planner import planner


class IRLEnv:
    def __init__(self, agent, current_inx: int, scenario_map):
        self.planned_trajectory = None
        self.agent = agent
        self.current_inx = current_inx

        self._position = np.array([agent.x_vec[current_inx], agent.y_vec[current_inx]])
        self._velocity = np.array([agent.vx_vec[current_inx], agent.vy_vec[current_inx]])
        self._heading = agent.psi_vec[current_inx]
        self._acceleration = np.array([agent.ax_vec[current_inx], agent.ay_vec[current_inx]])

        self._best_lane = scenario_map.best_lane_at(point=self._position, heading=self._heading)
        self._s, self._d = self.position_frenet(point=self._position, reference_line=self._best_lane.midline)

    def theta_frenet(self, distance: float):
        """
            Get the velocity vect on the frenet system

            :arg distance: the s value of the agent on frenet
            :return theta_frenet: the angle between the ego and the lane midline
        """
        lane_heading = self._best_lane.get_heading_at(distance)
        delta_theta = self._heading - lane_heading
        delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi

        rotation_matrix = np.array([
            [np.cos(delta_theta), np.sin(delta_theta)],
            [-np.sin(delta_theta), np.cos(delta_theta)]
        ])

        return rotation_matrix

    @staticmethod
    def position_frenet(point: np.ndarray, reference_line):
        """Get the s and d values along the lane midline"""
        p = Point(point)
        closest_point = reference_line.interpolate(reference_line.project(p))
        s = reference_line.project(p)
        d = p.distance(closest_point)

        return s, d

    @staticmethod
    def compute_angle(point1, point2):
        """
        Compute the angle (in degrees) from horizontal for the line defined by two points.
        """
        dx = point2.x - point1.x
        dy = point2.y - point1.y
        return np.degrees(np.arctan2(dy, dx))

    def position_local(self, s, d):
        """
        Convert Frenet coordinates (s, d) to local Cartesian coordinates on a curved road using Shapely.

        :param s: Longitudinal distance along the path.
        :param d: Lateral offset from the path, positive to the left, negative to the right.
        :param path_points: A list of tuples representing the points of the path (curved road).
        :return: Tuple representing the Cartesian coordinates (x, y) on the curved road.
        """
        # Create a LineString from the path points
        path = self._best_lane.midline

        # Interpolate the point at distance 's' along the path
        point_on_path = path.interpolate(s)

        # Compute the angle of the tangent at the interpolated point
        if s < path.length:
            # Compute tangent by looking ahead by a small distance
            look_ahead_distance = min(1.0, path.length - s)
            look_ahead_point = path.interpolate(s + look_ahead_distance)
            angle_degrees = self.compute_angle(point_on_path, look_ahead_point)
        else:
            # Handle the case where 's' is at the end of the path by looking backward
            look_back_point = path.interpolate(s - 1.0)
            angle_degrees = self.compute_angle(look_back_point, point_on_path)

        # Calculate the offset point by rotating and translating the original point
        # Note: Shapely's rotate function uses counter-clockwise rotation, so 'd' is positive to the left
        offset_point = translate(rotate(point_on_path, angle_degrees, origin=point_on_path, use_radians=False), d,
                                 0)

        return offset_point.x, offset_point.y

    def trajectory_planner(self, target_point, target_speed, time_horizon):
        """
        Plan a trajectory for the human-like (IRL) vehicle.
        """

        rotation_matrix = self.theta_frenet(distance=self._s)

        s_v, d_v = np.dot(rotation_matrix, self._velocity)
        s_a, d_a = np.dot(rotation_matrix, self._acceleration)

        s_d, s_d_d, s_d_d_d = self._s, s_v, s_a  # Longitudinal
        c_d, c_d_d, c_d_dd = self._d, d_v, d_a  # Lateral
        target_area, speed, T = target_point, target_speed, time_horizon

        target_area += np.random.normal(0, 0.2)

        path = planner(s_d, s_d_d, s_d_d_d, c_d, c_d_d, c_d_dd, target_area, speed, T)

        self.planned_trajectory = np.array([[x, y] for x, y in zip(path[0].x, path[0].y)])

    def sampling_space(self):
        """
            The target sampling space (longitudinal speed and lateral offset)
        """
        current_speed = np.sqrt(self._velocity[0]**2 + self._velocity[1]**2)
        lateral_offsets = np.array([0 - 4, self._d, 0 + 4])
        min_speed = current_speed - 5 if current_speed > 5 else 0
        max_speed = current_speed + 5
        target_speeds = np.linspace(min_speed, max_speed, 10)

        return lateral_offsets, target_speeds

    @property
    def position(self) -> np.ndarray:
        """ Get all LaneBorders of this Lane """
        return self._position

    @property
    def velocity(self) -> np.ndarray:
        """ Get all LaneBorders of this Lane """
        return self._velocity
