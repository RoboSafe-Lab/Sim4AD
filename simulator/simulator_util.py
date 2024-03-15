from collections import defaultdict
from enum import Enum
from typing import Tuple

import numpy as np
from shapely import Point

from sim4ad.common_constants import MISSING_NEARBY_AGENT_VALUE
from sim4ad.opendrive import Lane
from simulator.policy_agent import PolicyAgent
from simulator.state_action import State


class DeathCause(Enum):
    OFF_ROAD = 0
    COLLISION = 1
    TIMEOUT = 2  # The simulation was over before the agent had the chance to complete. Does not mean the agent
    # took to long to reach the goal.
    GOAL_REACHED = 3
    TRUNCATED = 4  # The agent took to many steps to reach the goal.


class PositionNearbyAgent(Enum):
    CENTER_IN_FRONT = "front_ego"
    CENTER_BEHIND = "behind_ego"
    LEFT_IN_FRONT = "front_left"
    LEFT_BEHIND = "behind_left"
    RIGHT_IN_FRONT = "front_right"
    RIGHT_BEHIND = "behind_right"


def compute_distance_markings(state: State) -> Tuple[float, float]:
    """
    Compute the distance to the left and right lane markings.

    :param previous_state: The previous state.
    :param state: The current state.
    :return: The distance to the left and right lane markings.
    """

    lane = state.lane

    if lane is None:
        raise ValueError(f"No lane found at position {state.position} and heading {state.heading}.")

    ds_on_lane = lane.distance_at(state.position)

    # 1. find the slope of the line perpendicular to the lane through the agent

    position = state.position
    # Find the point on the boundary closest to the agent
    closest_point = lane.boundary.boundary.interpolate(lane.boundary.boundary.project(position))

    # We now want to find if the point is on the left or right side of the agent
    # We can do this by using the cross product of the vector v1 from (init_x,init_y,0) to the agent and
    # vector v2 from (init_x,init_y,0) to the closest point, where init_x, and init_y is the first point in the
    # of the midline of the lane. If the cross product is negative, then the point is on the right side of the agent,
    # if it is positive, then it is on the left side.
    v_init = lane.point_at(0)
    v_init = np.array([v_init[0], v_init[1], 0])

    v1 = np.array([state.position.x, state.position.y, 0]) - v_init
    v2 = np.array([closest_point.x, closest_point.y, 0]) - v_init
    cross_product = np.cross(v1, v2)
    if cross_product[2] < 0:  # Check the z-component of the cross product
        # The point is on the right side of the agent=
        distance_right_lane_marking = position.distance(closest_point)
        distance_left_lane_marking = lane.get_width_at(ds_on_lane) - distance_right_lane_marking
    else:
        # The point is on the left side of the agent
        distance_left_lane_marking = position.distance(closest_point)
        distance_right_lane_marking = lane.get_width_at(ds_on_lane) - distance_left_lane_marking

    assert distance_left_lane_marking + distance_right_lane_marking - lane.get_width_at(ds_on_lane) < 1e-6

    return distance_left_lane_marking, distance_right_lane_marking


def __find_perpendicular(lane: Lane, state: State, length=50) -> Tuple[Point, Point]:
    # We need to take the tangent as we want the slope (ration dy/dx) and not the heading
    ds_on_lane = lane.distance_at(state.position)
    m = -1 / np.tan(lane.get_heading_at(ds_on_lane))

    # 2. find the equation of the line to the lane through the agent: y = m * (x - x0) + y0
    y = lambda x: m * (x - state.position.x) + state.position.y

    # 3. find the points lane's width away from the agent on the left and right using
    # https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point

    if state.heading < 0:
        x_left = state.position.x + length / np.sqrt(1 + m ** 2)
        x_right = state.position.x - length / np.sqrt(1 + m ** 2)
    else:
        x_right = state.position.x + length / np.sqrt(1 + m ** 2)
        x_left = state.position.x - length / np.sqrt(1 + m ** 2)

    y_left = y(x_left)
    y_right = y(x_right)

    # Check if the slope is correct
    assert (y_right - y_left) / (x_right - x_left) - m < 1e-6

    # return A and B, the left and right points.
    return Point(x_left, y_left), Point(x_right, y_right)


def get_nearby_vehicles(agent: PolicyAgent, state: State, all_agents: dict):
    """
    TODO: we assume that there is only one lane, and not consider that vehicle may be in different lane groups,
    e.g., if lane changes group in front and the agent in front is in that lane instead.
    """

    # 1. We need the id of the lane where the vehicle is on
    lane = state.lane

    if lane is None:
        raise ValueError(f"No lane found at position {state.position} and heading {state.heading}.")

    # 2. We want the lanes to the left and right of the current one (as long as they have the same flow of motion)
    # TODO: in urban environments, is this an acceptable limitation, or should we also include vehicles from
    #   the other direction, as they may surpass, merge / cut in front of the vehicle?

    nearby_lanes = {k: None for k in ["left", "center", "right"]}

    nearby_lanes["left"], nearby_lanes["center"], nearby_lanes["right"] = lane.traversable_neighbours(
        return_lfr_order=True)

    # 3. We want to further divide the lanes into two parts, the one in front and the one behind the vehicle.
    # We will use the perpendicular line to the lane to divide the lanes into two parts.
    # perpendicular = np.array(left_point, right_point) TODO: check this is still updated
    perpendicular = __find_perpendicular(lane, state)

    nearby_vehicles_features = defaultdict(None)
    vehicles_nearby = defaultdict(None)

    for lane_position, nearby_lane in nearby_lanes.items():

        closest_vehicle_front = None
        closest_vehicle_behind = None

        min_distance_front = float("inf")
        min_distance_behind = float("inf")

        # Loop through all agents and check if they are in the lane
        if nearby_lane is not None:
            for nearby_agent_id, nearby_agent in all_agents.items():
                if nearby_agent_id == agent.agent_id:
                    continue

                nearby_agent_position = nearby_agent.state.position

                if nearby_lane.boundary.contains(nearby_agent_position):
                    # We now need to compute the relative position of the vehicle, whether it is in front or behind
                    # the agent, by computing the cross product of the AB vector and AP vector, where A-B are the
                    # left and right points of the perpendicular line, and P is the position of the nearby vehicle.
                    # If the cross product is positive, then the vehicle is in front of the agent, if is negative,
                    # then it is behind the agent.
                    AB = np.array(
                        [perpendicular[1].x - perpendicular[0].x, perpendicular[1].y - perpendicular[0].y, 0])
                    AP = np.array(
                        [nearby_agent_position.x - perpendicular[0].x, nearby_agent_position.y - perpendicular[0].y,
                         0])
                    cross_product = np.cross(AB, AP)

                    distance = state.position.distance(nearby_agent_position)

                    if cross_product[2] > 0:
                        if distance < min_distance_front:
                            min_distance_front = distance
                            closest_vehicle_front = nearby_agent
                    else:
                        if distance < min_distance_behind:
                            min_distance_behind = distance
                            closest_vehicle_behind = nearby_agent

        # For each lane, we want to store the closest vehicle in front and behind the agent
        if lane_position == "left":
            front = PositionNearbyAgent.LEFT_IN_FRONT
            behind = PositionNearbyAgent.LEFT_BEHIND
        elif lane_position == "center":
            front = PositionNearbyAgent.CENTER_IN_FRONT
            behind = PositionNearbyAgent.CENTER_BEHIND
        elif lane_position == "right":
            front = PositionNearbyAgent.RIGHT_IN_FRONT
            behind = PositionNearbyAgent.RIGHT_BEHIND

        nearby_vehicles_features[front] = get_vehicle_features(closest_vehicle_front, state)
        nearby_vehicles_features[behind] = get_vehicle_features(closest_vehicle_behind, state)

        vehicles_nearby[front] = closest_vehicle_front
        vehicles_nearby[behind] = closest_vehicle_behind

    return nearby_vehicles_features, vehicles_nearby


def collision_check(agent_state: State, nearby_vehicles: dict):
    """
    Check if the agent is colliding with a nearby vehicle.

    :param agent_state: The state of the agent.
    :param nearby_vehicles: Nearby vehicles.
    :return: True if the agent is colliding with any of the nearby vehicle, False otherwise.
    """

    for nearby_vehicle in nearby_vehicles.values():
        if nearby_vehicle is not None:
            if agent_state.bbox.overlaps(nearby_vehicle.state.bbox):
                return True
    return False


def get_vehicle_features(nearby_vehicle: PolicyAgent, state: State):
    """
    :param nearby_vehicle: The vehicle to get the features from, w.r.t the ego agent.
    :param state: The current state of the ego agent.
    """
    features = defaultdict(lambda: MISSING_NEARBY_AGENT_VALUE)

    if nearby_vehicle is not None:
        features["rel_dx"] = nearby_vehicle.state.position.x - state.position.x
        features["rel_dy"] = nearby_vehicle.state.position.y - state.position.y
        features["speed"] = nearby_vehicle.state.speed
        features["a"] = nearby_vehicle.state.acceleration
        features["heading"] = nearby_vehicle.state.heading
    return features
