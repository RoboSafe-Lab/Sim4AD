from enum import Enum


class DeathCause(Enum):
    OFF_ROAD = 0
    COLLISION = 1
    TIMEOUT = 2
    GOAL_REACHED = 3


class PositionNearbyAgent(Enum):
    CENTER_IN_FRONT = "front_ego"
    CENTER_BEHIND = "behind_ego"
    LEFT_IN_FRONT = "front_left"
    LEFT_BEHIND = "behind_left"
    RIGHT_IN_FRONT = "front_right"
    RIGHT_BEHIND = "behind_right"

