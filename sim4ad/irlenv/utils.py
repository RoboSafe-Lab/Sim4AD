from __future__ import division, print_function

import importlib

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Point

EPSILON = 0.01


def constrain(x, a, b):
    return np.minimum(np.maximum(x, a), b)


def not_zero(x):
    if abs(x) > EPSILON:
        return x
    elif x > 0:
        return EPSILON
    else:
        return -EPSILON


def wrap_to_pi(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def point_in_rectangle(point, rect_min, rect_max):
    """
    Check if a point is inside a rectangle
    :param point: a point (x, y)
    :param rect_min: x_min, y_min
    :param rect_max: x_max, y_max
    """
    return rect_min[0] <= point[0] <= rect_max[0] and rect_min[1] <= point[1] <= rect_max[1]


def point_in_rotated_rectangle(point, center, length, width, angle):
    """
    Check if a point is inside a rotated rectangle
    :param point: a point
    :param center: rectangle center
    :param length: rectangle length
    :param width: rectangle width
    :param angle: rectangle angle [rad]
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return point_in_rectangle(ru, [-length / 2, -width / 2], [length / 2, width / 2])


def point_in_ellipse(point, center, angle, length, width):
    """
    Check if a point is inside an ellipse
    :param point: a point
    :param center: ellipse center
    :param angle: ellipse main axis angle
    :param length: ellipse big axis
    :param width: ellipse small axis
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.matrix([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return np.sum(np.square(ru / np.array([length, width]))) < 1


def rotated_rectangles_intersect(rect1, rect2):
    """
    Do two rotated rectangles intersect?
    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    """
    return has_corner_inside(rect1, rect2) or has_corner_inside(rect2, rect1)


def has_corner_inside(rect1, rect2):
    """
    Check if rect1 has a corner inside rect2
    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    """
    (c1, l1, w1, a1) = rect1
    (c2, l2, w2, a2) = rect2
    c1 = np.array(c1)
    l1v = np.array([l1 / 2, 0])
    w1v = np.array([0, w1 / 2])
    r1_points = np.array([[0, 0],
                          - l1v, l1v, -w1v, w1v,
                          - l1v - w1v, - l1v + w1v, + l1v - w1v, + l1v + w1v])
    c, s = np.cos(a1), np.sin(a1)
    r = np.array([[c, -s], [s, c]])
    rotated_r1_points = r.dot(r1_points.transpose()).transpose()
    return any([point_in_rotated_rectangle(c1 + np.squeeze(p), c2, l2, w2, a2) for p in rotated_r1_points])


def do_every(duration, timer):
    return duration < timer


def remap(v, x, y):
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def class_from_path(path):
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object


def compute_angle(point1, point2):
    """
    Compute the angle (in degrees) from horizontal for the line defined by two points.
    """
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    return np.degrees(np.arctan2(dy, dx))


def frenet2local(reference_line, s: float, d: float):
    """
    Convert Frenet coordinates (s, d) to local Cartesian coordinates on a curved road using Shapely.

    :param reference_line: the middle line of a lane
    :param s: Longitudinal distance along the path.
    :param d: Lateral offset from the path, positive to the left, negative to the right.
    :return: Tuple representing the Cartesian coordinates (x, y) on the curved road.
    """
    # Create a LineString from the path points
    path = reference_line

    # Interpolate the point at distance 's' along the path
    point_on_path = path.interpolate(s)

    # Compute the angle of the tangent at the interpolated point
    if s < path.length:
        # Compute tangent by looking ahead by a small distance
        look_ahead_distance = min(1.0, path.length - s)
        look_ahead_point = path.interpolate(s + look_ahead_distance)
        angle_degrees = compute_angle(point_on_path, look_ahead_point)
    else:
        # Handle the case where 's' is at the end of the path by looking backward
        look_back_point = path.interpolate(s - 1.0)
        angle_degrees = compute_angle(look_back_point, point_on_path)

    # Calculate the offset point by rotating and translating the original point
    # Note: Shapely's rotate function uses counter-clockwise rotation, so 'd' is positive to the left
    offset_point = translate(rotate(point_on_path, angle_degrees, origin=point_on_path, use_radians=False), d,
                             0)

    return offset_point.x, offset_point.y


def local2frenet(point: np.ndarray, reference_line):
    """Get the s and d values along the lane midline"""
    p = Point(point)
    s = reference_line.project(p)
    closest_point = reference_line.interpolate(s)

    # Compute tangent vector at the closest point
    if s < reference_line.length:
        look_ahead_distance = min(1.0, reference_line.length - s)
        look_ahead_point = reference_line.interpolate(s + look_ahead_distance)
        tangent_vector = np.array([look_ahead_point.x - closest_point.x, look_ahead_point.y - closest_point.y])
    else:
        look_back_point = reference_line.interpolate(s - 1.0)
        tangent_vector = np.array([closest_point.x - look_back_point.x, closest_point.y - look_back_point.y])

    # Compute vector from the closest point to p
    point_vector = np.array([p.x - closest_point.x, p.y - closest_point.y])

    # Cross product to determine the side
    cross_product = np.cross(tangent_vector, point_vector)
    side = np.sign(cross_product)

    d = p.distance(closest_point) * side

    return s, d
