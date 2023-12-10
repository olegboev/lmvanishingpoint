from typing import List

import cv2
import numpy as np


def detect_line_segments(image: np.ndarray) -> List[np.ndarray]:
    """
    Detects line segments in the given image.

    Parameters:
        image (np.ndarray): Input image.

    Returns:
        List[np.ndarray]: List of line segments represented as numpy arrays.
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
    image_edges = cv2.Canny(image_gray, 50, 200)
    image_edges = cv2.dilate(image_edges, np.ones((3, 3), dtype=np.uint8))

    lines = cv2.HoughLinesP(image_edges, rho=1, theta=np.deg2rad(1),
                            threshold=300, minLineLength=100, maxLineGap=50)

    COSINE_THRESHOLD = 0.9
    vertical_dir = np.array([0, 1])

    segments = []
    if lines is not None:
        lines = np.squeeze(lines, axis=1)

        for x1, y1, x2, y2 in lines:
            dir = np.array([x1, y1]) - np.array([x2, y2])
            dir = dir / np.linalg.norm(dir)
            if abs(np.dot(dir, vertical_dir)) > COSINE_THRESHOLD:
                segments.append([[x1, y1, 1], [x2, y2, 1]])

    segments = [[np.array(segment[0]), np.array(segment[1])]
                for segment in segments]
    return segments


def segment_intersection(segment1: List, segment2: List) -> np.ndarray:
    """
    Finds the intersection of two line segments.

    Parameters:
        segment1 (list): Line segment represented as a list of 2 vertices,
        each vertex is np.ndarray of 3 elements (homogeneous representation).
        segment2 (list): Line segment represented as a list of 2 vertices,
        each vertex is np.ndarray of 3 elements (homogeneous representation).

    Returns:
        np.ndarray: Intersection point.
    """
    line1 = np.cross(segment1[0], segment1[1])
    line2 = np.cross(segment2[0], segment2[1])
    intersection = np.cross(line1, line2)
    return intersection


def line_homogeneous(line: List) -> np.ndarray:
    """
    Returns the homogeneous representation of a line.

    Parameters:
        line (list): Line represented as a list of 2 vertices, each vertex is
        np.ndarray of 2 elements.

    Returns:
        np.ndarray: Normalized homogeneous representation of a line.
    """
    # Homogeneous representation
    line_h = np.cross(np.hstack([line[0], 1]), np.hstack([line[1], 1]))
    if not np.allclose(line_h[2], 0):
        line_h = line_h / line_h[2]

    return line_h


def pairwise_intersections(lines: List) -> List:
    """
    Finds intersection points between each line pair.

    Parameters:
        lines (list): List of lines, where each line is represented
        as a list of 2 vertices, each vertex is np.ndarray of 3 elements
        (homogeneous representation).

    Returns:
        list: Intersection point coordinates as a list of numpy arrays.
    """
    intersections = []
    for i in range(len(lines)):
        line1 = lines[i]
        for j in range(i + 1, len(lines)):
            line2 = lines[j]
            intersection = segment_intersection(line1, line2)
            intersections.append(intersection)
    return intersections


def intersection_centroid(lines: List) -> np.ndarray:
    """
    Finds the centroid of intersection points between each line pair.

    Parameters:
        lines (list): List of lines, where each line is represented
        as a list of 2 vertices, each vertex is np.ndarray of 3 elements
        (homogeneous representation).

    Returns:
        np.ndarray: Point coordinates as numpy array.
    """
    intersections = pairwise_intersections(lines)

    # Check if the majority of intersection points are infinite
    intersections = np.array(intersections)
    infinite_points_idx = np.isclose(intersections[:, 2], 0.0)
    if np.sum(infinite_points_idx) > (intersections.shape[0] -
                                      np.sum(infinite_points_idx)):
        # The majority of intersection points are infinite
        centroid = np.mean(intersections[infinite_points_idx], axis=0)
    else:
        finite_intersections = intersections[
            np.logical_not(infinite_points_idx)]
        finite_intersections = (finite_intersections /
                                finite_intersections[:, 2:3])
        centroid = np.mean(finite_intersections, axis=0)

    return centroid


def segment_line_sine(segment: List, p: np.ndarray) -> float:
    """
    Returns the sine between the segment and a line from its center to point p.

    Parameters:
        segment (list): Line segment represented as a list of 2 vertices,
        each vertex is 2D point in homogeneous coordinates
        as np.ndarray of 3 elements.
        p (np.ndarray): Point coordinates.

    Returns:
        float: Sine value.
    """
    segment_line = np.cross(segment[0], segment[1])
    segment_line_normal = segment_line[:2]
    segment_line_normal = (segment_line_normal /
                           np.linalg.norm(segment_line_normal))

    segment_midpoint = (segment[0] + segment[1]) / 2
    aux_line = np.cross(p, segment_midpoint)
    aux_line_normal = aux_line[:2]
    aux_line_normal = aux_line_normal / np.linalg.norm(aux_line_normal)
    result = np.cross(segment_line_normal, aux_line_normal)
    return result
