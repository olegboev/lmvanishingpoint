from typing import List

import numpy as np

from util import intersection_centroid, segment_line_sine
from optimization_task import OptimizationTask


class VanishingPointOptimizationTask(OptimizationTask):
    """
    A class representing an optimization task for finding the vanishing point.
    """

    def __init__(self, segments: List[np.ndarray]):
        """
        Initialize the VanishingPointOptimizationTask.

        Parameters:
            segments (List[np.ndarray]): List of line segments for
            optimization.
        """
        self.segments = segments
        self.vp = intersection_centroid(segments)

    def residuals(self, parameters: np.ndarray) -> np.ndarray:
        """
        Calculate residuals for the optimization task.

        Parameters:
            parameters (np.ndarray): Current parameters (vanishing point).

        Returns:
            np.ndarray: Residuals for each line segment.
        """
        return np.array([segment_line_sine(s, parameters)
                         for s in self.segments])

    def jacobian(self) -> np.ndarray:
        """
        Calculate the Jacobian matrix for the optimization task using the
        central difference scheme.

        Returns:
            np.ndarray: Jacobian matrix.
        """
        jacobian_matrix = np.empty((len(self.segments), self.vp.shape[0]))

        for r in range(len(self.segments)):
            for x in range(self.vp.shape[0]):
                # Use the central difference scheme for numerical
                # differentiation
                # The step value used here is suggested in the book
                # "Multiple View Geometry in Computer Vision"
                # by R. Hartley and A. Zisserman.
                step = np.max([1e-6, np.abs(self.vp[x]) * 1e-4])
                # Create a vector with all zeros except step on x position
                delta = step * np.squeeze(np.eye(1, self.vp.shape[0], k=x))
                jacobian_matrix[r, x] = ((segment_line_sine(self.segments[r],
                                                            self.vp + delta) -
                                         segment_line_sine(self.segments[r],
                                                           self.vp - delta)) /
                                         (2 * step))
        return jacobian_matrix

    @property
    def parameters(self) -> np.ndarray:
        """
        Get the current parameters (vanishing point).

        Returns:
            np.ndarray: Current parameters (vanishing point).
        """
        return self.vp

    @parameters.setter
    def parameters(self, parameters: np.ndarray):
        """
        Set the current parameters (vanishing point).

        Parameters:
            parameters (np.ndarray): New parameters (vanishing point).
        """
        self.vp = parameters
