from abc import ABC, abstractmethod

import numpy as np


class OptimizationTask(ABC):
    """
    Abstract base class for optimization tasks.

    This class defines the interface for optimization tasks, including methods
    for calculating the Jacobian, residuals, and handling parameters.
    """

    def __init__(self):
        """
        Initialize an instance of OptimizationTask.

        This method serves as a placeholder for any common initialization logic
        required by subclasses.

        Parameters:
            None
        """
        pass

    @abstractmethod
    def jacobian(self) -> np.ndarray:
        """
        Abstract method to calculate the Jacobian matrix for the optimization
        task.
        Subclasses must implement this method to provide the Jacobian matrix
        specific to their optimization problem.

        Returns:
            np.ndarray: The Jacobian matrix.
        """
        pass

    @abstractmethod
    def residuals(self, parameters: np.ndarray) -> np.ndarray:
        """
        Abstract method to calculate residuals for the optimization task.
        Subclasses must implement this method to provide residuals specific to
        their optimization problem.

        Parameters:
            parameters (np.ndarray): Current parameters for the optimization
            task.

        Returns:
            np.ndarray: Residuals for each data point.
        """
        pass

    @property
    @abstractmethod
    def parameters(self) -> np.ndarray:
        """
        Abstract property to get the current parameters for the optimization
        task.

        Subclasses must implement this property to provide access to the
        current parameters.

        Returns:
            np.ndarray: Current parameters.
        """
        pass

    @parameters.setter
    @abstractmethod
    def parameters(self, parameters: np.ndarray):
        """
        Abstract property to set the current parameters for the optimization
        task.

        Subclasses must implement this property to allow setting new
        parameters.

        Parameters:
            parameters (np.ndarray): New parameters for the optimization task.

        Returns:
            None
        """
        pass
