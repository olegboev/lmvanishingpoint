import numpy as np

from optimization_task import OptimizationTask


class LMSolver:
    """
    A class representing a solver using the Levenberg-Marquardt algorithm for
    optimization problems.
    """

    def __init__(self, optimization_task: OptimizationTask,
                 loss_threshold: float = 1e-10, max_iterations: int = 100):
        """
        Initialize the LMSolver.

        Parameters:
            optimization_task (OptimizationTask): An instance of the
            OptimizationTask class.
            loss_threshold (float): Threshold for convergence based on loss
            difference.
            max_iterations (int): Maximum number of iterations for the solver.
        """
        self.optimization_task = optimization_task
        self.loss_threshold = loss_threshold
        self.max_iterations = max_iterations

    def solve(self) -> np.ndarray:
        """
        Solve the optimization problem using the Levenberg-Marquardt algorithm.

        Returns:
            np.ndarray: The optimized parameters.
        """
        lambda_ = None  # Initial damping factor

        # Initial loss
        residuals = self.optimization_task.residuals(
            self.optimization_task.parameters)
        loss_prev = np.dot(residuals, residuals)

        for iteration in range(self.max_iterations):
            parameters = self.optimization_task.parameters
            jacobian = self.optimization_task.jacobian()
            n = jacobian.T @ jacobian

            # Initialize damping factor if it's None
            if lambda_ is None:
                lambda_ = 1e-3 * np.mean(np.diag(n))

            # Calculate the step using the damped least squares solution
            step = -(np.linalg.pinv(n + np.eye(parameters.shape[0]) *
                                    lambda_) @ jacobian.T @
                     self.optimization_task.residuals(parameters))
            parameters += step
            residuals = self.optimization_task.residuals(parameters)
            loss = np.dot(residuals, residuals)

            # Update parameters and reduce damping if the new loss is lower
            if loss < loss_prev:
                lambda_ /= 10
                self.optimization_task.parameters = parameters
            else:
                lambda_ *= 10

            # Check for convergence based on loss difference
            if np.abs(loss - loss_prev) < self.loss_threshold:
                break
            loss_prev = loss

        return parameters
