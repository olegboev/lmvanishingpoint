import argparse

import cv2
import numpy as np

from lm_solver import LMSolver
from util import detect_line_segments
from vp_optimization_task import VanishingPointOptimizationTask


def main():
    parser = argparse.ArgumentParser(description="Vanishing Point Detection")
    parser.add_argument("--input", required=True,
                        help="Path to the input image")
    args = parser.parse_args()

    image_path = args.input
    image = cv2.imread(image_path)

    segments = detect_line_segments(image)
    task = VanishingPointOptimizationTask(segments)
    solver = LMSolver(task)
    vanishing_point = solver.solve()

    # Normalize vanishing point if necessary
    if not np.isclose(vanishing_point[2], 0.0):
        vanishing_point /= vanishing_point[2]

    print(f'Found vanishing point: {vanishing_point}')


if __name__ == "__main__":
    main()
