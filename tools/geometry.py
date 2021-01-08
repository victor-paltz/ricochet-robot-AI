from typing import Tuple

import numpy as np


def angular_dist_modulo(a: float, b: float, angle: float = np.pi/2) -> float:
    """Angle between lignes with angles a and b, modulo the given angle"""
    return min((a-b) % angle, angle-(a-b) % angle)


def intersection(line1: Tuple[float, float], line2: Tuple[float, float]) -> Tuple[float, float]:
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return (x0, y0)
