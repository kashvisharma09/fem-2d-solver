"""Constant Strain Triangle (CST) element.

Element DOF ordering: [u1, v1, u2, v2, u3, v3].
"""
import numpy as np


def signed_area(coords: np.ndarray) -> float:
    """Signed area of a triangle. Positive if nodes are counter-clockwise."""
    (x1, y1), (x2, y2), (x3, y3) = coords
    return 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def b_matrix(coords: np.ndarray):
    """Strain-displacement matrix B (3x6) and the element area.

    epsilon = B @ u_e is constant over the element, hence 'constant strain'.
    """
    (x1, y1), (x2, y2), (x3, y3) = coords
    A = signed_area(coords)
    if abs(A) < 1e-14:
        raise ValueError("Degenerate (zero-area) triangle")

    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

    B = (1.0 / (2.0 * A)) * np.array([
        [b1, 0., b2, 0., b3, 0.],
        [0., c1, 0., c2, 0., c3],
        [c1, b1, c2, b2, c3, b3],
    ])
    return B, abs(A)


def element_stiffness(coords: np.ndarray, D: np.ndarray, t: float) -> np.ndarray:
    """Element stiffness matrix Ke = t * A * B^T D B  (6x6)."""
    B, A = b_matrix(np.asarray(coords, dtype=float))
    return t * A * B.T @ D @ B